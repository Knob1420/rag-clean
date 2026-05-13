"""
BM25 检索对比评估脚本

对比三种查询工程策略在同一 query_string 格式下的效果：
1. raw: 原始 query 直接做 query_string 检索（不做关键词提取/boost）
2. keyword: 用 ChineseKeywordExtractor 提取关键词，构建带 boost + 同义词的 query_string
3. rewrite: 先 QueryUnderstanding + QueryRewrite，再用改写后的查询构建 query_string

关键：三种模式都用 query_string 格式发送给 ES，区别仅在于输入的查询字符串不同。
      这样才能公平对比查询工程策略本身的效果。

Usage:
    python scripts/eval_bm25_compare.py
    python scripts/eval_bm25_compare.py --limit 5 --top-k 25
    python scripts/eval_bm25_compare.py --queries scripts/queries.txt
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from core.retrieve.retrieval import RetrievalService, _sub_special_char
from core.retrieve.retrieval_models import RetrievedChunk, RetrievalOptions
from core.query_engineer.keyword_extractor import get_keyword_extractor
from core.query_engineer.query_understanding import get_query_understanding_service
from core.query_engineer.query_rewrite import get_query_rewrite_service
from core.query_engineer.synonym import get_synonym_lookup
from store import get_store

# 抑制 info 日志
logger.disable("core")

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "eval_bm25_compare"

VALID_MODES = {"raw", "keyword", "rewrite"}


@dataclass
class ChunkResult:
    """单个检索结果"""

    rank: int
    chunk_id: str
    doc_title: str
    chunk_type: str
    score: float
    content_preview: str
    content_length: int
    parent_id: Optional[str] = None


@dataclass
class ModeResult:
    """单个检索模式的结果"""

    mode: str
    query_used: str  # 实际用于检索的 query 描述
    query_string: str = ""  # 生成的 Lucene query_string
    chunks: List[ChunkResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class QueryEvalResult:
    """单个 query 的多模式对比结果"""

    original_query: str
    mode_results: Dict[str, ModeResult] = field(default_factory=dict)
    error: Optional[str] = None


class BM25CompareEvaluator:
    """BM25 检索对比评估器

    三种模式共享同一 query_string 格式（与 RetrievalService._build_bm25_query 一致）：
    - raw: 原始 query 直接作为 query_string（无关键词提取/boost）
    - keyword: 关键词提取 + 同义词扩展 + boost 归一化 + 近邻查询
    - rewrite: QueryUnderstanding + QueryRewrite 后的关键词提取 + boost
    """

    def __init__(
        self,
        modes: List[str],
        top_k: int = 25,
        dataset_ids: Optional[List[str]] = None,
        output_dir: str = str(OUTPUT_DIR),
    ):
        invalid = set(modes) - VALID_MODES
        if invalid:
            raise ValueError(f"无效的检索模式: {invalid}，可选: {VALID_MODES}")

        self.modes = modes
        self.top_k = top_k
        self.dataset_ids = dataset_ids

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.output_dir / f"eval_{self.eval_id}"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # ES client
        self._store = get_store()
        self._es = self._store.es
        from config import settings
        self._index = settings.es_index_chunks

        # 查询工程组件
        self.retrieval_service = RetrievalService()
        self.keyword_extractor = get_keyword_extractor()
        self.understanding_service = get_query_understanding_service()
        self.rewrite_service = get_query_rewrite_service()
        self.synonym_lookup = get_synonym_lookup()
        self.results: List[QueryEvalResult] = []

    # ================================================================
    # 公共：filter 条件构建
    # ================================================================

    def _build_filter_conditions(self) -> List[Dict]:
        """构建 ES bool filter 条件"""
        filters = [
            {"term": {"is_latest": True}},
            {"terms": {"chunk_type": ["child", "summary"]}},
        ]
        if self.dataset_ids:
            filters.append({"terms": {"dataset_id": self.dataset_ids}})
        return filters

    # ================================================================
    # raw: 原始 query 直接做 query_string（不做任何关键词提取/boost）
    # ================================================================

    def _build_raw_query_string(self, query: str) -> str:
        """raw 模式：原始 query 转义后直接作为 query_string"""
        return _sub_special_char(query)

    # ================================================================
    # keyword: 复用 RetrievalService._build_bm25_query 的逻辑
    # ================================================================

    def _build_keyword_query_string(self, query: str) -> str:
        """keyword 模式：直接复用 RetrievalService._build_bm25_query"""
        options = RetrievalOptions()
        if self.dataset_ids:
            options.dataset_ids = self.dataset_ids
        return self.retrieval_service._build_bm25_query(query, options)

    # ================================================================
    # rewrite: QueryUnderstanding + QueryRewrite → 构建合并的 query_string
    # ================================================================

    def _build_rewrite_query_string(
        self, query: str
    ) -> tuple[str, str]:
        """rewrite 模式：先理解+改写，再构建 query_string

        Returns:
            (query_string, query_used_description)
        """
        # Step 1: Query Understanding
        understanding = self.understanding_service.parse(query)

        # Step 2: 对每个子问句做 Query Rewrite，然后提取关键词构建 query_string
        all_query_strings = []
        queries_used = []

        for sub_q in understanding.sub_queries:
            rewritten = self.rewrite_service.rewrite(sub_q.query, intent=sub_q.intent)
            for rq in rewritten.rewritten_queries:
                queries_used.append(rq)
                # 对改写后的 query 构建 query_string
                options = RetrievalOptions()
                if self.dataset_ids:
                    options.dataset_ids = self.dataset_ids
                qs = self.retrieval_service._build_bm25_query(rq, options)
                all_query_strings.append(qs)

        # 合并多个 query_string：用 OR 连接，每个用括号包裹
        if all_query_strings:
            combined = " OR ".join(f"({qs})" for qs in all_query_strings)
        else:
            # 兜底：对原始 query 构建
            options = RetrievalOptions()
            if self.dataset_ids:
                options.dataset_ids = self.dataset_ids
            combined = self.retrieval_service._build_bm25_query(query, options)

        query_used = " | ".join(queries_used) if queries_used else query
        return combined, query_used

    # ================================================================
    # ES 执行：统一用 query_string 格式
    # ================================================================

    def _execute_query_string(
        self, query_string: str, top_k: int
    ) -> List[RetrievedChunk]:
        """执行 query_string 格式的 ES 查询"""
        dsl = {
            "query": {
                "bool": {
                    "filter": self._build_filter_conditions(),
                    "must": [
                        {
                            "query_string": {
                                "query": query_string,
                                "fields": ["doc_title^3", "content"],
                                "type": "best_fields",
                                "minimum_should_match": 1,
                            }
                        }
                    ],
                }
            },
            "size": top_k,
        }

        try:
            response = self._es.search(index=self._index, body=dsl)
        except Exception as e:
            logger.error(f"ES 查询失败: {e}")
            return []

        hits = response.get("hits", {}).get("hits", [])
        chunks = []
        for hit in hits:
            source = hit.get("_source", {})
            chunk = RetrievedChunk(
                chunk_id=source.get("chunk_id", ""),
                doc_id=source.get("doc_id", ""),
                content=source.get("content", ""),
                score=hit.get("_score", 0.0),
                doc_title=source.get("doc_title"),
                dataset_id=source.get("dataset_id"),
                chunk_type=source.get("chunk_type"),
                doc_hash=source.get("doc_hash"),
                parent_id=source.get("parent_id"),
            )
            chunks.append(chunk)
        return chunks

    def _chunks_to_results(self, chunks: List[RetrievedChunk]) -> List[ChunkResult]:
        """将 RetrievedChunk 列表转为 ChunkResult 列表"""
        results = []
        for i, c in enumerate(chunks):
            results.append(
                ChunkResult(
                    rank=i + 1,
                    chunk_id=c.chunk_id,
                    doc_title=c.doc_title or "",
                    chunk_type=c.chunk_type or "",
                    score=round(c.score, 4) if c.score else 0.0,
                    content_preview=c.content[:200],
                    content_length=len(c.content),
                    parent_id=c.parent_id,
                )
            )
        return results

    # ================================================================
    # 三种模式的执行方法
    # ================================================================

    def _run_raw_bm25(self, query: str) -> ModeResult:
        """raw: 原始 query 直接 query_string"""
        t0 = time.time()
        try:
            qs = self._build_raw_query_string(query)
            chunks = self._execute_query_string(qs, self.top_k)
            duration = time.time() - t0
            return ModeResult(
                mode="raw",
                query_used=query,
                query_string=qs,
                chunks=self._chunks_to_results(chunks),
                duration_seconds=round(duration, 3),
            )
        except Exception as e:
            return ModeResult(mode="raw", query_used=query, error=str(e))

    def _run_keyword_bm25(self, query: str) -> ModeResult:
        """keyword: 关键词提取 + boost + 近邻 query_string"""
        t0 = time.time()
        try:
            qs = self._build_keyword_query_string(query)
            chunks = self._execute_query_string(qs, self.top_k)
            duration = time.time() - t0
            # 构建查询描述
            weighted_keywords = self.keyword_extractor.extract(query)
            kw_desc = " ".join(f"{kw}({w:.1f})" for kw, w in weighted_keywords[:8])
            return ModeResult(
                mode="keyword",
                query_used=f"keywords: {kw_desc}",
                query_string=qs,
                chunks=self._chunks_to_results(chunks),
                duration_seconds=round(duration, 3),
            )
        except Exception as e:
            return ModeResult(mode="keyword", query_used=query, error=str(e))

    def _run_rewrite_bm25(self, query: str) -> ModeResult:
        """rewrite: QueryUnderstanding + QueryRewrite + keyword query_string"""
        t0 = time.time()
        try:
            qs, query_used = self._build_rewrite_query_string(query)
            chunks = self._execute_query_string(qs, self.top_k)
            duration = time.time() - t0
            return ModeResult(
                mode="rewrite",
                query_used=query_used,
                query_string=qs,
                chunks=self._chunks_to_results(chunks),
                duration_seconds=round(duration, 3),
            )
        except Exception as e:
            return ModeResult(mode="rewrite", query_used=query, error=str(e))

    def _run_single_mode(self, query: str, mode: str) -> ModeResult:
        """执行单模式检索"""
        if mode == "raw":
            return self._run_raw_bm25(query)
        elif mode == "keyword":
            return self._run_keyword_bm25(query)
        elif mode == "rewrite":
            return self._run_rewrite_bm25(query)
        else:
            return ModeResult(mode=mode, query_used=query, error=f"未知模式: {mode}")

    # ================================================================
    # 批量评估 + 报告
    # ================================================================

    def evaluate(self, query: str) -> QueryEvalResult:
        """评估单个 query 的所有模式"""
        result = QueryEvalResult(original_query=query)

        for mode in self.modes:
            mode_result = self._run_single_mode(query, mode)
            result.mode_results[mode] = mode_result

        self.results.append(result)
        return result

    def batch_eval(self, queries: List[str]) -> List[QueryEvalResult]:
        """批量评估"""
        results = []
        for i, query in enumerate(queries):
            print(f"\n{'#' * 60}")
            print(f"# [{i + 1}/{len(queries)}] {query[:60]}")
            print(f"{'#' * 60}")

            result = self.evaluate(query)
            results.append(result)
            self._save_result(result)

            # 打印摘要
            if result.error:
                print(f"  [ERROR] {result.error[:100]}")
            else:
                for mode, mr in result.mode_results.items():
                    if mr.error:
                        print(f"  [{mode}] ERROR: {mr.error[:80]}")
                    else:
                        qs_preview = mr.query_string[:60] if mr.query_string else ""
                        print(
                            f"  [{mode}] {len(mr.chunks)} chunks, "
                            f"{mr.duration_seconds:.2f}s, "
                            f"QS: {qs_preview}..."
                        )

            time.sleep(0.3)

        return results

    def _save_result(self, result: QueryEvalResult):
        """保存单个结果为 JSON"""
        safe_query = (
            result.original_query.replace("/", "_").replace(" ", "_").replace("\n", "_")[:40]
        )
        filename = f"query_{safe_query}_{self.eval_id}.json"
        filepath = self.eval_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

        print(f"  [Output] {filepath.name}")

    def generate_report(self) -> Dict:
        """生成评估报告（JSON + TXT）"""
        total = len(self.results)

        # 统计
        mode_stats: Dict[str, Dict[str, Any]] = {}
        for mode in self.modes:
            mode_results = [r.mode_results.get(mode) for r in self.results]
            successes = [m for m in mode_results if m and not m.error]
            errors = [m for m in mode_results if m and m.error]
            avg_chunks = (
                sum(len(m.chunks) for m in successes) / len(successes)
                if successes
                else 0
            )
            avg_time = (
                sum(m.duration_seconds for m in successes) / len(successes)
                if successes
                else 0
            )
            mode_stats[mode] = {
                "total": total,
                "successes": len(successes),
                "errors": len(errors),
                "avg_chunks": round(avg_chunks, 1),
                "avg_time": round(avg_time, 3),
            }

        report = {
            "eval_id": self.eval_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "modes": self.modes,
                "top_k": self.top_k,
                "dataset_ids": self.dataset_ids,
                "query_format": "query_string",
                "fields": ["doc_title^3", "content"],
            },
            "mode_statistics": mode_stats,
            "total_queries": total,
            "queries": [],
        }

        for result in self.results:
            qr: Dict[str, Any] = {"query": result.original_query, "error": result.error}
            for mode, mr in result.mode_results.items():
                qr[mode] = {
                    "query_used": mr.query_used,
                    "query_string": mr.query_string[:300] if mr.query_string else "",
                    "chunks": len(mr.chunks) if not mr.error else None,
                    "duration": mr.duration_seconds if not mr.error else None,
                    "error": mr.error,
                    "top_chunks": [
                        {
                            "rank": c.rank,
                            "chunk_id": c.chunk_id,
                            "doc_title": c.doc_title,
                            "score": c.score,
                            "content_preview": c.content_preview[:150],
                        }
                        for c in (mr.chunks[:15] if not mr.error else [])
                    ],
                }
            report["queries"].append(qr)

        # 保存 JSON 报告
        report_path = self.eval_dir / "eval_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 生成人类可读 TXT 汇总
        summary_path = self.eval_dir / "eval_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("BM25 检索对比评估结果汇总（query_string 格式）\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"评估 ID: {self.eval_id}\n")
            f.write(f"查询格式: query_string (best_fields)\n")
            f.write(f"字段: doc_title^3, content\n")
            f.write(f"对比模式: {', '.join(self.modes)}\n")
            f.write(f"Top-K: {self.top_k}\n")
            f.write(f"总问题数: {total}\n\n")

            # 模式统计
            f.write("[模式统计]\n")
            for mode, stats in mode_stats.items():
                f.write(
                    f"  {mode}: {stats['successes']}/{stats['total']} 成功, "
                    f"avg {stats['avg_chunks']:.1f} chunks, "
                    f"avg {stats['avg_time']:.3f}s\n"
                )
            f.write("\n")

            # 逐 query 输出
            for i, result in enumerate(self.results, 1):
                f.write(f"{'=' * 70}\n")
                f.write(f"Q{i}: {result.original_query}\n")
                f.write(f"{'=' * 70}\n")

                if result.error:
                    f.write(f"[错误] {result.error}\n\n")
                    continue

                for mode in self.modes:
                    mr = result.mode_results.get(mode)
                    if not mr:
                        continue
                    if mr.error:
                        f.write(f"\n--- [{mode}] ERROR: {mr.error} ---\n\n")
                        continue

                    f.write(
                        f"\n--- [{mode}] query='{mr.query_used[:60]}' | "
                        f"{len(mr.chunks)} chunks | "
                        f"{mr.duration_seconds:.3f}s ---\n"
                    )
                    f.write(f"    QS: {mr.query_string[:120]}\n")
                    for c in mr.chunks:
                        preview = c.content_preview[:100].replace("\n", " ")
                        f.write(
                            f"  [{c.rank:2d}] {c.doc_title[:40]} "
                            f"(score={c.score:.4f}, type={c.chunk_type})\n"
                        )
                        f.write(f"       {preview}...\n")

                # 对比分析：chunk 重叠度
                if len(self.modes) >= 2:
                    f.write("\n  [重叠分析]\n")
                    chunk_ids_per_mode: Dict[str, Set[str]] = {}
                    for mode in self.modes:
                        mr = result.mode_results.get(mode)
                        if mr and not mr.error:
                            chunk_ids_per_mode[mode] = {c.chunk_id for c in mr.chunks}
                        else:
                            chunk_ids_per_mode[mode] = set()

                    for j, m1 in enumerate(self.modes):
                        for m2 in self.modes[j + 1 :]:
                            ids1 = chunk_ids_per_mode.get(m1, set())
                            ids2 = chunk_ids_per_mode.get(m2, set())
                            overlap = ids1 & ids2
                            union = ids1 | ids2
                            jaccard = len(overlap) / len(union) if union else 0
                            f.write(
                                f"    {m1} ∩ {m2}: "
                                f"{len(overlap)}/{len(union)} "
                                f"(Jaccard={jaccard:.3f})\n"
                            )

                f.write("\n")

        print(f"\n{'=' * 60}")
        print(f"[评估完成]")
        print(f"  总问题数: {total}")
        for mode, stats in mode_stats.items():
            print(
                f"  [{mode}] {stats['successes']}/{stats['total']} 成功, "
                f"avg {stats['avg_chunks']:.1f} chunks, "
                f"avg {stats['avg_time']:.3f}s"
            )
        print(f"  Report:  {report_path}")
        print(f"  Summary: {summary_path}")
        print(f"{'=' * 60}")

        return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="BM25 检索对比评估工具")
    parser.add_argument(
        "--queries",
        "-q",
        default=str(Path(__file__).parent / "queries.txt"),
        help="查询文件路径（每行一个问题）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(OUTPUT_DIR),
        help="输出目录",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=25,
        help="检索返回数量（默认 25）",
    )
    parser.add_argument(
        "--modes",
        "-m",
        default="raw,keyword,rewrite",
        help="对比模式，逗号分隔（默认 raw,keyword,rewrite）",
    )
    parser.add_argument(
        "--dataset-ids",
        default=None,
        help="按数据集筛选，逗号分隔",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="限制查询数量（0=全部）",
    )

    args = parser.parse_args()

    # 解析模式
    modes = [m.strip() for m in args.modes.split(",")]
    invalid = set(modes) - VALID_MODES
    if invalid:
        print(f"无效的检索模式: {invalid}，可选: {VALID_MODES}")
        sys.exit(1)

    # 解析 dataset_ids
    dataset_ids = None
    if args.dataset_ids:
        dataset_ids = [d.strip() for d in args.dataset_ids.split(",")]

    # 加载查询
    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"查询文件不存在: {queries_path}")
        sys.exit(1)

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    if args.limit > 0:
        queries = queries[: args.limit]

    print(f"加载了 {len(queries)} 个查询")
    print(f"Modes: {modes}")
    print(f"Top-K: {args.top_k}")
    print(f"Queries file: {queries_path}")
    print(f"Output dir: {args.output_dir}")

    evaluator = BM25CompareEvaluator(
        modes=modes,
        top_k=args.top_k,
        dataset_ids=dataset_ids,
        output_dir=args.output_dir,
    )

    evaluator.batch_eval(queries)
    evaluator.generate_report()


if __name__ == "__main__":
    main()
