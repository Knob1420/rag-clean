"""
Pipeline 对比评估脚本

3 种 pipeline 配置对比：
1. raw:        原始 query → hybrid（无 keyword/HyDE，无 rewrite，无 understand）
2. kw_hyde:    keyword 增强 BM25 + HyDE 向量 → hybrid（无 rewrite，无 understand）
3. full:       完整 pipeline（understand + rewrite + hybrid）

每种模式记录：bm25 chunks、vector chunks、hybrid chunks、耗时、答案

Usage:
    python scripts/eval_pipeline_compare.py
    python scripts/eval_pipeline_compare.py --limit 5
    python scripts/eval_pipeline_compare.py --modes raw,kw_hyde,full
    python scripts/eval_pipeline_compare.py --top-k 20 --rerank-top-k 10
    python scripts/eval_pipeline_compare.py --no-generation
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from core.pipeline.rag_pipeline import RAGPipeline
from core.retrieve.retrieval import RetrievalService
from core.retrieve.retrieval_models import RetrievedChunk, RetrievalOptions

# 抑制 info 日志
logger.disable("core")

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "eval_pipeline_compare"

VALID_MODES = {"raw", "kw_hyde", "full"}


# ── 数据结构 ──────────────────────────────────────────


@dataclass
class ChunkInfo:
    """单个 chunk 摘要"""
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
    """单个 pipeline 模式的结果"""

    mode: str
    query: str
    # 子问句 / 重写结果
    sub_queries: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)
    rewritten_queries: List[str] = field(default_factory=list)
    # BM25 / Vector / Hybrid chunks
    bm25_chunks: List[ChunkInfo] = field(default_factory=list)
    vector_chunks: List[ChunkInfo] = field(default_factory=list)
    hybrid_chunks: List[ChunkInfo] = field(default_factory=list)
    # 统计
    bm25_count: int = 0
    vector_count: int = 0
    hybrid_count: int = 0
    # 答案
    answer: Optional[str] = None
    answer_preview: Optional[str] = None
    generation_tokens: int = 0
    # 耗时
    timing: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class QueryEvalResult:
    """单个 query 的多模式对比结果"""

    query: str
    mode_results: Dict[str, ModeResult] = field(default_factory=dict)
    error: Optional[str] = None


# ── 辅助函数 ──────────────────────────────────────────


def _chunks_to_infos(chunks: List[RetrievedChunk], max_preview: int = 200) -> List[ChunkInfo]:
    """RetrievedChunk 列表 → ChunkInfo 列表"""
    infos = []
    for i, c in enumerate(chunks):
        infos.append(
            ChunkInfo(
                rank=i + 1,
                chunk_id=c.chunk_id,
                doc_title=c.doc_title or "",
                chunk_type=c.chunk_type or "",
                score=round(c.score, 4) if c.score else 0.0,
                content_preview=c.content[:max_preview],
                content_length=len(c.content),
                parent_id=c.parent_id,
            )
        )
    return infos


def _merge_dedup_chunks(
    per_query_chunks: Dict[str, List[RetrievedChunk]],
) -> List[ChunkInfo]:
    """合并多个 query 的 chunks 并去重，返回 ChunkInfo 列表"""
    seen = set()
    merged = []
    for _, chunks in per_query_chunks.items():
        for c in chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                merged.append(c)
    return _chunks_to_infos(merged)


# ── 评估器 ──────────────────────────────────────────


class PipelineCompareEvaluator:
    """Pipeline 对比评估器"""

    def __init__(
        self,
        modes: List[str],
        top_k: int = 20,
        rerank_top_k: int = 10,
        use_generation: bool = True,
        dataset_ids: Optional[List[str]] = None,
        output_dir: str = str(OUTPUT_DIR),
    ):
        invalid = set(modes) - VALID_MODES
        if invalid:
            raise ValueError(f"无效的模式: {invalid}，可选: {VALID_MODES}")

        self.modes = modes
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.use_generation = use_generation
        self.dataset_ids = dataset_ids

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.output_dir / f"eval_{self.eval_id}"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline = RAGPipeline()
        self.retrieval = RetrievalService()
        self.results: List[QueryEvalResult] = []

    # ── 模式 1: raw hybrid（无 keyword 增强、无 HyDE、无 rewrite、无 understand）──

    def _run_raw(self, query: str) -> ModeResult:
        """raw: 直接用原始 query 做 hybrid 检索"""
        t0 = time.time()

        options = RetrievalOptions(
            top_k=self.top_k,
            use_rerank=False,
            use_hyde=False,
            dataset_ids=self.dataset_ids,
        )

        # 预处理：构建 query_string + query_vector
        query_string = self.retrieval._build_bm25_query(query, options)
        from core.client.embedder import encode
        query_vector = encode(query)

        # 一次 hybrid 拿到 bm25 / vector / hybrid 三组结果 + 细粒度 timing
        hybrid_chunks, bm25_chunks, vector_chunks, hybrid_timing = self.retrieval._hybrid_search(
            query_string, query_vector, options, return_intermediates=True
        )

        # 可选 rerank（输入截断到 rerank_top_k，与 pipeline 行为一致）
        t_rerank = 0.0
        rerank_input = hybrid_chunks[: self.rerank_top_k] if self.rerank_top_k else hybrid_chunks
        if self.rerank_top_k and rerank_input:
            t_rr = time.time()
            rerank_opts = RetrievalOptions(
                top_k=self.top_k,
                use_rerank=True,
                rerank_top_k=self.rerank_top_k,
                dataset_ids=self.dataset_ids,
            )
            from core.query_engineer.rerank_query import build_rerank_query
            rerank_query = build_rerank_query(query)
            hybrid_chunks = self.retrieval._rerank(rerank_query, rerank_input, rerank_opts)
            t_rerank = time.time() - t_rr

        # 可选 generation
        t_gen = 0.0
        answer = None
        if self.use_generation and hybrid_chunks:
            t_g = time.time()
            answer = self._generate_answer(query, hybrid_chunks)
            t_gen = time.time() - t_g

        timing = {
            "embedding": hybrid_timing.get("embedding", 0),
            "hyde": hybrid_timing.get("hyde", 0),
            "bm25": hybrid_timing.get("bm25", 0),
            "vector": hybrid_timing.get("vector", 0),
            "rrf": hybrid_timing.get("rrf", 0),
            "rerank": round(t_rerank, 3),
            "generation": round(t_gen, 3),
            "total": round(time.time() - t0, 3),
        }

        return ModeResult(
            mode="raw",
            query=query,
            bm25_chunks=_chunks_to_infos(bm25_chunks),
            vector_chunks=_chunks_to_infos(vector_chunks),
            hybrid_chunks=_chunks_to_infos(hybrid_chunks),
            bm25_count=len(bm25_chunks),
            vector_count=len(vector_chunks),
            hybrid_count=len(hybrid_chunks),
            answer=answer,
            answer_preview=answer[:300] if answer else None,
            generation_tokens=0,
            timing=timing,
            duration_seconds=time.time() - t0,
        )

    # ── 模式 2: keyword + HyDE hybrid（无 rewrite、无 understand）──

    def _run_kw_hyde(self, query: str) -> ModeResult:
        """kw_hyde: keyword 增强 BM25 + HyDE 向量 → hybrid"""
        t0 = time.time()

        # 使用 pipeline 但关闭 understand 和 rewrite
        result = self.pipeline.run(
            query=query,
            top_k=self.top_k,
            use_understand=False,
            use_rewrite=False,
            use_rerank=True,
            rerank_top_k=self.rerank_top_k,
            use_generation=False,
            dataset_ids=self.dataset_ids,
        )

        timing = {k: round(v, 3) for k, v in result.timing.items()}

        # 提取 BM25 / Vector 原始结果
        bm25_infos = _merge_dedup_chunks(result.per_query_bm25_chunks or {})
        vector_infos = _merge_dedup_chunks(result.per_query_vector_chunks or {})

        # 可选 generation
        t_gen = 0.0
        answer = None
        if self.use_generation and result.chunks:
            t_g = time.time()
            answer = self._generate_answer(query, result.chunks)
            t_gen = time.time() - t_g

        timing["generation"] = round(t_gen, 3)

        return ModeResult(
            mode="kw_hyde",
            query=query,
            rewritten_queries=result.rewritten_queries,
            bm25_chunks=bm25_infos,
            vector_chunks=vector_infos,
            hybrid_chunks=_chunks_to_infos(result.chunks),
            bm25_count=len(bm25_infos),
            vector_count=len(vector_infos),
            hybrid_count=len(result.chunks),
            answer=answer,
            answer_preview=answer[:300] if answer else None,
            generation_tokens=0,
            timing=timing,
            duration_seconds=time.time() - t0,
        )

    # ── 模式 3: full pipeline（understand + rewrite + hybrid）──

    def _run_full(self, query: str) -> ModeResult:
        """full: 完整 pipeline（understand + rewrite + hybrid）"""
        t0 = time.time()

        result = self.pipeline.run(
            query=query,
            top_k=self.top_k,
            use_understand=True,
            use_rewrite=True,
            use_rerank=True,
            rerank_top_k=self.rerank_top_k,
            use_generation=False,
            dataset_ids=self.dataset_ids,
        )

        timing = {k: round(v, 3) for k, v in result.timing.items()}

        # 提取 sub_queries / intents
        sub_queries = []
        intents = []
        if result.understanding_result:
            sub_queries = [sq.query for sq in result.understanding_result.sub_queries]
            intents = [sq.intent for sq in result.understanding_result.sub_queries]

        # 提取 BM25 / Vector 原始结果
        bm25_infos = _merge_dedup_chunks(result.per_query_bm25_chunks or {})
        vector_infos = _merge_dedup_chunks(result.per_query_vector_chunks or {})

        # 可选 generation
        t_gen = 0.0
        answer = None
        if self.use_generation and result.chunks:
            t_g = time.time()
            answer = self._generate_answer(query, result.chunks)
            t_gen = time.time() - t_g

        timing["generation"] = round(t_gen, 3)

        return ModeResult(
            mode="full",
            query=query,
            sub_queries=sub_queries,
            intents=intents,
            rewritten_queries=result.rewritten_queries,
            bm25_chunks=bm25_infos,
            vector_chunks=vector_infos,
            hybrid_chunks=_chunks_to_infos(result.chunks),
            bm25_count=len(bm25_infos),
            vector_count=len(vector_infos),
            hybrid_count=len(result.chunks),
            answer=answer,
            answer_preview=answer[:300] if answer else None,
            generation_tokens=0,
            timing=timing,
            duration_seconds=time.time() - t0,
        )

    def _generate_answer(self, query: str, chunks: List[RetrievedChunk]) -> Optional[str]:
        """调用 generation 生成答案"""
        try:
            from core.generation.generation import get_generation_service
            gen_svc = get_generation_service()
            answer, _ = gen_svc.generate(query=query, chunks=chunks)
            return answer
        except Exception as e:
            logger.warning(f"Generation 失败: {e}")
            return None

    # ── 主入口 ──────────────────────────────────────────

    def _run_single_mode(self, query: str, mode: str) -> ModeResult:
        """执行单模式评估"""
        if mode == "raw":
            return self._run_raw(query)
        elif mode == "kw_hyde":
            return self._run_kw_hyde(query)
        elif mode == "full":
            return self._run_full(query)
        else:
            return ModeResult(mode=mode, query=query, error=f"未知模式: {mode}")

    def evaluate(self, query: str) -> QueryEvalResult:
        """评估单个 query 的所有模式"""
        result = QueryEvalResult(query=query)

        for mode in self.modes:
            try:
                mode_result = self._run_single_mode(query, mode)
            except Exception as e:
                mode_result = ModeResult(mode=mode, query=query, error=str(e))
            result.mode_results[mode] = mode_result

        self.results.append(result)
        return result

    def batch_eval(self, queries: List[str]) -> List[QueryEvalResult]:
        """批量评估"""
        results = []
        for i, query in enumerate(queries):
            print(f"\n{'#'*60}")
            print(f"# [{i+1}/{len(queries)}] {query[:60]}")
            print(f"{'#'*60}")

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
                        print(
                            f"  [{mode}] bm25={mr.bm25_count} vec={mr.vector_count} "
                            f"hybrid={mr.hybrid_count} | {mr.duration_seconds:.2f}s"
                        )

            time.sleep(0.5)

        return results

    def _save_result(self, result: QueryEvalResult):
        """保存单个结果为 JSON"""
        safe_query = (
            result.query.replace("/", "_").replace(" ", "_").replace("\n", "_")[:40]
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

            avg_bm25 = sum(m.bm25_count for m in successes) / len(successes) if successes else 0
            avg_vector = sum(m.vector_count for m in successes) / len(successes) if successes else 0
            avg_hybrid = sum(m.hybrid_count for m in successes) / len(successes) if successes else 0
            avg_time = sum(m.duration_seconds for m in successes) / len(successes) if successes else 0

            mode_stats[mode] = {
                "total": total,
                "successes": len(successes),
                "errors": len(errors),
                "avg_bm25": round(avg_bm25, 1),
                "avg_vector": round(avg_vector, 1),
                "avg_hybrid": round(avg_hybrid, 1),
                "avg_time": round(avg_time, 3),
            }

        report = {
            "eval_id": self.eval_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "modes": self.modes,
                "top_k": self.top_k,
                "rerank_top_k": self.rerank_top_k,
                "use_generation": self.use_generation,
                "dataset_ids": self.dataset_ids,
            },
            "mode_statistics": mode_stats,
            "total_queries": total,
            "queries": [],
        }

        for result in self.results:
            qr: Dict[str, Any] = {"query": result.query, "error": result.error}
            for mode, mr in result.mode_results.items():
                qr[mode] = {
                    "sub_queries": mr.sub_queries,
                    "intents": mr.intents,
                    "rewritten_queries": mr.rewritten_queries,
                    "bm25_count": mr.bm25_count,
                    "vector_count": mr.vector_count,
                    "hybrid_count": mr.hybrid_count,
                    "duration": mr.duration_seconds,
                    "timing": mr.timing,
                    "error": mr.error,
                    "answer_preview": mr.answer_preview,
                    "top_hybrid_chunks": [
                        {
                            "rank": c.rank,
                            "chunk_id": c.chunk_id,
                            "doc_title": c.doc_title,
                            "score": c.score,
                            "content_preview": c.content_preview[:150],
                        }
                        for c in mr.hybrid_chunks[:15]
                    ],
                    "top_bm25_chunks": [
                        {
                            "rank": c.rank,
                            "doc_title": c.doc_title,
                            "score": c.score,
                            "content_preview": c.content_preview[:100],
                        }
                        for c in mr.bm25_chunks[:10]
                    ],
                    "top_vector_chunks": [
                        {
                            "rank": c.rank,
                            "doc_title": c.doc_title,
                            "score": c.score,
                            "content_preview": c.content_preview[:100],
                        }
                        for c in mr.vector_chunks[:10]
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
            f.write("Pipeline 对比评估结果汇总\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"评估 ID: {self.eval_id}\n")
            f.write(f"模式: {', '.join(self.modes)}\n")
            f.write(f"Top-K: {self.top_k}\n")
            f.write(f"Rerank Top-K: {self.rerank_top_k}\n")
            f.write(f"Generation: {self.use_generation}\n")
            f.write(f"总问题数: {total}\n\n")

            # 模式统计
            f.write("[模式统计]\n")
            for mode, stats in mode_stats.items():
                f.write(
                    f"  {mode}: {stats['successes']}/{stats['total']} 成功, "
                    f"avg bm25={stats['avg_bm25']:.0f} vec={stats['avg_vector']:.0f} "
                    f"hybrid={stats['avg_hybrid']:.0f}, "
                    f"avg {stats['avg_time']:.3f}s\n"
                )
            f.write("\n")

            # 逐 query 输出
            for i, result in enumerate(self.results, 1):
                f.write(f"{'='*70}\n")
                f.write(f"Q{i}: {result.query}\n")
                f.write(f"{'='*70}\n")

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
                        f"\n--- [{mode}] bm25={mr.bm25_count} vec={mr.vector_count} "
                        f"hybrid={mr.hybrid_count} | {mr.duration_seconds:.3f}s ---\n"
                    )

                    # sub_queries / rewritten
                    if mr.sub_queries:
                        f.write(f"  子问句: {mr.sub_queries}\n")
                        f.write(f"  意图: {mr.intents}\n")
                    if mr.rewritten_queries:
                        f.write(f"  重写: {mr.rewritten_queries}\n")

                    # hybrid top chunks
                    for c in mr.hybrid_chunks[:5]:
                        preview = c.content_preview[:80].replace("\n", " ")
                        f.write(
                            f"  [{c.rank:2d}] {c.doc_title[:40]} "
                            f"(score={c.score:.4f}, type={c.chunk_type})\n"
                        )
                        f.write(f"       {preview}...\n")

                    # 答案
                    if mr.answer_preview:
                        f.write(f"  [答案] {mr.answer_preview[:200]}...\n")

                f.write("\n")

        print(f"\n{'='*60}")
        print(f"[评估完成]")
        print(f"  总问题数: {total}")
        for mode, stats in mode_stats.items():
            print(
                f"  [{mode}] {stats['successes']}/{stats['total']} 成功, "
                f"avg bm25={stats['avg_bm25']:.0f} vec={stats['avg_vector']:.0f} "
                f"hybrid={stats['avg_hybrid']:.0f}, "
                f"avg {stats['avg_time']:.3f}s"
            )
        print(f"  Report:  {report_path}")
        print(f"  Summary: {summary_path}")
        print(f"{'='*60}")

        return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline 对比评估工具")
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
        default=20,
        help="检索返回数量（默认 20）",
    )
    parser.add_argument(
        "--rerank-top-k",
        "-r",
        type=int,
        default=10,
        help="Rerank 后保留数量（默认 10）",
    )
    parser.add_argument(
        "--modes",
        "-m",
        default="raw,kw_hyde,full",
        help="对比模式，逗号分隔（默认 raw,kw_hyde,full）",
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="跳过 Generation（只测检索）",
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
        print(f"无效的模式: {invalid}，可选: {VALID_MODES}")
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
    print(f"Rerank Top-K: {args.rerank_top_k}")
    print(f"Generation: {not args.no_generation}")
    print(f"Queries file: {queries_path}")
    print(f"Output dir: {args.output_dir}")

    evaluator = PipelineCompareEvaluator(
        modes=modes,
        top_k=args.top_k,
        rerank_top_k=args.rerank_top_k,
        use_generation=not args.no_generation,
        dataset_ids=dataset_ids,
        output_dir=args.output_dir,
    )

    evaluator.batch_eval(queries)
    evaluator.generate_report()


if __name__ == "__main__":
    main()
