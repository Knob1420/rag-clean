"""
Vector Retrieval Only 评估脚本

跳过 query transform，只走 vector/bm25/hybrid/hyde 检索，可选 rerank + parent expand。
用于单独评估检索效果，输出各模式 top-k 结果方便人工判定相关性。

模式说明:
- vector: 纯向量检索
- bm25:   纯 BM25 检索
- hybrid:  混合检索（BM25 + 向量，ES 内部融合），不使用 HyDE
- hyde:    混合检索 + HyDE（假设性文档嵌入），与 hybrid 对比 HyDE 增益

Usage:
    python scripts/eval_vector.py
    python scripts/eval_vector.py --modes vector --limit 3 --top-k 10
    python scripts/eval_vector.py --modes vector,bm25,hybrid,hyde --limit 5
    python scripts/eval_vector.py --modes hybrid,hyde --hyde-num 2 --limit 5
    python scripts/eval_vector.py --modes vector --use-rerank --rerank-top-k 12
    python scripts/eval_vector.py --modes hybrid --expand-parent --top-k 25
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
from core.retrieve.retrieval import RetrievalService
from core.retrieve.retrieval_models import RetrievedChunk, RetrievalOptions
from core.client.embedder import encode
from core.pipeline.rag_pipeline import RAGPipeline

# 抑制 info 日志
logger.disable("core")

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "eval_vector"

VALID_MODES = {"vector", "bm25", "hybrid", "hyde"}


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
    query: str
    chunks: List[ChunkResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class QueryEvalResult:
    """单个 query 的多模式对比结果"""

    query: str
    mode_results: Dict[str, ModeResult] = field(default_factory=dict)
    error: Optional[str] = None


class VectorEvaluator:
    """Vector Retrieval 评估器"""

    def __init__(
        self,
        modes: List[str],
        top_k: int = 25,
        use_rerank: bool = False,
        rerank_top_k: int = 12,
        expand_parent: bool = False,
        hyde_num: int = 1,
        dataset_ids: Optional[List[str]] = None,
        output_dir: str = str(OUTPUT_DIR),
    ):
        invalid = set(modes) - VALID_MODES
        if invalid:
            raise ValueError(f"无效的检索模式: {invalid}，可选: {VALID_MODES}")

        self.modes = modes
        self.top_k = top_k
        self.use_rerank = use_rerank
        self.rerank_top_k = rerank_top_k
        self.expand_parent = expand_parent
        self.hyde_num = hyde_num
        self.dataset_ids = dataset_ids

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.output_dir / f"eval_{self.eval_id}"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.retrieval = RetrievalService()
        # 只在需要 parent expand 时实例化 pipeline
        self._pipeline = None
        self.results: List[QueryEvalResult] = []

    @property
    def pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            self._pipeline = RAGPipeline()
        return self._pipeline

    def _build_options(self, use_hyde: bool = False) -> RetrievalOptions:
        """构建检索选项"""
        return RetrievalOptions(
            top_k=self.top_k,
            use_rerank=False,  # rerank 由脚本自行控制
            use_hyde=use_hyde,
            hyde_num_hypotheses=self.hyde_num,
            dataset_ids=self.dataset_ids,
        )

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

    def _run_single_mode(self, query: str, mode: str) -> ModeResult:
        """执行单模式检索"""
        use_hyde = mode == "hyde"
        options = self._build_options(use_hyde=use_hyde)
        t0 = time.time()

        try:
            if mode == "vector":
                query_vector = encode(query)
                if query_vector is None:
                    return ModeResult(
                        mode=mode, query=query, error="向量化失败"
                    )
                chunks = self.retrieval._execute_vector_search(
                    query_vector, options, self.top_k
                )
            elif mode == "bm25":
                query_string = self.retrieval._build_bm25_query(query, options)
                chunks = self.retrieval._execute_bm25(
                    query_string, options, self.top_k
                )
            elif mode in ("hybrid", "hyde"):
                # hybrid: 不使用 HyDE; hyde: 使用 HyDE
                query_string = self.retrieval._build_bm25_query(query, options)
                query_vector = encode(query)
                chunks = self.retrieval._hybrid_search(query_string, query_vector, options)
            else:
                return ModeResult(
                    mode=mode, query=query, error=f"未知模式: {mode}"
                )

            # 可选 rerank
            if self.use_rerank and chunks:
                rerank_opts = RetrievalOptions(
                    top_k=self.top_k,
                    use_rerank=True,
                    rerank_top_k=self.rerank_top_k,
                    dataset_ids=self.dataset_ids,
                )
                from core.query_engineer.rerank_query import build_rerank_query
                rerank_query = build_rerank_query(query)
                chunks = self.retrieval._rerank(rerank_query, chunks, rerank_opts)

            # 可选 parent expand
            if self.expand_parent and chunks:
                chunks = self.pipeline._expand_to_parent_chunks(chunks)

            duration = time.time() - t0
            chunk_results = self._chunks_to_results(chunks)

            return ModeResult(
                mode=mode,
                query=query,
                chunks=chunk_results,
                duration_seconds=round(duration, 3),
            )

        except Exception as e:
            return ModeResult(
                mode=mode,
                query=query,
                error=str(e),
            )

    def evaluate(self, query: str) -> QueryEvalResult:
        """评估单个 query 的所有模式"""
        result = QueryEvalResult(query=query)

        for mode in self.modes:
            mode_result = self._run_single_mode(query, mode)
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
                            f"  [{mode}] {len(mr.chunks)} chunks, "
                            f"{mr.duration_seconds:.2f}s"
                        )

            time.sleep(0.3)

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
                "use_rerank": self.use_rerank,
                "rerank_top_k": self.rerank_top_k,
                "expand_parent": self.expand_parent,
                "hyde_num": self.hyde_num,
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
            f.write("Vector Retrieval 评估结果汇总\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"评估 ID: {self.eval_id}\n")
            f.write(f"检索模式: {', '.join(self.modes)}\n")
            f.write(f"Top-K: {self.top_k}\n")
            f.write(f"Rerank: {self.use_rerank}")
            if self.use_rerank:
                f.write(f" (top_k={self.rerank_top_k})")
            f.write("\n")
            f.write(f"Parent Expand: {self.expand_parent}\n")
            if "hyde" in self.modes:
                f.write(f"HyDE hypotheses: {self.hyde_num}\n")
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
                        f"\n--- [{mode}] {len(mr.chunks)} chunks | "
                        f"{mr.duration_seconds:.3f}s ---\n"
                    )
                    for c in mr.chunks:
                        preview = c.content_preview[:100].replace("\n", " ")
                        f.write(
                            f"  [{c.rank:2d}] {c.doc_title[:40]} "
                            f"(score={c.score:.4f}, type={c.chunk_type})\n"
                        )
                        f.write(f"       {preview}...\n")

                f.write("\n")

        print(f"\n{'='*60}")
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
        print(f"{'='*60}")

        return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Vector Retrieval 评估工具")
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
        default="vector,bm25,hybrid,hyde",
        help="检索模式，逗号分隔（默认 vector,bm25,hybrid,hyde）",
    )
    parser.add_argument(
        "--use-rerank",
        action="store_true",
        help="是否 rerank",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=12,
        help="rerank 保留数量（默认 12）",
    )
    parser.add_argument(
        "--expand-parent",
        action="store_true",
        help="是否展开 parent",
    )
    parser.add_argument(
        "--hyde-num",
        type=int,
        default=1,
        help="HyDE 生成假设性文档数量（默认 1）",
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
    print(f"Rerank: {args.use_rerank} (top_k={args.rerank_top_k})")
    print(f"Expand Parent: {args.expand_parent}")
    if "hyde" in modes:
        print(f"HyDE hypotheses: {args.hyde_num}")
    print(f"Queries file: {queries_path}")
    print(f"Output dir: {args.output_dir}")

    evaluator = VectorEvaluator(
        modes=modes,
        top_k=args.top_k,
        use_rerank=args.use_rerank,
        rerank_top_k=args.rerank_top_k,
        expand_parent=args.expand_parent,
        hyde_num=args.hyde_num,
        dataset_ids=dataset_ids,
        output_dir=args.output_dir,
    )

    evaluator.batch_eval(queries)
    evaluator.generate_report()


if __name__ == "__main__":
    main()
