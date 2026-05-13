"""
RAG Pipeline 批量评估脚本

类似 ragflow_rag_eval.py 的评估格式，记录所有中间结果

Usage:
    python scripts/batch_eval.py
    python scripts/batch_eval.py --queries scripts/queries.txt
    python scripts/batch_eval.py --output-dir ./eval_results
"""

import os
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
from core.retrieve.retrieval_models import RetrievedChunk

# 抑制 info 日志
logger.disable("core")

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "eval_results"


@dataclass
class RetrievalResult:
    """单个检索结果"""

    rank: int
    chunk_id: str
    doc_title: str
    chunk_type: str
    content: str
    content_length: int
    score: float
    parent_id: Optional[str] = None


@dataclass
class SearchIntermediateResult:
    """搜索阶段中间结果"""

    raw_query: str
    sub_queries: List[str]
    intents: List[str]
    rewritten_queries: List[str]
    top_k: int = 0
    total_chunks: int = 0
    duration_seconds: float = 0.0
    bm25_chunks_count: int = 0
    vector_chunks_count: int = 0
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    # 原始 BM25 和 Vector 检索结果（per rewritten query）
    bm25_chunks: List[RetrievalResult] = field(default_factory=list)
    vector_chunks: List[RetrievalResult] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class GenerateIntermediateResult:
    """生成阶段中间结果"""

    model: str
    question: str
    answer: str
    tokens: int
    prompt_tokens: int
    completion_tokens: int
    duration_seconds: float
    reference_count: int = 0
    references: List[Dict] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class FullRAGResult:
    """完整的 RAG 结果"""

    eval_id: str
    timestamp: str
    query: str
    search_result: Optional[SearchIntermediateResult] = None
    generate_result: Optional[GenerateIntermediateResult] = None
    timing: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


class RAGPipelineEvaluator:
    """RAG Pipeline 评估器"""

    def __init__(
        self,
        top_k: int = 20,
        rerank_top_k: int = 10,
        use_rewrite: bool = True,
        use_generation: bool = True,
        output_dir: str = str(OUTPUT_DIR),
    ):
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.use_rewrite = use_rewrite
        self.use_generation = use_generation

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.output_dir / f"eval_{self.eval_id}"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline = RAGPipeline()
        self.results: List[FullRAGResult] = []

    def evaluate(self, query: str) -> FullRAGResult:
        """执行单个 query 的完整流程"""
        eval_result = FullRAGResult(
            eval_id=self.eval_id,
            timestamp=datetime.now().isoformat(),
            query=query,
        )

        start_time = time.time()

        try:
            result = self.pipeline.run(
                query=query,
                top_k=self.top_k,
                use_understand=True,
                use_rewrite=self.use_rewrite,
                use_rerank=True,
                rerank_top_k=self.rerank_top_k,
                use_generation=self.use_generation,
            )

            elapsed = time.time() - start_time

            # 构建检索结果（最终结果 after rerank + parent expand）
            retrieval_results = []
            for i, chunk in enumerate(result.chunks):
                retrieval_results.append(
                    RetrievalResult(
                        rank=i + 1,
                        chunk_id=chunk.chunk_id,
                        doc_title=chunk.doc_title or "",
                        chunk_type=chunk.chunk_type or "",
                        content=chunk.content[:500],  # 截断保存
                        content_length=len(chunk.content),
                        score=round(chunk.score, 4) if chunk.score else 0,
                        parent_id=chunk.parent_id,
                    )
                )

            # 构建原始 BM25 检索结果（去重合并）
            all_bm25: List[RetrievalResult] = []
            seen_bm25_ids = set()
            for rq, chunks in result.per_query_bm25_chunks.items():
                for chunk in chunks:
                    if chunk.chunk_id not in seen_bm25_ids:
                        seen_bm25_ids.add(chunk.chunk_id)
                        all_bm25.append(
                            RetrievalResult(
                                rank=len(all_bm25) + 1,
                                chunk_id=chunk.chunk_id,
                                doc_title=chunk.doc_title or "",
                                chunk_type=chunk.chunk_type or "",
                                content=chunk.content[:500],
                                content_length=len(chunk.content),
                                score=round(chunk.score, 4) if chunk.score else 0,
                                parent_id=chunk.parent_id,
                            )
                        )

            # 构建原始 Vector 检索结果（去重合并）
            all_vector: List[RetrievalResult] = []
            seen_vector_ids = set()
            for rq, chunks in result.per_query_vector_chunks.items():
                for chunk in chunks:
                    if chunk.chunk_id not in seen_vector_ids:
                        seen_vector_ids.add(chunk.chunk_id)
                        all_vector.append(
                            RetrievalResult(
                                rank=len(all_vector) + 1,
                                chunk_id=chunk.chunk_id,
                                doc_title=chunk.doc_title or "",
                                chunk_type=chunk.chunk_type or "",
                                content=chunk.content[:500],
                                content_length=len(chunk.content),
                                score=round(chunk.score, 4) if chunk.score else 0,
                                parent_id=chunk.parent_id,
                            )
                        )

            # 获取 sub_queries 和 intents
            sub_queries = []
            intents = []
            if result.understanding_result:
                sub_queries = [
                    sq.query for sq in result.understanding_result.sub_queries
                ]
                intents = [sq.intent for sq in result.understanding_result.sub_queries]

            search_result = SearchIntermediateResult(
                raw_query=query,
                sub_queries=sub_queries,
                intents=intents,
                rewritten_queries=result.rewritten_queries,
                top_k=self.top_k,
                total_chunks=len(result.chunks),
                duration_seconds=result.timing.get("retrieve", 0),
                bm25_chunks_count=len(all_bm25),
                vector_chunks_count=len(all_vector),
                retrieval_results=retrieval_results,
                bm25_chunks=all_bm25,
                vector_chunks=all_vector,
            )
            eval_result.search_result = search_result
            eval_result.timing = result.timing

            # 构建生成结果
            if self.use_generation and result.generation_answer:
                usage = result.generation_usage
                gen_result = GenerateIntermediateResult(
                    model="deepseek-v4-flash",
                    question=query,
                    answer=result.generation_answer,
                    tokens=usage.total_tokens if usage else 0,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    duration_seconds=result.timing.get("generation", 0),
                    reference_count=len(result.chunks),
                    references=[
                        {
                            "doc_title": c.doc_title,
                            "chunk_id": c.chunk_id,
                            "content_preview": c.content[:200],
                        }
                        for c in result.chunks[:10]
                    ],
                )
                eval_result.generate_result = gen_result

        except Exception as e:
            eval_result.error = str(e)
            import traceback

            traceback.print_exc()

        self.results.append(eval_result)
        return eval_result

    def batch_eval(self, queries: List[str]) -> List[FullRAGResult]:
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
                if result.search_result:
                    print(
                        f"  [检索] {result.search_result.total_chunks} chunks, "
                        f"BM25={result.search_result.bm25_chunks_count}, "
                        f"Vector={result.search_result.vector_chunks_count}"
                    )
                if result.generate_result:
                    print(
                        f"  [生成] {result.generate_result.tokens} tokens, "
                        f"{result.generate_result.duration_seconds:.1f}s"
                    )
                    print(f"  [答案] {result.generate_result.answer[:150]}...")

            time.sleep(0.5)

        return results

    def _save_result(self, result: FullRAGResult):
        """保存单个结果"""
        safe_query = (
            result.query.replace("/", "_").replace(" ", "_").replace("\n", "_")[:40]
        )
        filename = f"query_{safe_query}_{result.eval_id}.json"
        filepath = self.eval_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

        print(f"  [Output] {filepath.name}")

    def generate_report(self) -> Dict:
        """生成评估报告"""
        total = len(self.results)
        errors = sum(1 for r in self.results if r.error)
        successes = total - errors

        total_tokens = 0
        total_prompt = 0
        total_completion = 0
        total_time = 0.0
        total_retrieve_time = 0.0
        total_rewrite_time = 0.0
        total_understanding_time = 0.0
        total_generation_time = 0.0

        for r in self.results:
            if r.generate_result:
                total_tokens += r.generate_result.tokens
                total_prompt += r.generate_result.prompt_tokens
                total_completion += r.generate_result.completion_tokens
            if r.timing:
                total_time += sum(r.timing.values())
                total_retrieve_time += r.timing.get("retrieve", 0)
                total_rewrite_time += r.timing.get("rewrite", 0)
                total_understanding_time += r.timing.get("understanding", 0)
                total_generation_time += r.timing.get("generation", 0)

        report = {
            "eval_id": self.eval_id,
            "timestamp": datetime.now().isoformat(),
            "total_queries": total,
            "successes": successes,
            "errors": errors,
            "top_k": self.top_k,
            "rerank_top_k": self.rerank_top_k,
            "use_rewrite": self.use_rewrite,
            "statistics": {
                "total_tokens": total_tokens,
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "avg_tokens_per_query": total_tokens // max(successes, 1),
                "total_time_seconds": round(total_time, 2),
                "avg_time_per_query": round(total_time / max(successes, 1), 2),
                "retrieve_time": round(total_retrieve_time, 2),
                "rewrite_time": round(total_rewrite_time, 2),
                "understanding_time": round(total_understanding_time, 2),
                "generation_time": round(total_generation_time, 2),
            },
            "queries": [],
        }

        for result in self.results:
            qr = {
                "query": result.query,
                "error": result.error,
                "timing": {k: round(v, 3) for k, v in result.timing.items()},
            }
            if result.search_result:
                qr["search"] = {
                    "sub_queries": result.search_result.sub_queries,
                    "rewritten_queries": result.search_result.rewritten_queries,
                    "total_chunks": result.search_result.total_chunks,
                    "bm25_count": result.search_result.bm25_chunks_count,
                    "vector_count": result.search_result.vector_chunks_count,
                    "duration": round(result.search_result.duration_seconds, 3),
                    "top_chunks": [
                        {
                            "rank": c.rank,
                            "doc_title": c.doc_title,
                            "chunk_type": c.chunk_type,
                            "score": c.score,
                            "content_preview": c.content[:150],
                        }
                        for c in result.search_result.retrieval_results
                    ],
                    "bm25_chunks": [
                        {
                            "rank": c.rank,
                            "doc_title": c.doc_title,
                            "score": c.score,
                            "content_preview": c.content[:150],
                        }
                        for c in result.search_result.bm25_chunks
                    ],
                    "vector_chunks": [
                        {
                            "rank": c.rank,
                            "doc_title": c.doc_title,
                            "score": c.score,
                            "content_preview": c.content[:150],
                        }
                        for c in result.search_result.vector_chunks
                    ],
                }
            if result.generate_result:
                qr["generation"] = {
                    "tokens": result.generate_result.tokens,
                    "prompt_tokens": result.generate_result.prompt_tokens,
                    "completion_tokens": result.generate_result.completion_tokens,
                    "duration": round(result.generate_result.duration_seconds, 3),
                    "answer_preview": result.generate_result.answer[:300],
                    "reference_count": result.generate_result.reference_count,
                }
            report["queries"].append(qr)

        # 保存 JSON 报告
        report_path = self.eval_dir / "eval_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 生成人类可读总结
        summary_path = self.eval_dir / "eval_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("RAG Pipeline 评估结果汇总\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"评估 ID: {self.eval_id}\n")
            f.write(f"总问题数: {total}\n")
            f.write(f"成功: {successes}\n")
            f.write(f"失败: {errors}\n\n")

            f.write(f"[统计]\n")
            f.write(f"  Top_K: {self.top_k}, Rerank Top_K: {self.rerank_top_k}\n")
            f.write(
                f"  总 Tokens: {total_tokens} "
                f"(prompt={total_prompt}, completion={total_completion})\n"
            )
            f.write(f"  平均 Tokens/Query: {total_tokens // max(successes, 1)}\n")
            f.write(
                f"  总耗时: {total_time:.1f}s, 平均: {total_time/max(successes, 1):.1f}s/query\n"
            )
            f.write(f"  Retrieval: {total_retrieve_time:.1f}s\n")
            f.write(f"  Rewrite: {total_rewrite_time:.1f}s\n")
            f.write(f"  Understanding: {total_understanding_time:.1f}s\n")
            f.write(f"  Generation: {total_generation_time:.1f}s\n\n")

            for i, result in enumerate(self.results, 1):
                f.write(f"{'='*60}\n")
                f.write(f"Q{i}: {result.query}\n")
                f.write(f"{'='*60}\n")

                if result.error:
                    f.write(f"[错误] {result.error}\n\n")
                    continue

                if result.search_result:
                    f.write(
                        f"[检索] {result.search_result.total_chunks} chunks | "
                        f"BM25={result.search_result.bm25_chunks_count}, "
                        f"Vector={result.search_result.vector_chunks_count} | "
                        f"{result.search_result.duration_seconds:.2f}s\n"
                    )
                    f.write(f"[Final Chunks] ({len(result.search_result.retrieval_results)})\n")
                    for c in result.search_result.retrieval_results:
                        preview = c.content[:80].replace("\n", " ")
                        f.write(f"  [{c.rank}] {c.doc_title} (score={c.score:.3f})\n")
                        f.write(f"      {preview}...\n")
                    f.write(f"[BM25 Chunks] ({len(result.search_result.bm25_chunks)})\n")
                    for c in result.search_result.bm25_chunks:
                        preview = c.content[:80].replace("\n", " ")
                        f.write(f"  [{c.rank}] {c.doc_title} (score={c.score:.3f})\n")
                        f.write(f"      {preview}...\n")
                    f.write(f"[Vector Chunks] ({len(result.search_result.vector_chunks)})\n")
                    for c in result.search_result.vector_chunks:
                        preview = c.content[:80].replace("\n", " ")
                        f.write(f"  [{c.rank}] {c.doc_title} (score={c.score:.3f})\n")
                        f.write(f"      {preview}...\n")

                if result.generate_result:
                    f.write(f"[回答]\n{result.generate_result.answer}\n\n")
                    f.write(
                        f"[统计] "
                        f"tokens={result.generate_result.tokens} | "
                        f"{result.generate_result.duration_seconds:.2f}s\n"
                    )
                f.write("\n")

        print(f"\n{'='*60}")
        print(f"[评估完成]")
        print(f"  总问题数: {total}")
        print(f"  成功/失败: {successes}/{errors}")
        print(f"  总 Tokens: {total_tokens} (p={total_prompt}, c={total_completion})")
        print(f"  总耗时: {total_time:.1f}s")
        print(f"  Report: {report_path}")
        print(f"  Summary: {summary_path}")
        print(f"{'='*60}")

        return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG Pipeline 评估工具")
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
        help="检索返回数量",
    )
    parser.add_argument(
        "--rerank-top-k",
        "-r",
        type=int,
        default=12,
        help="Rerank 后保留数量",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="跳过 Query Rewrite",
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="跳过 Generation（只测检索）",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="限制查询数量（0=全部）",
    )

    args = parser.parse_args()

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
    print(f"Queries file: {queries_path}")
    print(f"Output dir: {args.output_dir}")

    evaluator = RAGPipelineEvaluator(
        top_k=args.top_k,
        rerank_top_k=args.rerank_top_k,
        use_rewrite=not args.no_rewrite,
        use_generation=not args.no_generation,
        output_dir=args.output_dir,
    )

    evaluator.batch_eval(queries)
    evaluator.generate_report()


if __name__ == "__main__":
    main()
