"""
RAG Pipeline 批量评估脚本 — 通过 API 调用

类似 ragflow_rag_eval.py 的评估格式，通过 HTTP API 调用

Usage:
    python scripts/batch_eval_api.py
    python scripts/batch_eval_api.py --queries scripts/queries.txt
    python scripts/batch_eval_api.py --output-dir ./eval_results
"""

import os
import sys
import json
import time
import httpx
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "eval_results"

# API 配置
API_BASE = "http://localhost:8000"
API_KEY = "rag-clean-api-key"  # 如有认证


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


class RAGPipelineAPIEvaluator:
    """通过 API 调用 RAG Pipeline 评估器"""

    def __init__(
        self,
        api_base: str = API_BASE,
        api_key: str = API_KEY,
        top_k: int = 20,
        rerank_top_k: int = 12,
        use_rewrite: bool = True,
        output_dir: str = str(OUTPUT_DIR),
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.use_rewrite = use_rewrite

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.output_dir / f"eval_{self.eval_id}"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[FullRAGResult] = []

        # HTTP 客户端
        self.client = httpx.Client(timeout=300.0)

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def evaluate(self, query: str) -> FullRAGResult:
        """通过 API 执行单个 query"""
        eval_result = FullRAGResult(
            eval_id=self.eval_id,
            timestamp=datetime.now().isoformat(),
            query=query,
        )

        start_time = time.time()

        try:
            # 调用 RAG API
            payload = {
                "query": query,
                "top_k": self.top_k,
                "use_rewrite": self.use_rewrite,
                "use_rerank": True,
                "rerank_top_k": self.rerank_top_k,
            }

            response = self.client.post(
                f"{self.api_base}/api/v1/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )

            elapsed = time.time() - start_time

            if response.status_code != 200:
                eval_result.error = f"HTTP {response.status_code}: {response.text[:500]}"
                return eval_result

            data = response.json()

            # 解析响应
            answer = data.get("answer", "")
            usage = data.get("usage", {})
            sources = data.get("sources", [])
            timing = data.get("time", {})

            # 构建检索结果（从 sources 提取）
            retrieval_results = []
            for i, src in enumerate(sources):
                retrieval_results.append(RetrievalResult(
                    rank=i + 1,
                    chunk_id=src.get("chunk_id", ""),
                    doc_title=src.get("doc_name", ""),
                    chunk_type=src.get("chunk_type", ""),
                    content=src.get("snippet", "")[:500],
                    content_length=len(src.get("snippet", "")),
                    score=src.get("score", 0.0),
                ))

            search_result = SearchIntermediateResult(
                raw_query=query,
                sub_queries=[query],  # API 不返回 sub_queries
                intents=[],
                rewritten_queries=[],
                top_k=self.top_k,
                total_chunks=len(sources),
                duration_seconds=timing.get("retrieve", 0),
                retrieval_results=retrieval_results,
            )
            eval_result.search_result = search_result
            eval_result.timing = timing

            # 构建生成结果
            gen_result = GenerateIntermediateResult(
                model="deepseek-v4-flash",
                question=query,
                answer=answer,
                tokens=usage.get("total_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                duration_seconds=timing.get("generation", elapsed),
                reference_count=len(sources),
                references=sources[:10],
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
                    print(f"  [检索] {result.search_result.total_chunks} chunks")
                if result.generate_result:
                    print(f"  [生成] {result.generate_result.tokens} tokens, "
                          f"{result.generate_result.duration_seconds:.1f}s")
                    print(f"  [答案] {result.generate_result.answer[:150]}...")

            time.sleep(0.5)

        return results

    def _save_result(self, result: FullRAGResult):
        """保存单个结果"""
        safe_query = (
            result.query.replace("/", "_")
            .replace(" ", "_")
            .replace("\n", "_")[:40]
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
        total_generation_time = 0.0

        for r in self.results:
            if r.generate_result:
                total_tokens += r.generate_result.tokens
                total_prompt += r.generate_result.prompt_tokens
                total_completion += r.generate_result.completion_tokens
            if r.timing:
                total_time += sum(r.timing.values())
                total_retrieve_time += r.timing.get("retrieve", 0)
                total_generation_time += r.timing.get("generation", 0)

        report = {
            "eval_id": self.eval_id,
            "timestamp": datetime.now().isoformat(),
            "total_queries": total,
            "successes": successes,
            "errors": errors,
            "api_base": self.api_base,
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
                    "total_chunks": result.search_result.total_chunks,
                    "duration": round(result.search_result.duration_seconds, 3),
                    "top_chunks": [
                        {
                            "rank": c.rank,
                            "doc_title": c.doc_title,
                            "score": c.score,
                            "content_preview": c.content[:150],
                        }
                        for c in result.search_result.retrieval_results[:5]
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
            f.write("RAG Pipeline 评估结果汇总 (API 模式)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"API: {self.api_base}\n")
            f.write(f"评估 ID: {self.eval_id}\n")
            f.write(f"总问题数: {total}\n")
            f.write(f"成功: {successes}\n")
            f.write(f"失败: {errors}\n\n")

            f.write(f"[统计]\n")
            f.write(f"  Top_K: {self.top_k}, Rerank Top_K: {self.rerank_top_k}\n")
            f.write(f"  总 Tokens: {total_tokens} "
                    f"(prompt={total_prompt}, completion={total_completion})\n")
            f.write(f"  平均 Tokens/Query: {total_tokens // max(successes, 1)}\n")
            f.write(f"  总耗时: {total_time:.1f}s, 平均: {total_time/max(successes, 1):.1f}s/query\n")
            f.write(f"  Retrieval: {total_retrieve_time:.1f}s\n")
            f.write(f"  Generation: {total_generation_time:.1f}s\n\n")

            for i, result in enumerate(self.results, 1):
                f.write(f"{'='*60}\n")
                f.write(f"Q{i}: {result.query}\n")
                f.write(f"{'='*60}\n")

                if result.error:
                    f.write(f"[错误] {result.error}\n\n")
                    continue

                if result.search_result:
                    f.write(f"[检索] {result.search_result.total_chunks} chunks | "
                            f"{result.search_result.duration_seconds:.2f}s\n")
                    for c in result.search_result.retrieval_results[:3]:
                        preview = c.content[:80].replace("\n", " ")
                        f.write(f"  [{c.rank}] {c.doc_title} (score={c.score:.3f})\n")
                        f.write(f"      {preview}...\n")

                if result.generate_result:
                    f.write(f"[回答]\n{result.generate_result.answer}\n\n")
                    f.write(f"[统计] "
                            f"tokens={result.generate_result.tokens} | "
                            f"{result.generate_result.duration_seconds:.2f}s\n")
                f.write("\n")

        print(f"\n{'='*60}")
        print(f"[评估完成]")
        print(f"  API: {self.api_base}")
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

    parser = argparse.ArgumentParser(description="RAG Pipeline 评估工具 (API 模式)")
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
        "--api-base",
        default=API_BASE,
        help="API 基础 URL",
    )
    parser.add_argument(
        "--api-key",
        default=API_KEY,
        help="API Key",
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
    print(f"API Base: {args.api_base}")
    print(f"Output dir: {args.output_dir}")

    evaluator = RAGPipelineAPIEvaluator(
        api_base=args.api_base,
        api_key=args.api_key,
        top_k=args.top_k,
        rerank_top_k=args.rerank_top_k,
        use_rewrite=not args.no_rewrite,
        output_dir=args.output_dir,
    )

    evaluator.batch_eval(queries)
    evaluator.generate_report()


if __name__ == "__main__":
    main()
