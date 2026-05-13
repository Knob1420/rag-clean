"""
ReAct Agent 批量评估脚本

逐步执行多个查询的 ReAct 循环，记录所有中间结果：
- 每步的 thought / tool_calls / observation
- 终止原因、token 用量、累积 chunks
- 所有结果保存到 JSON 文件 + 可读摘要

用法：
    python scripts/batch_react_eval.py
    python scripts/batch_react_eval.py --queries scripts/queries.txt
    python scripts/batch_react_eval.py --max-iter 5 --limit 3
    python scripts/batch_react_eval.py --output-dir ./react_eval_results
"""

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from core.agent.react_agent import ReActAgent, ReActResult, AgentStep, _parse_json_tiered
from core.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from core.retrieve.retrieval_models import RetrievedChunk, TokenUsage

# 抑制 info 日志
logger.disable("core")

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "react_eval_results"


# ══════════════════════════════════════════════════════════════
# 数据模型
# ══════════════════════════════════════════════════════════════


def _chunk_to_dict(chunk: RetrievedChunk) -> Dict[str, Any]:
    """将 RetrievedChunk 转为可序列化 dict"""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_title": chunk.doc_title,
        "content": chunk.content[:500],
        "content_length": len(chunk.content),
        "score": round(chunk.score, 4) if chunk.score else 0,
        "chunk_type": chunk.chunk_type,
        "parent_id": chunk.parent_id,
    }


def _step_to_dict(step: AgentStep) -> Dict[str, Any]:
    """将 AgentStep 转为可序列化 dict"""
    return {
        "iteration": step.iteration,
        "thought": step.thought[:500],
        "action": step.action,
        "action_input": {
            k: (str(v)[:200] if isinstance(v, str) else v)
            for k, v in step.action_input.items()
        },
        "observation": step.observation[:500],
        "duration": round(step.duration, 3),
    }


@dataclass
class ReactEvalResult:
    """单个查询的 ReAct 评估结果"""

    eval_id: str
    timestamp: str
    query: str
    answer: str = ""
    terminated_reason: str = ""
    total_iterations: int = 0
    steps: List[Dict] = field(default_factory=list)
    accumulated_chunks: List[Dict] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


class ReactAgentEvaluator:
    """ReAct Agent 评估器"""

    def __init__(
        self,
        max_iterations: int = 10,
        max_llm_retries: int = 2,
        output_dir: str = str(OUTPUT_DIR),
    ):
        self.max_iterations = max_iterations
        self.max_llm_retries = max_llm_retries

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.output_dir / f"react_eval_{self.eval_id}"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[ReactEvalResult] = []

    def evaluate(self, query: str) -> ReactEvalResult:
        """执行单个 query 的 ReAct Agent，记录所有中间结果"""
        eval_result = ReactEvalResult(
            eval_id=self.eval_id,
            timestamp=datetime.now().isoformat(),
            query=query,
        )

        start_time = time.time()

        try:
            agent = ReActAgent(
                max_iterations=self.max_iterations,
                max_llm_retries=self.max_llm_retries,
            )
            result: ReActResult = agent.run(query)

            elapsed = time.time() - start_time

            eval_result.answer = result.answer
            eval_result.terminated_reason = result.terminated_reason
            eval_result.total_iterations = result.total_iterations

            # 步骤记录
            eval_result.steps = [_step_to_dict(s) for s in result.steps]

            # 累积 chunks
            eval_result.accumulated_chunks = [
                _chunk_to_dict(c) for c in result.chunks
            ]

            # Token 用量
            if result.usage:
                eval_result.token_usage = {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                }

            eval_result.timing = {
                "total": round(elapsed, 3),
                **{k: round(v, 3) for k, v in result.timing.items()},
            }

        except Exception as e:
            eval_result.error = str(e)
            import traceback
            traceback.print_exc()

        self.results.append(eval_result)
        return eval_result

    def batch_eval(self, queries: List[str]) -> List[ReactEvalResult]:
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
                print(
                    f"  [终止] {result.terminated_reason} | "
                    f"迭代={result.total_iterations}"
                )
                print(
                    f"  [检索] {len(result.accumulated_chunks)} chunks"
                )
                if result.token_usage:
                    print(
                        f"  [Token] total={result.token_usage['total_tokens']} | "
                        f"prompt={result.token_usage['prompt_tokens']} | "
                        f"completion={result.token_usage['completion_tokens']}"
                    )
                if result.timing:
                    print(f"  [耗时] {result.timing.get('total', 0):.1f}s")
                print(f"  [答案] {result.answer[:150]}...")

            time.sleep(0.5)

        return results

    def _save_result(self, result: ReactEvalResult):
        """保存单个结果"""
        safe_query = "".join(
            c if c.isalnum() or c in "_-" else "_"
            for c in result.query[:30]
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
        total_iterations = 0

        terminate_reasons: Dict[str, int] = {}

        for r in self.results:
            if r.token_usage:
                total_tokens += r.token_usage.get("total_tokens", 0)
                total_prompt += r.token_usage.get("prompt_tokens", 0)
                total_completion += r.token_usage.get("completion_tokens", 0)
            total_time += r.timing.get("total", 0)
            total_iterations += r.total_iterations
            reason = r.terminated_reason or "unknown"
            terminate_reasons[reason] = terminate_reasons.get(reason, 0) + 1

        report = {
            "eval_id": self.eval_id,
            "eval_type": "react_agent",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_iterations": self.max_iterations,
                "max_llm_retries": self.max_llm_retries,
            },
            "summary": {
                "total_queries": total,
                "successes": successes,
                "errors": errors,
                "terminate_reasons": terminate_reasons,
                "total_iterations": total_iterations,
                "avg_iterations_per_query": round(
                    total_iterations / max(successes, 1), 1
                ),
                "total_tokens": total_tokens,
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "avg_tokens_per_query": total_tokens // max(successes, 1),
                "total_time_seconds": round(total_time, 2),
                "avg_time_per_query": round(total_time / max(successes, 1), 2),
            },
            "queries": [],
        }

        for result in self.results:
            qr: Dict[str, Any] = {
                "query": result.query,
                "error": result.error,
                "terminated_reason": result.terminated_reason,
                "iterations": result.total_iterations,
                "timing": {k: round(v, 3) for k, v in result.timing.items()},
                "token_usage": result.token_usage,
                "chunks_count": len(result.accumulated_chunks),
                "steps_count": len(result.steps),
                "steps_summary": [
                    {
                        "iteration": s["iteration"],
                        "action": s["action"],
                        "duration": s["duration"],
                    }
                    for s in result.steps
                ],
                "answer_preview": result.answer[:300],
                "top_chunks": [
                    {
                        "doc_title": c["doc_title"],
                        "score": c["score"],
                        "content_preview": c["content"][:150],
                    }
                    for c in result.accumulated_chunks[:5]
                ],
            }
            report["queries"].append(qr)

        # 保存 JSON 报告
        report_path = self.eval_dir / "eval_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 生成可读摘要
        summary_path = self.eval_dir / "eval_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("ReAct Agent 评估结果汇总\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"评估 ID: {self.eval_id}\n")
            f.write(f"最大迭代数: {self.max_iterations}\n")
            f.write(f"总问题数: {total}\n")
            f.write(f"成功: {successes}\n")
            f.write(f"失败: {errors}\n\n")

            f.write(f"[终止原因分布]\n")
            for reason, count in terminate_reasons.items():
                f.write(f"  {reason}: {count}\n")
            f.write("\n")

            f.write(f"[统计]\n")
            f.write(f"  总迭代数: {total_iterations}\n")
            f.write(
                f"  平均迭代/查询: "
                f"{total_iterations / max(successes, 1):.1f}\n"
            )
            f.write(
                f"  总 Tokens: {total_tokens} "
                f"(prompt={total_prompt}, completion={total_completion})\n"
            )
            f.write(f"  平均 Tokens/查询: {total_tokens // max(successes, 1)}\n")
            f.write(
                f"  总耗时: {total_time:.1f}s, "
                f"平均: {total_time / max(successes, 1):.1f}s/查询\n\n"
            )

            for i, result in enumerate(self.results, 1):
                f.write(f"{'=' * 60}\n")
                f.write(f"Q{i}: {result.query}\n")
                f.write(f"{'=' * 60}\n")

                if result.error:
                    f.write(f"[错误] {result.error}\n\n")
                    continue

                f.write(
                    f"[终止] {result.terminated_reason} | "
                    f"迭代={result.total_iterations}\n"
                )
                f.write(f"[检索] {len(result.accumulated_chunks)} chunks\n")

                if result.token_usage:
                    f.write(
                        f"[Token] total={result.token_usage.get('total_tokens', 0)} "
                        f"| prompt={result.token_usage.get('prompt_tokens', 0)} "
                        f"| completion={result.token_usage.get('completion_tokens', 0)}\n"
                    )

                f.write(f"[耗时] {result.timing.get('total', 0):.1f}s\n\n")

                # 步骤概要
                f.write(f"[步骤] ({len(result.steps)})\n")
                for s in result.steps:
                    f.write(
                        f"  iter={s['iteration']} | "
                        f"action={s['action']} | "
                        f"dur={s['duration']:.1f}s\n"
                    )
                    if s["thought"]:
                        f.write(f"    thought: {s['thought'][:120]}\n")
                    if s["observation"]:
                        f.write(f"    obs: {s['observation'][:120]}\n")

                f.write(f"\n[回答]\n{result.answer}\n\n")

        print(f"\n{'=' * 60}")
        print(f"[评估完成]")
        print(f"  总问题数: {total}")
        print(f"  成功/失败: {successes}/{errors}")
        print(f"  终止原因: {terminate_reasons}")
        print(f"  总迭代数: {total_iterations}")
        print(f"  总 Tokens: {total_tokens} (p={total_prompt}, c={total_completion})")
        print(f"  总耗时: {total_time:.1f}s")
        print(f"  Report: {report_path}")
        print(f"  Summary: {summary_path}")
        print(f"{'=' * 60}")

        return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ReAct Agent 批量评估工具")
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
        "--max-iter",
        type=int,
        default=10,
        help="ReAct 最大迭代数",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="LLM 最大重试次数",
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
    print(f"Max iterations: {args.max_iter}")

    evaluator = ReactAgentEvaluator(
        max_iterations=args.max_iter,
        max_llm_retries=args.max_retries,
        output_dir=args.output_dir,
    )

    evaluator.batch_eval(queries)
    evaluator.generate_report()


if __name__ == "__main__":
    main()
