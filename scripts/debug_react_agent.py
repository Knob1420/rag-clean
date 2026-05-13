"""
ReAct Agent Debug 脚本

逐步执行 ReAct 循环，打印每一步的详细信息：
- LLM 返回的 thought / tool_calls
- 工具执行结果（截断显示）
- 终止原因
- 所有中间结果保存到 JSON 文件

用法：
    python scripts/debug_react_agent.py "智加G1的功耗是多少"
    python scripts/debug_react_agent.py "比较G1和G3" --max-iter 5
    python scripts/debug_react_agent.py "智加G1的功耗是多少" --save
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from core.agent.react_agent import (
    ReActAgent,
    ReActResult,
    AgentStep,
    _parse_json_tiered,
)
from core.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from core.retrieve.retrieval_models import RetrievedChunk


def _chunk_to_dict(chunk: RetrievedChunk) -> Dict[str, Any]:
    """将 RetrievedChunk 转为可序列化 dict"""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "content": chunk.content,
        "score": chunk.score,
        "doc_title": chunk.doc_title,
        "dataset_id": chunk.dataset_id,
        "chunk_type": chunk.chunk_type,
        "doc_hash": chunk.doc_hash,
        "parent_id": chunk.parent_id,
    }


def debug_run(
    query: str,
    max_iterations: int = 10,
    max_llm_retries: int = 2,
    save: bool = False,
    output_dir: str = "debug_output",
):
    """逐步执行 ReAct Agent，打印每步详情，保存所有中间结果"""

    agent = ReActAgent(
        max_iterations=max_iterations,
        max_llm_retries=max_llm_retries,
    )

    # ── 收集所有中间结果 ──
    trace: Dict[str, Any] = {
        "query": query,
        "max_iterations": max_iterations,
        "max_llm_retries": max_llm_retries,
        "tools": [t["function"]["name"] for t in TOOL_DEFINITIONS],
        "steps": [],
        "messages": [],
        "terminated_reason": "",
        "final_answer": "",
        "total_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "accumulated_chunks": [],
    }

    print("=" * 80)
    print(f"  QUERY: {query}")
    print(f"  MAX_ITERATIONS: {max_iterations}")
    print(f"  MAX_LLM_RETRIES: {max_llm_retries}")
    print(f"  TOOLS: {trace['tools']}")
    print("=" * 80)

    # ── 初始化 ──
    agent.tool_executor.reset()
    agent.tool_executor.set_current_query(query)

    from core.agent.prompts import REACT_SYSTEM_PROMPT

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    trace["messages"].append({"role": "system", "content": REACT_SYSTEM_PROMPT})
    trace["messages"].append({"role": "user", "content": query})

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    terminated_reason = ""

    for iteration in range(max_iterations):
        print(f"\n{'─' * 80}")
        print(f"  ITERATION {iteration + 1}/{max_iterations}")
        print(f"{'─' * 80}")

        # ── 1. THINK ──
        print(f"\n[THINK] 调用 LLM...")
        step_start = time.time()

        response_data = agent.llm.call_with_tools(
            messages, TOOL_DEFINITIONS, max_retries=max_llm_retries
        )

        if response_data is None:
            print("[ERROR] LLM 调用彻底失败！")
            answer, syn_usage = agent._synthesize_from_accumulated(query)
            print(f"[降级] 合成回答: {answer[:200]}...")
            terminated_reason = "error"
            trace["terminated_reason"] = terminated_reason
            trace["final_answer"] = answer
            break

        assistant_msg = response_data["message"]
        usage = response_data.get("usage")

        if usage:
            total_usage["prompt_tokens"] += usage.prompt_tokens
            total_usage["completion_tokens"] += usage.completion_tokens
            total_usage["total_tokens"] += usage.total_tokens
            print(
                f"[USAGE] prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens}, "
                f"total={usage.total_tokens}"
            )

        messages.append(assistant_msg)
        trace["messages"].append(assistant_msg)

        # ── 2. ANALYZE ──
        tool_calls = assistant_msg.get("tool_calls")
        content = (assistant_msg.get("content") or "").strip()

        if content:
            print(f"[THOUGHT] {content[:500]}")

        if not tool_calls:
            if not content:
                # 空响应重试
                print("[ANALYZE] 空内容，尝试引导重试...")
                empty_retries = 0
                max_empty_retries = 2
                while empty_retries < max_empty_retries:
                    messages.append(
                        {
                            "role": "user",
                            "content": "请调用工具检索信息，然后使用 finish 提交回答。",
                        }
                    )
                    retry_resp = agent.llm.call_with_tools(
                        messages, TOOL_DEFINITIONS, max_retries=0
                    )
                    if retry_resp is None:
                        break
                    retry_msg = retry_resp["message"]
                    retry_usage = retry_resp.get("usage")
                    if retry_usage:
                        total_usage["prompt_tokens"] += retry_usage.prompt_tokens
                        total_usage[
                            "completion_tokens"
                        ] += retry_usage.completion_tokens
                        total_usage["total_tokens"] += retry_usage.total_tokens
                    messages.append(retry_msg)

                    tool_calls = retry_msg.get("tool_calls")
                    content = (retry_msg.get("content") or "").strip()
                    if tool_calls or content:
                        assistant_msg = retry_msg
                        if content:
                            print(f"[THOUGHT] (重试) {content[:500]}")
                        break
                    empty_retries += 1

                if not tool_calls and not content:
                    print("[ERROR] 重试也失败，优雅降级")
                    answer, _ = agent._synthesize_from_accumulated(query)
                    terminated_reason = "stuck"
                    trace["terminated_reason"] = terminated_reason
                    trace["final_answer"] = answer
                    break

            if not tool_calls:
                print(f"\n{'=' * 80}")
                print(f"  FINAL ANSWER (natural_stop)")
                print(f"{'=' * 80}")
                print(content)
                terminated_reason = "natural_stop"
                trace["terminated_reason"] = terminated_reason
                trace["final_answer"] = content
                break

        # ── 3. ACT ──
        step_record: Dict[str, Any] = {
            "iteration": iteration,
            "thought": content,
            "tool_calls": [],
        }

        for tool_call in tool_calls:
            tc_id = tool_call.get("id", "")
            tc_function = tool_call.get("function", {})
            tool_name = tc_function.get("name", "")
            tool_args_str = tc_function.get("arguments", "{}")

            print(f"\n[ACTION] tool={tool_name}")
            print(f"  args_raw: {tool_args_str[:300]}")

            tool_args = _parse_json_tiered(tool_args_str)
            if tool_args is None:
                tool_args = {}
                print("  [WARN] 参数解析失败，使用空 dict")
            else:
                print(
                    f"  args_parsed: {json.dumps(tool_args, ensure_ascii=False)[:300]}"
                )

            tc_record: Dict[str, Any] = {
                "tool_name": tool_name,
                "tool_call_id": tc_id,
                "args_raw": tool_args_str,
                "args_parsed": tool_args,
                "observation": "",
                "observation_len": 0,
                "duration": 0.0,
            }

            # finish 终止
            if tool_name == "finish":
                answer = tool_args.get("answer", content or "")
                elapsed = time.time() - step_start
                tc_record["observation"] = f"提交最终回答（{len(answer)} 字）"
                tc_record["duration"] = elapsed
                step_record["tool_calls"].append(tc_record)
                trace["steps"].append(step_record)

                print(f"\n{'=' * 80}")
                print(f"  FINAL ANSWER (finish)")
                print(f"  elapsed: {elapsed:.2f}s")
                print(f"{'=' * 80}")
                print(answer)
                terminated_reason = "finish"
                trace["terminated_reason"] = terminated_reason
                trace["final_answer"] = answer
                break

            # 执行工具
            print(f"[OBSERVE] 执行 {tool_name}...")
            obs_start = time.time()
            observation = agent.tool_executor.execute(tool_name, tool_args)
            obs_elapsed = time.time() - obs_start

            tc_record["observation"] = observation
            tc_record["observation_len"] = len(observation)
            tc_record["duration"] = obs_elapsed

            # 显示结果（截断）
            if len(observation) > 600:
                print(f"  result ({len(observation)} chars, {obs_elapsed:.2f}s):")
                print(f"  {observation[:400]}")
                print(f"  ... [截断] ...")
                print(f"  {observation[-150:]}")
            else:
                print(f"  result ({obs_elapsed:.2f}s): {observation}")

            # 追加到 messages
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": observation,
                }
            )
            trace["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": observation[:2000],  # trace 中截断防文件过大
                }
            )

        trace["steps"].append(step_record)

        if terminated_reason:
            break
    else:
        # 超过最大迭代数
        print(f"\n[WARN] 达到最大迭代数 {max_iterations}，优雅降级")
        answer, syn_usage = agent._synthesize_from_accumulated(query)
        print(f"\n{'=' * 80}")
        print(f"  FINAL ANSWER (max_iterations)")
        print(f"{'=' * 80}")
        print(answer)
        terminated_reason = "max_iterations"
        trace["terminated_reason"] = terminated_reason
        trace["final_answer"] = answer

    # ── 保存累积 chunks ──
    trace["accumulated_chunks"] = [
        _chunk_to_dict(c) for c in agent.tool_executor.accumulated_chunks
    ]
    trace["total_usage"] = total_usage

    _print_summary(trace, agent.tool_executor)

    # ── 保存到文件 ──
    if save:
        _save_trace(trace, query, output_dir)

    return trace


def _print_summary(trace: Dict[str, Any], executor: ToolExecutor):
    """打印执行摘要"""
    total_usage = trace["total_usage"]
    print(f"\n{'━' * 80}")
    print(f"  SUMMARY")
    print(f"{'━' * 80}")
    print(f"  总迭代: {len(trace['steps'])}")
    print(f"  终止原因: {trace['terminated_reason']}")
    print(f"  累积 chunks: {len(executor.accumulated_chunks)}")
    print(
        f"  Token 用量: prompt={total_usage['prompt_tokens']}, "
        f"completion={total_usage['completion_tokens']}, "
        f"total={total_usage['total_tokens']}"
    )

    # 打印 chunks 来源
    if executor.accumulated_chunks:
        print(f"\n  累积 chunks 详情:")
        for i, chunk in enumerate(executor.accumulated_chunks):
            doc_name = chunk.doc_title or chunk.doc_id
            print(
                f"    [{i+1}] {doc_name} | type={chunk.chunk_type} | "
                f"score={chunk.score:.4f} | content_len={len(chunk.content)}"
            )


def _save_trace(trace: Dict[str, Any], query: str, output_dir: str):
    """保存完整 trace 到 JSON 文件"""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 文件名：时间戳 + query 前20字
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() or c in "_-" else "_" for c in query[:20])
    filename = f"react_trace_{timestamp}_{safe_query}.json"
    filepath = out_path / filename

    # 构建可序列化的 trace（messages 中可能含不可序列化对象）
    serializable = _make_serializable(trace)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] 完整 trace 已保存: {filepath}")
    print(f"       文件大小: {filepath.stat().st_size / 1024:.1f} KB")


def _make_serializable(obj: Any) -> Any:
    """递归将对象转为 JSON 可序列化结构"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, RetrievedChunk):
        return _chunk_to_dict(obj)
    # 其他类型转 str
    return str(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Agent Debug")
    parser.add_argument("query", help="查询问题")
    parser.add_argument("--max-iter", type=int, default=10, help="最大迭代数")
    parser.add_argument("--max-retries", type=int, default=2, help="LLM 最大重试次数")
    parser.add_argument("--save", action="store_true", help="保存中间结果到 JSON")
    parser.add_argument("--output-dir", default="debug_output", help="输出目录")
    parser.add_argument(
        "--debug", action="store_true", help="启用 loguru DEBUG 级别日志"
    )
    args = parser.parse_args()

    if args.debug:
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")

    debug_run(args.query, args.max_iter, args.max_retries, args.save, args.output_dir)
