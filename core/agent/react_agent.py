"""
ReAct Agent — 推理 + 行动循环引擎

基于"探索-推理-验证"认知流设计，适配 rag-clean 项目：
- LLM 支持 OpenAI Function Calling（tool calling）
- 3 个工具：search_knowledge, spec_query, finish
- 认知工作流：概念探路 → 线索分析/顺藤摸瓜 → 拼图收敛

Resilience 模式：
- 三级 JSON 解析容错
- 卡住循环检测（连续 2 轮相同内容且无工具调用）
- 空响应重试（追加引导消息，最多 2 次）
- 优雅降级（超过最大迭代数时从累积 chunks 合成回答）
- 工具输出截断（防上下文窗口中毒）

流式模式（run_stream）：
- 每步产出 SSE 事件，前端可实时看到 Agent 推理进度
- 事件类型：step_start, step_end, answer_token, done, error
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

from loguru import logger

from core.agent.prompts import REACT_SYSTEM_PROMPT
from core.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from core.generation.llm import LLMClient, get_llm_client
from core.retrieve.retrieval_models import RetrievedChunk, TokenUsage

# ════════════════════════════════════════════════════════════════
# 数据模型
# ════════════════════════════════════════════════════════════════


@dataclass
class AgentStep:
    """单个 ReAct 步骤记录"""

    iteration: int
    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    duration: float = 0.0
    timing: Dict[str, float] = field(default_factory=dict)  # 各阶段耗时


@dataclass
class StreamEvent:
    """Agent 流式事件（供 SSE 推送）"""

    event_type: str  # step_start | step_end | answer_token | done | error
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReActResult:
    """ReAct Agent 执行结果"""

    answer: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    chunks: List[RetrievedChunk] = field(default_factory=list)
    total_iterations: int = 0
    timing: Dict[str, float] = field(default_factory=dict)
    usage: Optional[TokenUsage] = None
    terminated_reason: str = ""  # "finish" | "max_iterations" | "stuck" | "error"


# ════════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════════


def _accumulate_usage(total: TokenUsage, delta: Optional[TokenUsage]) -> None:
    """将 delta 的 token usage 累加到 total（原地修改）"""
    if delta is None:
        return
    total.prompt_tokens += delta.prompt_tokens
    total.completion_tokens += delta.completion_tokens
    total.total_tokens += delta.total_tokens


# ════════════════════════════════════════════════════════════════
# JSON 三级容错解析
# ════════════════════════════════════════════════════════════════


def _repair_json(text: str) -> str:
    """修复常见的 LLM JSON 输出问题"""
    # 修复截断的 JSON：补全缺失的括号
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    if open_braces > 0:
        text += "}" * open_braces
    if open_brackets > 0:
        text += "]" * open_brackets
    # 修复尾随逗号
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # 修复单引号 → 双引号（仅替换 JSON 结构性引号，保留字符串内容中的缩写等）
    result = []
    in_double_quote = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_double_quote = not in_double_quote
            result.append(ch)
        elif ch == "'" and not in_double_quote:
            result.append('"')
        else:
            result.append(ch)
        i += 1
    text = "".join(result)
    return text


def _parse_json_tiered(text: str) -> Optional[Dict[str, Any]]:
    """三级 JSON 解析容错：strict → repair → regex"""
    # Tier 1: 严格解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Tier 2: 修复后解析
    try:
        repaired = _repair_json(text)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Tier 3: 正则提取最外层 JSON 对象
    try:
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass

    # Tier 3b: 尝试提取 ```json 代码块
    try:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except json.JSONDecodeError:
        pass

    return None


# ════════════════════════════════════════════════════════════════
# 终止结果构建辅助
# ════════════════════════════════════════════════════════════════


def _build_result(
    answer: str,
    steps: List[AgentStep],
    chunks: List[RetrievedChunk],
    iteration: int,
    start_time: float,
    total_usage: TokenUsage,
    reason: str,
) -> ReActResult:
    """统一构建 ReActResult"""
    return ReActResult(
        answer=answer,
        steps=steps,
        chunks=chunks,
        total_iterations=iteration,
        timing={"total": time.time() - start_time},
        usage=total_usage,
        terminated_reason=reason,
    )


def _build_done_data(
    iteration: int,
    reason: str,
    chunks: List[RetrievedChunk],
    start_time: float,
    total_usage: TokenUsage,
) -> Dict[str, Any]:
    """统一构建 done 事件的 data"""
    return {
        "iterations": iteration,
        "terminated_reason": reason,
        "chunks_count": len(chunks),
        "time": {"total": time.time() - start_time},
        "usage": total_usage.__dict__,
    }


# ════════════════════════════════════════════════════════════════
# ReAct Agent 核心
# ════════════════════════════════════════════════════════════════


class ReActAgent:
    """ReAct 推理引擎 — Thought + Action + Observation 循环"""

    def __init__(
        self,
        max_iterations: int = 8,
        max_llm_retries: int = 2,
        llm_client: Optional[LLMClient] = None,
        tool_executor: Optional[ToolExecutor] = None,
    ):
        self.max_iterations = max_iterations
        self.max_llm_retries = max_llm_retries
        self.llm = llm_client or get_llm_client()
        self.tool_executor = tool_executor or ToolExecutor()
        self._original_query = ""  # 保存原始 query 用于 rerank

    def run(self, query: str) -> ReActResult:
        """
        执行 ReAct 循环（非流式），内部委托给 run_stream 消费所有事件。
        """
        result: Optional[ReActResult] = None

        for event in self.run_stream(query):
            if event.event_type == "done":
                data = event.data
                result = ReActResult(
                    answer=self._collected_answer,
                    steps=self._collected_steps,
                    chunks=list(self.tool_executor.accumulated_chunks.values()),
                    total_iterations=data.get("iterations", 0),
                    timing=data.get("time", {}),
                    usage=TokenUsage(**data["usage"]) if data.get("usage") else None,
                    terminated_reason=data.get("terminated_reason", ""),
                )

        if result is None:
            result = ReActResult(terminated_reason="error")

        return result

    def run_stream(self, query: str) -> Generator[StreamEvent, None, None]:
        """
        流式执行 ReAct 循环，每步产出 StreamEvent。
        """
        start_time = time.time()
        self.tool_executor.reset()
        self._original_query = query

        # 初始化收集器（供 run() 非流式消费）
        self._collected_answer = ""
        self._collected_steps: List[AgentStep] = []

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        steps: List[AgentStep] = []
        consecutive_same = 0
        consecutive_empty = 0  # 连续无实质新信息的迭代次数
        last_tool_calls_sig = ""
        total_usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        for iteration in range(self.max_iterations):
            step_start = time.time()

            # 通知前端：开始新一轮
            yield StreamEvent(
                event_type="step_start",
                data={
                    "iteration": iteration + 1,
                    "max_iterations": self.max_iterations,
                    "action": "",  # 待 LLM 返回后填入
                },
            )

            # ── 1. THINK（流式）────────────────────────────
            t_think_start = time.time()

            response_data: Optional[Dict[str, Any]] = None
            stream_gen = self.llm.call_with_tools_stream(
                messages, TOOL_DEFINITIONS, max_retries=self.max_llm_retries
            )
            try:
                # 用 send(None) 而非 for 循环，以获取 generator 的 return 值
                while True:
                    try:
                        event = stream_gen.send(None)
                    except StopIteration as e:
                        response_data = e.value if e.value else None
                        break

                    etype = event.get("type", "")
                    if etype == "content":
                        yield StreamEvent(
                            event_type="answer_token", data={"content": event["delta"]}
                        )
                    elif etype == "tool_arg":
                        yield StreamEvent(
                            event_type="tool_arg",
                            data={
                                "index": event["index"],
                                "arguments_delta": event.get("arguments_delta", ""),
                            },
                        )
            except Exception as e:
                logger.error(f"[ReAct] 流式 LLM 调用异常: {e}")
                response_data = None

            t_think = time.time() - t_think_start

            if response_data is None:
                self.tool_executor.expand_accumulated_chunks(self._original_query)
                answer, syn_usage = self._synthesize_from_accumulated(query)
                _accumulate_usage(total_usage, syn_usage)
                yield StreamEvent(
                    event_type="error",
                    data={"error": "LLM 调用失败，已降级合成回答"},
                )
                for char in answer:
                    yield StreamEvent(event_type="answer_token", data={"content": char})
                self._collected_answer = answer
                self._collected_steps = steps
                yield StreamEvent(
                    event_type="done",
                    data=_build_done_data(
                        iteration + 1,
                        "error",
                        self.tool_executor.accumulated_chunks,
                        start_time,
                        total_usage,
                    ),
                )
                return

            assistant_msg = response_data["message"]
            usage = response_data.get("usage")
            _accumulate_usage(total_usage, usage)

            messages.append(assistant_msg)

            # ── 2. ANALYZE ──
            tool_calls = assistant_msg.get("tool_calls") or []

            # 推送 thought 内容到前端（LLM 推理内容）
            thought_content = (assistant_msg.get("content") or "").strip()
            if thought_content:
                yield StreamEvent(
                    event_type="thought",
                    data={
                        "iteration": iteration + 1,
                        "thought": thought_content[:300],
                    },
                )

            # 无工具调用 → 自然停止
            if not tool_calls:
                self.tool_executor.expand_accumulated_chunks(self._original_query)
                step = AgentStep(
                    iteration=iteration,
                    thought=thought_content,
                    duration=time.time() - step_start,
                )
                steps.append(step)

                yield StreamEvent(
                    event_type="step_end",
                    data={
                        "iteration": iteration + 1,
                        "action": "think",
                        "duration": round(step.duration, 2),
                        "timing": {"think": round(t_think, 3)},
                    },
                )

                for char in thought_content:
                    yield StreamEvent(event_type="answer_token", data={"content": char})
                self._collected_answer = thought_content
                self._collected_steps = steps
                yield StreamEvent(
                    event_type="done",
                    data=_build_done_data(
                        iteration + 1,
                        "natural_stop",
                        self.tool_executor.accumulated_chunks,
                        start_time,
                        total_usage,
                    ),
                )
                return

            # ── 卡住检测 ──
            current_sig = ";".join(
                f"{tc['function']['name']}({tc['function'].get('arguments', '')})"
                for tc in tool_calls
            )
            if current_sig == last_tool_calls_sig:
                consecutive_same += 1
            else:
                consecutive_same = 0
            last_tool_calls_sig = current_sig

            force_terminate = consecutive_same >= 2
            if force_terminate:
                logger.warning(
                    f"[ReAct] 卡住检测：连续 {consecutive_same} 轮相同工具调用"
                )

            # ── 3. ACT ──
            for tool_call in tool_calls:
                tc_id = tool_call.get("id", "")
                tc_function = tool_call.get("function", {})
                tool_name = tc_function.get("name", "")
                tool_args_str = tc_function.get("arguments", "{}")

                tool_args = _parse_json_tiered(tool_args_str)
                if tool_args is None:
                    tool_args = {}

                logger.info(
                    f"[ReAct] iter={iteration} tool={tool_name} "
                    f"args_keys={list(tool_args.keys())}"
                )

                # finish 工具 → 终止
                if tool_name == "finish":
                    # finish 时统一展开 parent chunks
                    t_expand_start = time.time()
                    self.tool_executor.expand_accumulated_chunks(self._original_query)
                    t_expand = time.time() - t_expand_start
                    answer = tool_args.get("answer", thought_content or "")
                    step = AgentStep(
                        iteration=iteration,
                        action="finish",
                        observation=f"提交最终回答（{len(answer)} 字）",
                        duration=time.time() - step_start,
                        timing={"think": round(t_think, 3), "expand": round(t_expand, 3)},
                    )
                    steps.append(step)

                    yield StreamEvent(
                        event_type="step_end",
                        data={
                            "iteration": iteration + 1,
                            "action": "finish",
                            "duration": round(step.duration, 2),
                            "timing": step.timing,
                        },
                    )

                    for char in answer:
                        yield StreamEvent(
                            event_type="answer_token", data={"content": char}
                        )
                    self._collected_answer = answer
                    self._collected_steps = steps
                    yield StreamEvent(
                        event_type="done",
                        data=_build_done_data(
                            iteration + 1,
                            "finish",
                            self.tool_executor.accumulated_chunks,
                            start_time,
                            total_usage,
                        ),
                    )
                    return

                # 执行工具
                t_tool_start = time.time()
                observation, truly_new_chunks = self.tool_executor.execute(
                    tool_name, tool_args
                )
                t_tool = time.time() - t_tool_start

                # 无实质新信息检测
                if tool_name in ("bm25_search", "vector_search"):
                    if len(truly_new_chunks) == 0:
                        consecutive_empty += 1
                        logger.info(
                            f"[ReAct] iter={iteration} 工具={tool_name} 无实质新信息 "
                            f"(consecutive_empty={consecutive_empty})"
                        )
                    else:
                        consecutive_empty = 0

                step = AgentStep(
                    iteration=iteration,
                    action=tool_name,
                    action_input=tool_args,
                    observation=observation[:500],
                    duration=time.time() - step_start,
                    timing={
                        "think": round(t_think, 3),
                        "tool": round(t_tool, 3),
                    },
                )
                steps.append(step)

                yield StreamEvent(
                    event_type="step_end",
                    data={
                        "iteration": iteration + 1,
                        "action": tool_name,
                        "duration": round(step.duration, 2),
                        "timing": step.timing,
                    },
                )

                # OBSERVE
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": observation,
                    }
                )

            # 无实质新信息 → 强制终止
            if consecutive_empty >= 2:
                logger.warning(
                    f"[ReAct] 连续 {consecutive_empty} 次无实质新信息，强制终止"
                )
                thought_content = ""  # 重置避免引用未定义变量
                self.tool_executor.expand_accumulated_chunks(self._original_query)
                answer, syn_usage = self._synthesize_from_accumulated(query)
                _accumulate_usage(total_usage, syn_usage)

                yield StreamEvent(
                    event_type="error",
                    data={"error": "Agent 卡住，已降级合成回答"},
                )
                for char in answer:
                    yield StreamEvent(event_type="answer_token", data={"content": char})
                self._collected_answer = answer
                self._collected_steps = steps
                yield StreamEvent(
                    event_type="done",
                    data=_build_done_data(
                        iteration + 1,
                        "stuck",
                        self.tool_executor.accumulated_chunks,
                        start_time,
                        total_usage,
                    ),
                )
                return

        # 超过最大迭代数 → 优雅降级
        logger.warning(f"[ReAct] 达到最大迭代数 {self.max_iterations}，优雅降级")
        self.tool_executor.expand_accumulated_chunks(self._original_query)
        answer, syn_usage = self._synthesize_from_accumulated(query)
        _accumulate_usage(total_usage, syn_usage)

        yield StreamEvent(
            event_type="error",
            data={"error": f"达到最大迭代数 {self.max_iterations}，已降级合成回答"},
        )
        for char in answer:
            yield StreamEvent(event_type="answer_token", data={"content": char})
        self._collected_answer = answer
        self._collected_steps = steps
        yield StreamEvent(
            event_type="done",
            data=_build_done_data(
                self.max_iterations,
                "max_iterations",
                self.tool_executor.accumulated_chunks,
                start_time,
                total_usage,
            ),
        )

    # ── 空响应重试 ──────────────────────────────────────────────────────

    def _retry_empty_response(
        self,
        messages: List[Dict[str, Any]],
        total_usage: TokenUsage,
    ) -> Tuple[Optional[list], str, Dict[str, Any]]:
        """
        当 LLM 返回空响应时，追加引导消息重试。

        Returns:
            (tool_calls, content, assistant_msg) — 可能仍为空
        """
        empty_retries = 0
        max_empty_retries = 2
        tool_calls = None
        content = ""
        assistant_msg: Dict[str, Any] = {}

        while empty_retries < max_empty_retries:
            messages.append(
                {
                    "role": "user",
                    "content": "请调用工具检索信息，然后使用 finish 提交回答。",
                }
            )
            retry_resp = self.llm.call_with_tools(
                messages, TOOL_DEFINITIONS, max_retries=0
            )
            if retry_resp is None:
                break
            retry_msg = retry_resp["message"]
            retry_usage = retry_resp.get("usage")
            _accumulate_usage(total_usage, retry_usage)
            messages.append(retry_msg)

            tool_calls = retry_msg.get("tool_calls")
            content = (retry_msg.get("content") or "").strip()
            if tool_calls or content:
                assistant_msg = retry_msg
                break
            empty_retries += 1

        return tool_calls, content, assistant_msg

    # ── 优雅降级：从累积 chunks 合成回答 ────────────────────────

    def _synthesize_from_accumulated(
        self, query: str
    ) -> Tuple[str, Optional[TokenUsage]]:
        """
        优雅降级模式：LLM 失败或超迭代时，
        从已累积的检索结果合成最终回答。
        """
        chunks = self.tool_executor.accumulated_chunks
        if isinstance(chunks, dict):
            chunk_list = list(chunks.values()) if chunks else []
        else:
            chunk_list = chunks
        if not chunk_list:
            return "抱歉，未能检索到相关信息来回答您的问题。", None

        # 构建简单的上下文（按 accumulated_chunks 顺序，截断 10000 字）
        context_parts = []
        for chunk in chunk_list:
            doc_name = chunk.doc_title or chunk.doc_id
            context_parts.append(f"[来源: {doc_name}]\n{chunk.content}")

        context = "\n\n---\n\n".join(context_parts)
        if len(context) > 16000:
            context = context[:16000] + "\n\n... [内容已截断] ..."

        # 复用 self.llm 做简单合成（不传 tools，避免再次进入循环）
        if not self.llm.api_key:
            # 无 API Key 时直接返回原始片段
            return (
                f"基于检索到的 {len(chunk_list)} 条信息：\n\n"
                + "\n".join(f"- {c.content[:200]}..." for c in chunk_list[:5]),
                None,
            )

        try:
            response_text = self.llm.call(
                [
                    {
                        "role": "system",
                        "content": "你是知识库助手。请基于以下检索到的信息片段，简要回答用户问题。"
                        "如果信息不足，请明确说明。",
                    },
                    {
                        "role": "user",
                        "content": f"问题：{query}\n\n检索到的信息：\n{context}",
                    },
                ],
                temperature=0.3,
            )
            return response_text, None
        except Exception as e:
            logger.warning(f"[ReAct] 降级合成也失败: {e}")
            return (
                f"基于检索到的 {len(chunks)} 条信息：\n\n"
                + "\n".join(
                    f"- [{c.doc_title or c.doc_id}] {c.content[:200]}..."
                    for c in chunks[:5]
                ),
                None,
            )
