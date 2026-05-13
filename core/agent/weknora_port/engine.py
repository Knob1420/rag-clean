"""
WeKnora Faithful Port — AgentEngine

Ported from WeKnora:
- internal/agent/engine.go (AgentEngine, Execute, executeLoop, runReActIteration)
- internal/agent/think.go (streamThinkingToEventBus, callLLMWithRetry)
- internal/agent/act.go (executeToolCalls)
- internal/agent/observe.go (analyzeResponse, appendToolResults, buildRuntimeContextBlock)
- internal/agent/finalize.go (handleMaxIterations, streamFinalAnswerToEventBus)

Full WeKnora architecture:
- Think → Analyze → Act → Observe cycle
- Streaming LLM call with StripThinkBlocks
- call_llm_with_retry with transient error detection
- analyze_response → responseVerdict (stop/content_filter/final_answer/tool_calls)
- execute_tool_calls with parallel support (ThreadPoolExecutor)
- Context window management: consolidation + compression
- Stuck loop detection (consecutiveSameContent)
- Empty response retry
- Graceful degradation
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from loguru import logger

from core.agent.weknora_port.const import (
    DEFAULT_AGENT_MAX_ITERATIONS,
    DEFAULT_AGENT_TEMPERATURE,
    DEFAULT_CONTEXT_TOKENS,
    MAX_EMPTY_RESPONSE_RETRIES,
    MAX_LLM_RETRIES,
    MAX_REPEATED_RESPONSE_ROUNDS,
    TRANSIENT_ERROR_MARKERS,
)
from core.agent.weknora_port.strip_think import strip_think_blocks
from core.agent.weknora_port.prompts import (
    KnowledgeBaseInfo,
    SelectedDocumentInfo,
    build_system_prompt_with_options,
    build_runtime_context_block,
    build_messages_with_llm_context,
    redact_history_kb_results,
)
from core.agent.weknora_port.tools import create_weknora_tool_registry
from core.agent.weknora_port.tools.registry import ToolRegistry
from core.agent.weknora_port.tools.final_answer import (
    final_answer_handler,
    parse_final_answer_args,
)
from core.agent.weknora_port.memory.estimator import Estimator
from core.agent.weknora_port.memory.consolidator import Consolidator
from core.agent.weknora_port.token.compress import compress_context
from core.generation.llm import LLMClient, get_llm_client
from core.retrieve.retrieval import RetrievalService
from core.retrieve.retrieval_models import RetrievedChunk, TokenUsage


# ══════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════


@dataclass
class AgentStep:
    """Single ReAct step record."""
    iteration: int
    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    duration: float = 0.0


@dataclass
class StreamEvent:
    """Agent streaming event for SSE push."""
    event_type: str  # step_start | step_end | answer_token | done | error
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """WeKnora-port Agent execution result."""
    answer: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    chunks: List[RetrievedChunk] = field(default_factory=list)
    total_iterations: int = 0
    timing: Dict[str, float] = field(default_factory=dict)
    usage: Optional[TokenUsage] = None
    terminated_reason: str = ""  # "final_answer" | "max_iterations" | "stuck" | "error" | "content_filter" | "natural_stop"


@dataclass
class ResponseVerdict:
    """Result of analyzing an LLM response."""
    has_tool_calls: bool = False
    has_final_answer: bool = False
    is_stop: bool = False  # natural stop (no tool calls, no finish)
    is_content_filter: bool = False
    is_empty: bool = False
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════


def _accumulate_usage(total: TokenUsage, delta: Optional[TokenUsage]) -> None:
    """Accumulate token usage (in-place)."""
    if delta is None:
        return
    total.prompt_tokens += delta.prompt_tokens
    total.completion_tokens += delta.completion_tokens
    total.total_tokens += delta.total_tokens


def _build_result(
    answer: str,
    steps: List[AgentStep],
    chunks: List[RetrievedChunk],
    iteration: int,
    start_time: float,
    total_usage: TokenUsage,
    reason: str,
) -> AgentResult:
    """Build an AgentResult."""
    return AgentResult(
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
    """Build data for done event."""
    return {
        "iterations": iteration,
        "terminated_reason": reason,
        "chunks_count": len(chunks),
        "time": {"total": time.time() - start_time},
        "usage": total_usage.__dict__,
    }


# ══════════════════════════════════════════════════════════════════
# AgentEngine — Core ReAct loop (faithful WeKnora port)
# ══════════════════════════════════════════════════════════════════


class AgentEngine:
    """
    WeKnora AgentEngine: full ReAct loop with Think→Analyze→Act→Observe.

    Key features ported from WeKnora:
    - StripThinkBlocks on LLM output
    - Transient error retry with linear backoff
    - Parallel tool call execution
    - Context window management (consolidation + compression)
    - Stuck loop detection (consecutive same-content without tool calls)
    - Empty response retry with guidance message
    - Graceful degradation on max iterations
    - History redaction for old KB results
    - Runtime context block in user messages
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retrieval_service: Optional[RetrievalService] = None,
        max_iterations: int = DEFAULT_AGENT_MAX_ITERATIONS,
        max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        knowledge_bases: Optional[List[KnowledgeBaseInfo]] = None,
        selected_docs: Optional[List[SelectedDocumentInfo]] = None,
        language: str = "",
        session_id: Optional[str] = None,
    ):
        self.llm = llm_client or get_llm_client()
        self._retrieval = retrieval_service or RetrievalService()
        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens

        # KB context
        self.knowledge_bases = knowledge_bases or []
        self.selected_docs = selected_docs or []
        self.language = language
        self.session_id = session_id

        # Token management
        self.estimator = Estimator()
        self.consolidator = Consolidator(
            llm_call_fn=self._llm_call_simple,
            estimator=self.estimator,
            max_context_tokens=max_context_tokens,
        )

        # Shared tool state
        self._accumulated_chunks: List[RetrievedChunk] = []
        self._already_seen: Set[str] = set()
        self._current_query: str = ""

        # Tool registry
        self._registry: Optional[ToolRegistry] = None

        # Collection for non-streaming mode
        self._collected_answer = ""
        self._collected_steps: List[AgentStep] = []

    def _ensure_registry(self) -> ToolRegistry:
        """Lazily create tool registry with current state."""
        if self._registry is None:
            self._registry = create_weknora_tool_registry(
                retrieval_service=self._retrieval,
                accumulated_chunks=self._accumulated_chunks,
                already_seen=self._already_seen,
                current_query=self._current_query,
            )
        return self._registry

    # ── Public API ──────────────────────────────────────────────────────

    def execute(self, query: str) -> AgentResult:
        """
        Execute the ReAct loop (non-streaming), consuming all stream events.

        Returns:
            AgentResult with answer, steps, chunks, timing, usage
        """
        result: Optional[AgentResult] = None

        for event in self.execute_stream(query):
            if event.event_type == "done":
                data = event.data
                result = AgentResult(
                    answer=self._collected_answer,
                    steps=self._collected_steps,
                    chunks=list(self._accumulated_chunks),
                    total_iterations=data.get("iterations", 0),
                    timing=data.get("time", {}),
                    usage=TokenUsage(**data["usage"]) if data.get("usage") else None,
                    terminated_reason=data.get("terminated_reason", ""),
                )

        if result is None:
            result = AgentResult(terminated_reason="error")

        return result

    def execute_stream(self, query: str) -> Generator[StreamEvent, None, None]:
        """
        Stream the ReAct loop, yielding StreamEvent per step.

        Event types:
        - step_start: iteration begins
        - step_end: iteration ends
        - answer_token: final answer tokens
        - done: entire agent finished
        - error: error occurred
        - debug: internal debug info (raw LLM response, think blocks, etc.)
        """
        start_time = time.time()
        self._reset_state(query)

        # Build initial messages
        messages = build_messages_with_llm_context(
            query=query,
            knowledge_bases=self.knowledge_bases,
            selected_docs=self.selected_docs,
            language=self.language,
            session_id=self.session_id,
        )

        steps: List[AgentStep] = []
        total_usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        consecutive_same = 0
        last_content_sig = ""
        empty_response_rounds = 0

        for iteration in range(self.max_iterations):
            step_start = time.time()

            # Notify: new round
            yield StreamEvent(
                event_type="step_start",
                data={"iteration": iteration + 1, "max_iterations": self.max_iterations},
            )

            # ── 1. THINK ──
            response_data = self._call_llm_with_retry(messages)

            if response_data is None:
                # LLM call failed after retries → graceful degradation
                answer, syn_usage = self._synthesize_from_accumulated(query)
                _accumulate_usage(total_usage, syn_usage)
                yield StreamEvent(
                    event_type="error",
                    data={"error": "LLM call failed, degraded synthesis"},
                )
                for char in answer:
                    yield StreamEvent(event_type="answer_token", data={"content": char})
                self._collected_answer = answer
                self._collected_steps = steps
                yield StreamEvent(
                    event_type="done",
                    data=_build_done_data(
                        iteration + 1, "error",
                        self._accumulated_chunks, start_time, total_usage,
                    ),
                )
                return

            assistant_msg = response_data["message"]
            usage = response_data.get("usage")
            _accumulate_usage(total_usage, usage)

            # Strip think blocks from content
            raw_content = assistant_msg.get("content", "") or ""
            content = strip_think_blocks(raw_content)
            think_stripped = (content != raw_content)
            if think_stripped:
                assistant_msg = dict(assistant_msg)
                assistant_msg["content"] = content if content else None

            # Yield debug event with raw LLM response (before strip)
            yield StreamEvent(
                event_type="debug",
                data={
                    "raw_content_len": len(raw_content),
                    "cleaned_content_len": len(content),
                    "think_stripped": think_stripped,
                    "think_content": raw_content[:3000] if think_stripped else "",
                    "cleaned_content_preview": content[:1000],
                    "tool_calls_count": len(assistant_msg.get("tool_calls") or []),
                    "usage": usage.__dict__ if usage else None,
                },
            )

            messages.append(assistant_msg)

            # ── 2. ANALYZE ──
            verdict = self._analyze_response(assistant_msg)

            # Empty response retry (WeKnora pattern)
            if verdict.is_empty:
                empty_response_rounds += 1
                if empty_response_rounds <= MAX_EMPTY_RESPONSE_RETRIES:
                    messages.append({
                        "role": "user",
                        "content": "请调用工具检索信息，然后使用 final_answer 提交回答。",
                    })
                    logger.info(f"[Engine] Empty response, retry {empty_response_rounds}/{MAX_EMPTY_RESPONSE_RETRIES}")
                    continue
                else:
                    # Too many empty responses → degrade
                    answer, syn_usage = self._synthesize_from_accumulated(query)
                    _accumulate_usage(total_usage, syn_usage)
                    yield StreamEvent(
                        event_type="error",
                        data={"error": "Agent stuck with empty responses"},
                    )
                    for char in answer:
                        yield StreamEvent(event_type="answer_token", data={"content": char})
                    self._collected_answer = answer
                    self._collected_steps = steps
                    yield StreamEvent(
                        event_type="done",
                        data=_build_done_data(
                            iteration + 1, "stuck",
                            self._accumulated_chunks, start_time, total_usage,
                        ),
                    )
                    return

            empty_response_rounds = 0  # reset on non-empty

            # Content filter detection
            if verdict.is_content_filter:
                logger.warning("[Engine] Content filter triggered by LLM")
                self._collected_answer = "抱歉，模型因安全策略拒绝了响应。"
                self._collected_steps = steps
                yield StreamEvent(
                    event_type="done",
                    data=_build_done_data(
                        iteration + 1, "content_filter",
                        self._accumulated_chunks, start_time, total_usage,
                    ),
                )
                return

            # Natural stop (no tool calls, no final_answer)
            if verdict.is_stop:
                step = AgentStep(
                    iteration=iteration,
                    thought=verdict.content,
                    duration=time.time() - step_start,
                )
                steps.append(step)

                yield StreamEvent(
                    event_type="step_end",
                    data={
                        "iteration": iteration + 1,
                        "action": "think",
                        "duration": round(step.duration, 2),
                        "thought": verdict.content[:2000],
                        "tool_calls": [],
                        "observation_preview": "",
                    },
                )

                for char in verdict.content:
                    yield StreamEvent(event_type="answer_token", data={"content": char})
                self._collected_answer = verdict.content
                self._collected_steps = steps
                yield StreamEvent(
                    event_type="done",
                    data=_build_done_data(
                        iteration + 1, "natural_stop",
                        self._accumulated_chunks, start_time, total_usage,
                    ),
                )
                return

            # Final answer detection
            if verdict.has_final_answer:
                answer = verdict.content
                step = AgentStep(
                    iteration=iteration,
                    action="final_answer",
                    observation=f"Final answer submitted ({len(answer)} chars)",
                    duration=time.time() - step_start,
                )
                steps.append(step)

                yield StreamEvent(
                    event_type="step_end",
                    data={
                        "iteration": iteration + 1,
                        "action": "final_answer",
                        "duration": round(step.duration, 2),
                        "thought": verdict.content[:2000],
                        "tool_calls": [{"name": "final_answer", "args_preview": f"answer ({len(answer)} chars)"}],
                        "observation_preview": f"Final answer submitted ({len(answer)} chars)",
                    },
                )

                for char in answer:
                    yield StreamEvent(event_type="answer_token", data={"content": char})
                self._collected_answer = answer
                self._collected_steps = steps
                yield StreamEvent(
                    event_type="done",
                    data=_build_done_data(
                        iteration + 1, "final_answer",
                        self._accumulated_chunks, start_time, total_usage,
                    ),
                )
                return

            # ── Stuck loop detection ──
            tool_calls = verdict.tool_calls
            current_sig = ";".join(
                f"{tc.get('function', {}).get('name', '')}({tc.get('function', {}).get('arguments', '')})"
                for tc in tool_calls
            )
            if current_sig == last_content_sig and current_sig:
                consecutive_same += 1
            else:
                consecutive_same = 0
            last_content_sig = current_sig

            force_terminate = consecutive_same >= MAX_REPEATED_RESPONSE_ROUNDS

            # ── 3. ACT ──
            tool_results = self._execute_tool_calls(tool_calls)

            # Build step record
            tool_names = [tc.get("function", {}).get("name", "") for tc in tool_calls]
            observation_preview = "; ".join(
                r.get("content", "")[:200] for r in tool_results
            )

            step = AgentStep(
                iteration=iteration,
                action=", ".join(tool_names),
                action_input={tc.get("function", {}).get("name", ""): tc.get("function", {}).get("arguments", "")
                             for tc in tool_calls},
                observation=observation_preview[:500],
                duration=time.time() - step_start,
            )
            steps.append(step)

            yield StreamEvent(
                event_type="step_end",
                data={
                    "iteration": iteration + 1,
                    "action": ", ".join(tool_names),
                    "duration": round(step.duration, 2),
                    "thought": verdict.content[:2000],
                    "tool_calls": [
                        {
                            "name": tc.get("function", {}).get("name", ""),
                            "args_preview": tc.get("function", {}).get("arguments", "")[:300],
                        }
                        for tc in tool_calls
                    ],
                    "observation_preview": observation_preview[:200],
                },
            )

            # ── 4. OBSERVE ──
            # Check for final_answer in tool results
            for i, (tc, tr) in enumerate(zip(tool_calls, tool_results)):
                tool_name = tc.get("function", {}).get("name", "")
                if tool_name == "final_answer":
                    answer = tr.get("content", "")
                    self._collected_answer = answer
                    self._collected_steps = steps
                    yield StreamEvent(
                        event_type="done",
                        data=_build_done_data(
                            iteration + 1, "final_answer",
                            self._accumulated_chunks, start_time, total_usage,
                        ),
                    )
                    return

            # Append tool results to messages
            self._append_tool_results(messages, tool_calls, tool_results)

            # ── Context window management ──
            self._manage_context_window(messages, total_usage)

            # ── History redaction ──
            messages = redact_history_kb_results(messages, keep_recent=4)

            # ── Stuck loop termination ──
            if force_terminate:
                logger.warning(f"[Engine] Stuck loop detected: {consecutive_same} consecutive identical tool calls")
                answer, syn_usage = self._synthesize_from_accumulated(query)
                _accumulate_usage(total_usage, syn_usage)
                yield StreamEvent(
                    event_type="error",
                    data={"error": "Agent stuck, degraded synthesis"},
                )
                for char in answer:
                    yield StreamEvent(event_type="answer_token", data={"content": char})
                self._collected_answer = answer
                self._collected_steps = steps
                yield StreamEvent(
                    event_type="done",
                    data=_build_done_data(
                        iteration + 1, "stuck",
                        self._accumulated_chunks, start_time, total_usage,
                    ),
                )
                return

        # ── Max iterations reached → graceful degradation ──
        logger.warning(f"[Engine] Max iterations reached: {self.max_iterations}")
        answer, syn_usage = self._synthesize_from_accumulated(query)
        _accumulate_usage(total_usage, syn_usage)

        yield StreamEvent(
            event_type="error",
            data={"error": f"Max iterations reached ({self.max_iterations}), degraded synthesis"},
        )
        for char in answer:
            yield StreamEvent(event_type="answer_token", data={"content": char})
        self._collected_answer = answer
        self._collected_steps = steps
        yield StreamEvent(
            event_type="done",
            data=_build_done_data(
                self.max_iterations, "max_iterations",
                self._accumulated_chunks, start_time, total_usage,
            ),
        )

    # ── Internal: State management ──────────────────────────────────

    def _reset_state(self, query: str) -> None:
        """Reset shared state for a new query."""
        self._accumulated_chunks = []
        self._already_seen = set()
        self._current_query = query
        self._registry = None  # force re-creation with new state
        self._collected_answer = ""
        self._collected_steps = []

    # ── Internal: LLM calls ─────────────────────────────────────────

    def _call_llm_with_retry(
        self,
        messages: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Call LLM with tools, retrying on transient errors.

        Port of WeKnora callLLMWithRetry.
        """
        registry = self._ensure_registry()
        tool_defs = registry.get_function_definitions()

        for attempt in range(MAX_LLM_RETRIES + 1):
            try:
                return self.llm.call_with_tools(
                    messages=messages,
                    tools=tool_defs,
                    max_retries=0,  # We handle retries ourselves
                    temperature=DEFAULT_AGENT_TEMPERATURE,
                    max_tokens=4096,
                )
            except Exception as e:
                error_str = str(e).lower()
                is_transient = any(marker in error_str for marker in TRANSIENT_ERROR_MARKERS)

                if is_transient and attempt < MAX_LLM_RETRIES:
                    wait_time = attempt + 1  # linear backoff: 1s, 2s
                    logger.warning(
                        f"[Engine] Transient error (attempt {attempt+1}/{MAX_LLM_RETRIES+1}): "
                        f"{str(e)[:100]}, retrying in {wait_time}s"
                    )
                    time.sleep(wait_time)
                    continue

                logger.error(f"[Engine] LLM call failed: {e}")
                return None

        return None

    def _llm_call_simple(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        """Simple LLM call for consolidation (no tools)."""
        return self.llm.call(messages, temperature=temperature)

    # ── Internal: Response analysis ─────────────────────────────────

    def _analyze_response(self, assistant_msg: Dict[str, Any]) -> ResponseVerdict:
        """
        Analyze LLM response to determine what to do next.

        Port of WeKnora analyzeResponse.
        """
        content = (assistant_msg.get("content") or "").strip()
        tool_calls = assistant_msg.get("tool_calls")

        # Strip think blocks
        content = strip_think_blocks(content)

        # Content filter detection
        if self._is_content_filter(content):
            return ResponseVerdict(is_content_filter=True, content=content)

        # Empty response
        if not tool_calls and not content:
            return ResponseVerdict(is_empty=True)

        # Has tool calls
        if tool_calls:
            # Check if any tool call is final_answer
            has_final = any(
                tc.get("function", {}).get("name") == "final_answer"
                for tc in tool_calls
            )
            if has_final:
                # Extract answer from final_answer tool call
                for tc in tool_calls:
                    if tc.get("function", {}).get("name") == "final_answer":
                        raw_args = tc.get("function", {}).get("arguments", "{}")
                        parsed = parse_final_answer_args(raw_args)
                        answer = parsed.get("answer", content)
                        return ResponseVerdict(
                            has_final_answer=True,
                            content=answer,
                            tool_calls=tool_calls,
                        )

            return ResponseVerdict(
                has_tool_calls=True,
                content=content,
                tool_calls=tool_calls,
            )

        # Natural stop (no tool calls)
        return ResponseVerdict(is_stop=True, content=content)

    def _is_content_filter(self, content: str) -> bool:
        """Detect content filter refusal patterns."""
        if not content:
            return False
        filter_patterns = [
            "I cannot fulfill",
            "I'm unable to",
            "content violates",
            "against my guidelines",
            "I must decline",
        ]
        lower = content.lower()
        return any(p.lower() in lower for p in filter_patterns)

    # ── Internal: Tool execution ────────────────────────────────────

    def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls, with parallel support.

        Port of WeKnora executeToolCalls (errgroup-based parallel → ThreadPoolExecutor).

        Returns list of {"tool_call_id": str, "content": str} dicts.
        """
        registry = self._ensure_registry()

        if len(tool_calls) <= 1:
            # Single tool call: execute directly
            results = []
            for tc in tool_calls:
                result = self._execute_single_tool(registry, tc)
                results.append(result)
            return results

        # Parallel execution (WeKnora errgroup pattern → ThreadPoolExecutor)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results_dict: Dict[int, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 5)) as pool:
            future_to_idx = {
                pool.submit(self._execute_single_tool, registry, tc): i
                for i, tc in enumerate(tool_calls)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_dict[idx] = future.result()
                except Exception as e:
                    tc = tool_calls[idx]
                    tc_id = tc.get("id", "")
                    results_dict[idx] = {
                        "tool_call_id": tc_id,
                        "content": f"Tool execution error: {str(e)}\n\n[Analyze the error above and try a different approach.]",
                    }

        # Maintain original order
        return [results_dict[i] for i in range(len(tool_calls))]

    def _execute_single_tool(
        self,
        registry: ToolRegistry,
        tool_call: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single tool call and return result dict."""
        tc_id = tool_call.get("id", "")
        tc_function = tool_call.get("function", {})
        tool_name = tc_function.get("name", "")
        tool_args_str = tc_function.get("arguments", "{}")

        # Parse arguments (3-tier JSON parsing)
        from core.agent.react_agent import _parse_json_tiered
        tool_args = _parse_json_tiered(tool_args_str)
        if tool_args is None:
            tool_args = {}

        logger.info(
            f"[Engine] tool={tool_name} args_keys={list(tool_args.keys())}"
        )

        # Special handling for final_answer
        if tool_name == "final_answer":
            parsed = parse_final_answer_args(tool_args_str)
            answer = parsed.get("answer", "")
            return {"tool_call_id": tc_id, "content": answer}

        # Execute via registry
        result = registry.execute_tool(tool_name, tool_args)

        return {"tool_call_id": tc_id, "content": result}

    def _append_tool_results(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> None:
        """
        Append tool results to messages following OpenAI tool calling format.

        Port of WeKnora appendToolResults.
        """
        for tr in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tr.get("tool_call_id", ""),
                "content": tr.get("content", ""),
            })

    # ── Internal: Context window management ─────────────────────────

    def _manage_context_window(
        self,
        messages: List[Dict[str, Any]],
        total_usage: TokenUsage,
    ) -> None:
        """
        Two-tier context window management (WeKnora pattern):
        1. Consolidation (soft): LLM summarization at 50% threshold
        2. Compression (hard): Truncation at 80% threshold
        """
        current_tokens = self._estimate_current_tokens(messages, total_usage)

        # Tier 1: Consolidation
        if self.consolidator.should_consolidate(current_tokens):
            logger.info(f"[Engine] Context consolidation triggered at {current_tokens} tokens")
            consolidated = self.consolidator.consolidate(messages)
            messages[:] = consolidated

        # Tier 2: Compression
        current_tokens = self._estimate_current_tokens(messages, total_usage)
        compressed = compress_context(
            messages=messages,
            estimator=self.estimator,
            max_tokens=self.max_context_tokens,
            current_tokens=current_tokens,
        )
        if compressed is not messages:
            logger.info(f"[Engine] Context compression applied: {len(messages)} → {len(compressed)} messages")
            messages[:] = compressed

    def _estimate_current_tokens(
        self,
        messages: List[Dict[str, Any]],
        total_usage: TokenUsage,
    ) -> int:
        """
        Estimate current context token count.

        Uses API Usage as primary source, Estimator as supplementary.
        """
        # If we have usage from the API, use that as baseline
        if total_usage.prompt_tokens > 0:
            return total_usage.prompt_tokens

        # Fallback: estimate from messages
        return self.estimator.estimate_messages(messages)

    # ── Internal: Graceful degradation ──────────────────────────────

    def _synthesize_from_accumulated(
        self, query: str
    ) -> Tuple[str, Optional[TokenUsage]]:
        """
        WeKnora graceful degradation: synthesize answer from accumulated chunks.

        Called when:
        - LLM call fails after retries
        - Max iterations reached
        - Stuck loop detected
        """
        chunks = self._accumulated_chunks
        if not chunks:
            return "抱歉，未能检索到相关信息来回答您的问题。", None

        # Build context from chunks
        context_parts = []
        for chunk in chunks[:10]:
            doc_name = chunk.doc_title or chunk.doc_id
            context_parts.append(f"[来源: {doc_name}]\n{chunk.content[:500]}")

        context = "\n\n---\n\n".join(context_parts)

        if not self.llm.api_key:
            return (
                f"基于检索到的 {len(chunks)} 条信息：\n\n"
                + "\n".join(f"- {c.content[:200]}..." for c in chunks[:5]),
                None,
            )

        try:
            response_text = self.llm.call(
                [
                    {
                        "role": "system",
                        "content": "你是知识库助手。请基于以下检索到的信息片段，简要回答用户问题。如果信息不足，请明确说明。",
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
            logger.warning(f"[Engine] Degraded synthesis also failed: {e}")
            return (
                f"基于检索到的 {len(chunks)} 条信息：\n\n"
                + "\n".join(
                    f"- [{c.doc_title or c.doc_id}] {c.content[:200]}..."
                    for c in chunks[:5]
                ),
                None,
            )
