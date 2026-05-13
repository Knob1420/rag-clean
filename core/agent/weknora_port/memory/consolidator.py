"""
WeKnora Faithful Port — Memory Consolidator

Ported from WeKnora internal/agent/memory/consolidator.go

When context window grows too large, older messages are summarized
by the LLM into a compact memory block that preserves key facts.
"""

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from core.agent.weknora_port.memory.estimator import Estimator
from core.agent.weknora_port.const import (
    DEFAULT_CONSOLIDATION_THRESHOLD,
    CONSOLIDATION_TIMEOUT,
    MAX_CONSOLIDATION_ATTEMPTS,
)

# ── Consolidation system prompt (verbatim from WeKnora) ──────────────
CONSOLIDATION_SYSTEM_PROMPT = (
    "You are a conversation summarizer. "
    "Your task is to create a concise but comprehensive summary "
    "of a conversation between a user and an AI assistant.\n\n"
    "The summary should:\n"
    "- Be written in the same language as the original conversation\n"
    "- Preserve all key facts, numbers, and specific details\n"
    "- Include the outcomes of any tool executions\n"
    "- Note any errors or issues encountered\n"
    "- Be structured with clear sections if the conversation covered multiple topics\n"
    "- Be concise — aim for 30% or less of the original length\n\n"
    "Output only the summary, no preamble or explanation."
)


class Consolidator:
    """
    Compresses agent conversation history using LLM summarization.

    Two-tier context management:
    - Consolidation (soft): LLM summarization at 50% of context window
    - Compression (hard): Truncation at 80% (handled separately in token/compress.py)
    """

    def __init__(
        self,
        llm_call_fn,  # callable: (messages, temperature, max_tokens) -> str
        estimator: Estimator,
        max_context_tokens: int,
        threshold: float = DEFAULT_CONSOLIDATION_THRESHOLD,
    ):
        self._llm_call = llm_call_fn
        self.estimator = estimator
        self.max_tokens = max_context_tokens
        self.threshold = threshold if 0 < threshold < 1 else DEFAULT_CONSOLIDATION_THRESHOLD

    def should_consolidate(self, current_tokens: int) -> bool:
        """Check if consolidation is needed based on current token estimate."""
        if self.max_tokens <= 0:
            return False
        trigger_at = int(self.max_tokens * self.threshold)
        return current_tokens > trigger_at

    def consolidate(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Summarize older messages and return a compressed message array.

        Preserves:
        - The system prompt (first message)
        - The current turn (last user message + subsequent assistant/tool messages)
        - Recent history that fits within the token budget

        Older history is replaced with a summary system message.
        Falls back to raw text archiving if LLM summarization fails.
        """
        if len(messages) <= 3:
            return messages

        system_msg = messages[0]

        # Find the current user query — the last message with role "user"
        last_user_idx = 0
        for i in range(len(messages) - 1, 0, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx <= 1:
            return messages

        history = messages[1:last_user_idx]
        tail = messages[last_user_idx:]

        if len(history) < 2:
            return messages

        target_tokens = int(self.max_tokens * self.threshold * 0.6)  # aim for 60% of threshold

        tail_tokens = sum(self.estimator.estimate_message(tail[i]) for i in range(len(tail)))

        keep_from_end = self._find_keep_boundary(
            history, target_tokens, system_msg, tail_tokens
        )

        if keep_from_end >= len(history):
            return messages

        to_consolidate = history[: len(history) - keep_from_end]
        to_keep = history[len(history) - keep_from_end :]

        summary = self._summarize_with_retry(to_consolidate)

        summary_msg = {
            "role": "system",
            "content": (
                f"[Memory Summary - {len(to_consolidate)} earlier messages consolidated]\n\n"
                f"{summary}"
            ),
        }

        result = [system_msg, summary_msg] + to_keep + tail

        logger.info(
            f"[Consolidator] Consolidated {len(to_consolidate)} messages → "
            f"summary ({len(summary)} chars), "
            f"keeping {len(to_keep)} history + {len(tail)} current-turn messages"
        )

        return result

    def _find_keep_boundary(
        self,
        history: List[Dict[str, Any]],
        target_tokens: int,
        system_msg: Dict[str, Any],
        tail_tokens: int,
    ) -> int:
        """
        Determine how many messages from the end of history to keep.
        Respects tool_call/tool_result message pair boundaries.
        """
        budget = target_tokens - self.estimator.estimate_message(system_msg) - tail_tokens - 500

        if budget <= 0:
            return 0

        tokens = 0
        keep_count = 0
        i = len(history) - 1

        while i >= 0:
            msg = history[i]
            msg_tokens = self.estimator.estimate_message(msg)

            if msg.get("role") == "tool":
                # Group tool messages together with their preceding assistant message
                group_tokens = msg_tokens
                group_size = 1
                j = i - 1
                while j >= 0 and history[j].get("role") == "tool":
                    group_tokens += self.estimator.estimate_message(history[j])
                    group_size += 1
                    j -= 1
                if j >= 0 and history[j].get("role") == "assistant":
                    group_tokens += self.estimator.estimate_message(history[j])
                    group_size += 1

                if tokens + group_tokens > budget:
                    break
                tokens += group_tokens
                keep_count += group_size
                i -= group_size
            else:
                if tokens + msg_tokens > budget:
                    break
                tokens += msg_tokens
                keep_count += 1
                i -= 1

        return keep_count

    def _summarize_with_retry(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Attempt LLM summarization with retries. Falls back to raw archive."""
        prompt = self._build_consolidation_prompt(messages)
        last_err = None

        for attempt in range(1, MAX_CONSOLIDATION_ATTEMPTS + 1):
            try:
                resp = self._llm_call(
                    [
                        {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=2000,
                )
                if resp and resp.strip():
                    return resp.strip()
                last_err = "empty response from LLM"
            except Exception as e:
                last_err = e
                logger.warning(
                    f"[Consolidator] Summarization attempt {attempt}/{MAX_CONSOLIDATION_ATTEMPTS} failed: {e}"
                )

        # Fallback to raw archive
        logger.warning(
            f"[Consolidator] LLM summarization failed after {MAX_CONSOLIDATION_ATTEMPTS} attempts, "
            f"falling back to raw archive: {last_err}"
        )
        return self._raw_archive(messages)

    def _build_consolidation_prompt(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Build the prompt for LLM to summarize messages."""
        parts = [
            "Summarize the following conversation history, preserving:",
            "1. Key facts and decisions made",
            "2. Tool execution results and their outcomes",
            "3. User's original intent and requirements",
            "4. Any errors encountered and how they were resolved",
            "",
            "Conversation to summarize:",
            "",
        ]

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""

            if role == "user":
                parts.append(f"**User**: {_truncate_for_prompt(content, 2000)}")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    names = [tc.get("function", {}).get("name", "") for tc in tool_calls]
                    parts.append(
                        f"**Assistant** [called tools: {', '.join(names)}]: "
                        f"{_truncate_for_prompt(content, 1000)}"
                    )
                else:
                    parts.append(f"**Assistant**: {_truncate_for_prompt(content, 2000)}")
            elif role == "tool":
                tool_name = msg.get("name", "unknown")
                parts.append(f"**Tool [{tool_name}]**: {_truncate_for_prompt(content, 1000)}")

        return "\n\n".join(parts)

    @staticmethod
    def _raw_archive(messages: List[Dict[str, Any]]) -> str:
        """Create a simple text dump as fallback when LLM fails."""
        parts = ["Raw conversation archive (LLM summarization unavailable):\n"]

        for msg in messages:
            content = _truncate_for_prompt(msg.get("content", "") or "", 500)
            role = msg.get("role", "")

            if role == "user":
                parts.append(f"- User: {content}")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    names = [tc.get("function", {}).get("name", "") for tc in tool_calls]
                    parts.append(f"- Assistant [tools: {','.join(names)}]: {content}")
                else:
                    parts.append(f"- Assistant: {content}")
            elif role == "tool":
                parts.append(f"- Tool[{msg.get('name', 'unknown')}]: {content}")

        return "\n".join(parts)


def _truncate_for_prompt(s: str, max_len: int) -> str:
    """Truncate a string to max_len characters for use in prompts."""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."
