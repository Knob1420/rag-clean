"""
WeKnora Faithful Port — BPE Token Estimator

Ported from WeKnora internal/agent/token/estimator.go

Uses tiktoken (cl100k_base) for BPE token estimation.
Falls back to char/4 heuristic on encoding failure.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None
    logger.warning("[Estimator] tiktoken not available, falling back to char/4 estimation")

# ── Overhead constants (matching WeKnora) ──────────────────────────
_PER_MESSAGE_OVERHEAD = 3   # tokens per message metadata
_PER_CONVERSATION_TAIL = 3  # tokens for conversation formatting
_PER_TOOL_CALL_OVERHEAD = 4  # tokens per tool_call entry


class Estimator:
    """
    Token estimator using BPE tokenization (cl100k_base).

    Used for incremental (delta) estimation between LLM calls.
    The authoritative token count comes from the API's Usage response.
    """

    def __init__(self):
        self._codec = _ENCODING

    def estimate_string(self, s: str) -> int:
        """
        Estimate token count for a string using BPE tokenization.
        Falls back to (len(s) + 3) // 4 on failure.
        """
        if not s:
            return 0

        if self._codec is not None:
            try:
                return len(self._codec.encode(s))
            except Exception:
                pass

        # Fallback: char/4 heuristic
        return (len(s) + 3) // 4

    def estimate_message(self, msg: Dict[str, Any]) -> int:
        """
        Estimate token count for a single message.

        Includes per-message overhead + role + content + name + tool_calls.
        """
        tokens = _PER_MESSAGE_OVERHEAD
        tokens += self.estimate_string(msg.get("role", ""))
        tokens += self.estimate_string(msg.get("content", "") or "")
        tokens += self.estimate_string(msg.get("name", "") or "")

        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            tokens += self.estimate_string(func.get("name", ""))
            tokens += self.estimate_string(func.get("arguments", ""))
            tokens += _PER_TOOL_CALL_OVERHEAD

        return tokens

    def estimate_messages(self, messages: List[Dict[str, Any]]) -> int:
        """
        Estimate token count for a full message array.
        Includes per-conversation tail overhead.
        """
        total = 0
        for msg in messages:
            total += self.estimate_message(msg)
        total += _PER_CONVERSATION_TAIL
        return total
