"""
WeKnora Faithful Port — Message Sanitization

Ported from WeKnora internal/agent/tools/sanitize.go

Fixes common message array issues before sending to LLM:
- Consecutive same-role messages (merge or insert alternating role)
- Orphaned tool results (tool message without preceding assistant tool_calls)
- Empty content fields
"""

from typing import Any, Dict, List

from loguru import logger


def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sanitize a message array for LLM API compliance.

    Fixes:
    1. Consecutive same-role messages → merge content or insert separator
    2. Orphaned tool results → remove or wrap with assistant message
    3. Empty content → add placeholder

    Args:
        messages: Raw message array

    Returns:
        Sanitized message array that complies with OpenAI API format
    """
    if not messages:
        return messages

    result: List[Dict[str, Any]] = []
    i = 0

    while i < len(messages):
        msg = dict(messages[i])  # shallow copy
        role = msg.get("role", "")

        # Fix empty content
        if not msg.get("content"):
            if role in ("user", "assistant"):
                msg["content"] = " "
            elif role == "system":
                msg["content"] = " "

        # Handle consecutive same-role messages
        if result and result[-1].get("role") == role:
            prev = result[-1]
            if role in ("user", "assistant"):
                # Merge content
                prev_content = prev.get("content", "") or ""
                curr_content = msg.get("content", "") or ""
                prev["content"] = prev_content + "\n" + curr_content

                # Merge tool_calls if both have them
                if msg.get("tool_calls"):
                    if prev.get("tool_calls"):
                        prev["tool_calls"] = prev["tool_calls"] + msg["tool_calls"]
                    else:
                        prev["tool_calls"] = msg["tool_calls"]

                logger.debug(f"[Sanitize] Merged consecutive {role} messages")
                i += 1
                continue
            elif role == "tool":
                # Two consecutive tool messages: insert assistant separator
                result.append({
                    "role": "assistant",
                    "content": " ",
                })
            elif role == "system":
                # Two consecutive system messages: merge
                prev_content = prev.get("content", "") or ""
                curr_content = msg.get("content", "") or ""
                prev["content"] = prev_content + "\n\n" + curr_content
                i += 1
                continue

        # Handle orphaned tool results
        if role == "tool":
            prev_role = result[-1].get("role", "") if result else ""
            prev_has_tool_calls = bool(result[-1].get("tool_calls")) if result else False

            if prev_role != "assistant" or not prev_has_tool_calls:
                # Insert a synthetic assistant message with tool_calls
                tool_call_id = msg.get("tool_call_id", "orphan_" + str(i))
                result.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "orphan_tool",
                            "arguments": "{}",
                        },
                    }],
                })
                logger.debug(f"[Sanitize] Inserted assistant wrapper for orphaned tool result at index {i}")

        # Ensure tool messages have tool_call_id
        if role == "tool" and not msg.get("tool_call_id"):
            msg["tool_call_id"] = f"auto_{i}"

        result.append(msg)
        i += 1

    # Final check: if last message is assistant with tool_calls,
    # we need tool results — but that's handled by the engine loop
    return result
