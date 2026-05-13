"""
WeKnora Faithful Port — Context Compression

Ported from WeKnora internal/agent/token/compress.go

Hard truncation of older history messages when token count exceeds
the context threshold (80%). Preserves system prompt, current turn,
and never splits tool_call/tool_result pairs.
"""

from typing import Any, Dict, List

from core.agent.weknora_port.memory.estimator import Estimator
from core.agent.weknora_port.const import DEFAULT_CONTEXT_THRESHOLD_RATIO


def compress_context(
    messages: List[Dict[str, Any]],
    estimator: Estimator,
    max_tokens: int,
    current_tokens: int,
    threshold_ratio: float = DEFAULT_CONTEXT_THRESHOLD_RATIO,
) -> List[Dict[str, Any]]:
    """
    Trim older history messages to bring total token count below the threshold.

    Preserves:
    - The system prompt (first message)
    - The current turn: user query (last user message) + subsequent assistant/tool messages
    - tool_call / tool_result message pairs (never splits them)

    Args:
        messages: Full message array
        estimator: Token estimator
        max_tokens: Context window size
        current_tokens: Current estimated token count
        threshold_ratio: Ratio of context window that triggers compression (default 0.8)

    Returns:
        Compressed message array
    """
    if max_tokens <= 0 or len(messages) <= 2:
        return messages

    threshold = int(max_tokens * threshold_ratio)
    if current_tokens <= threshold:
        return messages

    system_msg = messages[0]

    # Find the current user query — the last message with role "user"
    last_user_idx = len(messages) - 1
    for i in range(len(messages) - 1, 0, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    history = messages[1:last_user_idx]
    tail = messages[last_user_idx:]

    if not history:
        return messages

    groups = _group_tool_messages(history)

    tokens_to_free = current_tokens - threshold
    freed = 0
    remove_up_to = 0

    for i, group in enumerate(groups):
        group_tokens = sum(estimator.estimate_message(msg) for msg in group)
        freed += group_tokens
        remove_up_to = i + 1
        if freed >= tokens_to_free:
            break

    remaining = [system_msg]
    for i in range(remove_up_to, len(groups)):
        remaining.extend(groups[i])
    remaining.extend(tail)

    return remaining


def _group_tool_messages(
    messages: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    Group middle messages into logical units:
    - An assistant message with tool_calls + its corresponding tool result messages = one group
    - A standalone message (user, assistant without tool_calls) = one group

    This ensures tool_call/tool_result pairs are never split during compression.
    """
    groups: List[List[Dict[str, Any]]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        # If this is an assistant message with tool_calls, group it with following tool results
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            group = [msg]
            i += 1
            # Collect all following tool result messages
            while i < len(messages) and messages[i].get("role") == "tool":
                group.append(messages[i])
                i += 1
            groups.append(group)
        else:
            groups.append([msg])
            i += 1

    return groups
