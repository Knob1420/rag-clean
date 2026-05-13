"""
WeKnora Faithful Port - StripThinkBlocks

Ported from WeKnora internal/agent/strip_think.go

Strips think blocks (DeepSeek, Qwen, etc.) from LLM output
to prevent reasoning traces from leaking into tool call parsing or user-visible content.
"""

import re

# Matches standard think blocks: <think>...</think> (non-greedy, dotall)
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

# Matches the DeepSeek-specific streaming think format
_DEEPSEEK_THINK_PATTERN = re.compile(r"\u25c1think\u25de.*?\u25c1/think\u25de", re.DOTALL)


def strip_think_blocks(content: str) -> str:
    """
    Remove think blocks from LLM output.

    DeepSeek and Qwen models sometimes wrap their reasoning in think tags.
    This strips those blocks to get the actual content.

    Also handles the DeepSeek-specific streaming format.

    Args:
        content: Raw LLM output text

    Returns:
        Content with think blocks removed, whitespace trimmed
    """
    if not content:
        return content

    # Strip standard think blocks
    result = _THINK_BLOCK_PATTERN.sub("", content)

    # Strip DeepSeek streaming format
    result = _DEEPSEEK_THINK_PATTERN.sub("", result)

    # Clean up leading/trailing whitespace left after removal
    return result.strip()
