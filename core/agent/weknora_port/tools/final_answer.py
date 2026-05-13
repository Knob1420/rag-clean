"""
WeKnora Faithful Port — final_answer Tool

Ported from WeKnora internal/agent/tools/final_answer.go

The terminal tool that submits the agent's final answer.
Uses WeKnora's 3-tier JSON parsing for robust argument extraction:
  Tier 1: Strict JSON parse
  Tier 2: Repair + parse (fix truncation, trailing commas, single quotes)
  Tier 3: Regex extraction of outermost JSON object

Includes finalAnswerParseFallback for deeply malformed outputs.
"""

import json
import re
from typing import Any, Dict, Optional

from loguru import logger


def final_answer_handler(args: Dict[str, Any], content: str = "") -> str:
    """
    Execute final_answer tool: extract answer and return it.

    The answer comes from:
    1. Parsed args['answer'] (if valid)
    2. Fallback to the raw content of the assistant message

    Args:
        args: Parsed tool arguments
        content: Raw assistant message content (fallback)

    Returns:
        The final answer text
    """
    answer = args.get("answer", "")
    if not answer:
        answer = content
    return answer or ""


def parse_final_answer_args(raw_arguments: str) -> Dict[str, Any]:
    """
    Parse final_answer tool call arguments using WeKnora's 3-tier strategy.

    Tier 1: Strict json.loads
    Tier 2: _repair_json + json.loads
    Tier 3: Regex extract outermost JSON object

    Falls back to {"answer": raw_arguments} if all tiers fail.

    Args:
        raw_arguments: The function.arguments string from tool_calls

    Returns:
        Parsed arguments dict with at least 'answer' key
    """
    if not raw_arguments:
        return {"answer": ""}

    text = raw_arguments.strip()

    # Tier 1: Strict parse
    try:
        result = json.loads(text)
        if isinstance(result, dict) and "answer" in result:
            return result
    except json.JSONDecodeError:
        pass

    # Tier 2: Repair + parse
    try:
        repaired = _repair_json(text)
        result = json.loads(repaired)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Tier 3: Regex extract outermost JSON object
    try:
        match = re.search(
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL
        )
        if match:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
    except json.JSONDecodeError:
        pass

    # Tier 3b: Try ```json code block
    try:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
            if isinstance(result, dict):
                return result
    except json.JSONDecodeError:
        pass

    # Final fallback: treat entire text as answer
    logger.warning(
        "[final_answer] All JSON parsing tiers failed, "
        "using raw text as answer"
    )
    return final_answer_parse_fallback(text)


def _repair_json(text: str) -> str:
    """
    Repair common LLM JSON output issues.

    - Fix truncated JSON: close missing brackets
    - Fix trailing commas before ] or }
    - Fix single quotes → double quotes (outside of string values)
    """
    # Fix truncated JSON: close missing brackets
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    if open_braces > 0:
        text += "}" * open_braces
    if open_brackets > 0:
        text += "]" * open_brackets

    # Fix trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Fix single quotes → double quotes (only outside double-quoted strings)
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


def final_answer_parse_fallback(text: str) -> Dict[str, Any]:
    """
    Last-resort fallback for deeply malformed final_answer arguments.

    Attempts to extract answer content from various formats:
    - "answer" key in partially parsed JSON
    - Plain text after "answer": prefix
    - The entire text as-is
    """
    # Try to find "answer" key value with loose matching
    patterns = [
        r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"\s*',
        r"'answer'\s*:\s*'((?:[^'\\]|\\.)*)'\s*",
        r"answer\s*[:=]\s*(.+?)(?:\s*[,}]\s*|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            answer = match.group(1)
            # Unescape basic JSON string escapes
            answer = answer.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
            return {"answer": answer}

    # If nothing found, return the whole text as answer
    return {"answer": text}


# ── Tool Definition for Registry ──────────────────────────────────

FINAL_ANSWER_DEFINITION = {
    "name": "final_answer",
    "description": (
        "Submit your final answer. This is your TERMINAL action — "
        "you MUST call this when you have gathered enough information "
        "to respond to the user. NEVER end without calling this tool. "
        "The 'answer' field contains your complete response (Markdown supported)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Your complete final answer (Markdown format)",
            },
        },
        "required": ["answer"],
    },
}
