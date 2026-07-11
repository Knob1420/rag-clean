"""
LLM Prompt 模板（按功能拆分）

- generation.py  → RAG 回答生成
- hyde.py        → HyDE 假设性文档生成 & Summary 生成
- agent.py       → ReAct Agent 系统 Prompt
"""

from prompt.agent import REACT_SYSTEM_PROMPT
from prompt.generation import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT_TEMPLATE,
)
from prompt.hyde import (
    HYDE_SYSTEM_PROMPT,
    HYDE_USER_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    build_summary_prompt,
)

__all__ = [
    # Generation
    "RAG_SYSTEM_PROMPT",
    "RAG_USER_PROMPT_TEMPLATE",
    # HyDE & Summary
    "HYDE_SYSTEM_PROMPT",
    "HYDE_USER_TEMPLATE",
    "SUMMARY_SYSTEM_PROMPT",
    "build_summary_prompt",
    # Agent
    "REACT_SYSTEM_PROMPT",
]
