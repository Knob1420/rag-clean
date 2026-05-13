"""
WeKnora Faithful Port — Package

A faithful 1:1 Python port of WeKnora's Progressive RAG Agent for comparison.
"""

from core.agent.weknora_port.engine import AgentEngine, AgentResult, StreamEvent, AgentStep
from core.agent.weknora_port.prompts import (
    KnowledgeBaseInfo,
    SelectedDocumentInfo,
    RecentDocInfo,
    build_system_prompt_with_options,
    build_runtime_context_block,
    build_messages_with_llm_context,
)
from core.agent.weknora_port.const import (
    DEFAULT_AGENT_MAX_ITERATIONS,
    DEFAULT_MAX_TOOL_OUTPUT,
)

__all__ = [
    "AgentEngine",
    "AgentResult",
    "StreamEvent",
    "AgentStep",
    "KnowledgeBaseInfo",
    "SelectedDocumentInfo",
    "RecentDocInfo",
    "build_system_prompt_with_options",
    "build_runtime_context_block",
    "build_messages_with_llm_context",
    "DEFAULT_AGENT_MAX_ITERATIONS",
    "DEFAULT_MAX_TOOL_OUTPUT",
]
