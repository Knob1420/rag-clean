"""
LLM 生成层 — LLM 客户端、RAG 回答生成
"""

from core.generation.llm import LLMClient, get_llm_client, parse_json_response
from core.generation.generation import GenerationService, get_generation_service

__all__ = [
    "LLMClient",
    "get_llm_client",
    "parse_json_response",
    "GenerationService",
    "get_generation_service",
]
