"""
检索层 — Embedding、Rerank、混合检索
"""

from core.client.embedder import EmbeddingClient, get_embedder, encode, encode_batch
from core.client.rerank_client import RerankClient, get_rerank_client, rerank_documents
from core.retrieve.retrieval import RetrievalService, get_retrieval_service
from core.retrieve.retrieval_models import (
    RetrievedChunk,
    RetrievalResult,
    RetrievalOptions,
    HighlightOptions,
    TokenUsage,
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    HealthResponse,
    SourceInfo,
)

__all__ = [
    "EmbeddingClient",
    "get_embedder",
    "encode",
    "encode_batch",
    "RerankClient",
    "get_rerank_client",
    "rerank_documents",
    "RetrievalService",
    "get_retrieval_service",
    "RetrievedChunk",
    "RetrievalResult",
    "RetrievalOptions",
    "HighlightOptions",
    "TokenUsage",
    "ChatRequest",
    "ChatResponse",
    "SearchRequest",
    "SearchResponse",
    "HealthResponse",
    "SourceInfo",
]
