"""
客户端层 — Embedding、Rerank、MinerU
"""

from core.client.embedder import EmbeddingClient, get_embedder, encode, encode_batch
from core.client.rerank_client import RerankClient, get_rerank_client, rerank_documents
from core.client.mineru_client import (
    is_mineru_service_running,
    start_mineru_service,
    stop_mineru_service,
    convert_with_mineru,
)

__all__ = [
    # Embedding
    "EmbeddingClient",
    "get_embedder",
    "encode",
    "encode_batch",
    # Rerank
    "RerankClient",
    "get_rerank_client",
    "rerank_documents",
    # MinerU
    "is_mineru_service_running",
    "start_mineru_service",
    "stop_mineru_service",
    "convert_with_mineru",
]
