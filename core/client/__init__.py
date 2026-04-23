"""
客户端层 — Embedding、Rerank
"""

from core.client.embedder import EmbeddingClient, get_embedder, encode, encode_batch
from core.client.rerank_client import RerankClient, get_rerank_client, rerank_documents

__all__ = [
    "EmbeddingClient",
    "get_embedder",
    "encode",
    "encode_batch",
    "RerankClient",
    "get_rerank_client",
    "rerank_documents",
]
