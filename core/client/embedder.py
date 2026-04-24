"""
Embedding 客户端 — 从 services/embedding_client.py 精简

通过 HTTP API 调用独立的 Embedding 服务。
"""

from typing import List, Optional

import httpx
import numpy as np
from loguru import logger

from config import settings


class EmbeddingClient:
    """Embedding 服务客户端"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or f"http://localhost:{settings.embedding_port}").rstrip("/")
        self.timeout = 300.0

    def encode(self, text: str) -> Optional[np.ndarray]:
        """单文本向量化"""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/encode",
                    json={"text": text},
                )
                response.raise_for_status()
                data = response.json()
                return np.array(data["embedding"], dtype=np.float32)
        except Exception as e:
            logger.error(f"向量化失败: {e}")
            return None

    def encode_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """批量向量化，失败后降级为逐条调用"""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/encode_batch",
                    json={"texts": texts, "max_batch_size": 32},
                )
                if response.status_code >= 400:
                    # 降级：逐条调用，附带调试信息
                    logger.warning(
                        f"批量接口返回 {response.status_code}，降级为逐条调用。 "
                        f"内容长度: {[len(t) for t in texts[:5]]}..."
                    )
                    return [self.encode(t) if t else None for t in texts]
                response.raise_for_status()
                data = response.json()
                return [
                    np.array(emb, dtype=np.float32) if emb else None
                    for emb in data["embeddings"]
                ]
        except Exception as e:
            logger.warning(f"批量向量化失败，降级为逐条调用: {e}")
            return [self.encode(t) if t else None for t in texts]


# ── 全局实例 ──────────────────────────────────────────

_client: Optional[EmbeddingClient] = None


def get_embedder() -> EmbeddingClient:
    global _client
    if _client is None:
        _client = EmbeddingClient()
    return _client


def encode(text: str) -> Optional[np.ndarray]:
    """快捷函数：单文本向量化"""
    return get_embedder().encode(text)


def encode_batch(texts: List[str]) -> List[Optional[np.ndarray]]:
    """快捷函数：批量向量化"""
    return get_embedder().encode_batch(texts)
