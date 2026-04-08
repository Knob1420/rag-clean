"""
Rerank 服务客户端

通过 HTTP API 调用独立的 Rerank 服务
"""

from typing import List, Optional, Tuple

import httpx
from loguru import logger

from config import settings


class RerankClient:
    """Rerank 服务客户端"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or f"http://localhost:{settings.rerank_port}").rstrip(
            "/"
        )
        self.timeout = 60.0

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """重排序文档"""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/rerank",
                    json={
                        "query": query,
                        "documents": documents,
                        "top_k": top_k,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return [(doc, score) for doc, score in data["results"]]
        except Exception as e:
            logger.error(f"Rerank 失败: {e}")
            return [(doc, 1.0 - i * 0.01) for i, doc in enumerate(documents)]


# ── 全局实例 ──────────────────────────────────────────

_rerank_client: Optional[RerankClient] = None


def get_rerank_client() -> RerankClient:
    """获取 Rerank 客户端单例"""
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = RerankClient()
    return _rerank_client


def rerank_documents(
    query: str,
    documents: List[str],
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """快捷函数：重排序文档"""
    return get_rerank_client().rerank(query=query, documents=documents, top_k=top_k)
