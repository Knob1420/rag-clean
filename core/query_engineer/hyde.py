"""
HyDE (Hypothetical Document Embeddings) 查询引擎

原理：
1. 用 LLM 根据用户查询生成「假设性回答文档」
2. 对假设性文档做 embedding
3. 用该 embedding 去向量库检索，比直接用 query embedding 更接近真实文档的语义空间

适用场景：
- 用户查询过短、口语化，与文档语体差距大
- 交叉语言检索（中文 query → 英文文档）
- 需要提升语义检索召回率的场景

优化：
- 多假设性文档并发生成（asyncio + semaphore）
- embedding 批量调用
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from core.client.embedder import EmbeddingClient, get_embedder
from prompt import HYDE_SYSTEM_PROMPT, HYDE_USER_TEMPLATE


# ── 数据结构 ──────────────────────────────────────────


@dataclass
class HyDEResult:
    """HyDE 查询结果"""

    original_query: str
    hypothetical_docs: List[str] = field(default_factory=list)
    hypothetical_embeddings: List[Optional[list]] = field(default_factory=list)
    fused_embedding: Optional[list] = None  # 融合后的单一向量


# ── 核心引擎 ──────────────────────────────────────────


class HyDEQueryEngine:
    """HyDE 查询引擎"""

    def __init__(
        self,
        llm_client=None,
        embedder: Optional[EmbeddingClient] = None,
        num_hypotheses: int = 1,
        temperature: float = 0.7,
    ):
        """
        Args:
            llm_client: LLM 客户端（默认使用全局实例）
            embedder: Embedding 客户端（默认使用全局实例）
            num_hypotheses: 生成假设性文档的数量（1-3，越多召回越广但越慢）
            temperature: 生成温度（越高假设性文档越多样）
        """
        if llm_client is None:
            from core.generation.llm import get_llm_client
            llm_client = get_llm_client()
        self.llm = llm_client
        self.embedder = embedder or get_embedder()
        self.num_hypotheses = max(1, min(num_hypotheses, 3))
        self.temperature = temperature

    def generate_hypothetical_docs(self, query: str) -> List[str]:
        """
        根据用户查询生成假设性文档。
        多篇假设性文档并发生成以减少延迟。

        Args:
            query: 用户原始查询

        Returns:
            假设性文档列表
        """
        if self.num_hypotheses == 1:
            # 单篇直接同步调用，避免 asyncio 开销
            return self._generate_single(query, 0)

        # 多篇并发生成
        return self._generate_concurrent(query)

    def _generate_single(self, query: str, index: int) -> List[str]:
        """同步生成单篇假设性文档"""
        prompt = HYDE_USER_TEMPLATE.format(query=query)
        try:
            doc = self.llm.call(
                messages=[
                    {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=128,
            )
            doc = doc.strip()
            if doc:
                logger.debug(f"HyDE 生成假设性文档 #{index + 1}: {doc[:80]}...")
                return [doc]
        except Exception as e:
            logger.warning(f"HyDE 生成假设性文档 #{index + 1} 失败: {e}")

        # 失败回退到原始查询
        logger.warning("HyDE 假设性文档生成失败，回退到原始查询")
        return [query]

    def _generate_concurrent(self, query: str) -> List[str]:
        """并发生成多篇假设性文档（asyncio + httpx）"""
        async def _run():
            semaphore = asyncio.Semaphore(self.num_hypotheses)

            async def _call_one(idx: int) -> tuple[int, Optional[str]]:
                async with semaphore:
                    prompt = HYDE_USER_TEMPLATE.format(query=query)
                    messages = [
                        {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                    try:
                        response = await self.llm._call_async(messages, max_tokens=128)
                        doc = response.strip()
                        if doc:
                            logger.debug(f"HyDE 生成假设性文档 #{idx + 1}: {doc[:80]}...")
                            return idx, doc
                    except Exception as e:
                        logger.warning(f"HyDE 生成假设性文档 #{idx + 1} 失败: {e}")
                    return idx, None

            tasks = [_call_one(i) for i in range(self.num_hypotheses)]
            results = await asyncio.gather(*tasks)
            # 按原始顺序排列
            docs = []
            for _, doc in sorted(results, key=lambda x: x[0]):
                if doc is not None:
                    docs.append(doc)

            if not docs:
                logger.warning("HyDE 所有假设性文档均生成失败，回退到原始查询")
                docs = [query]
            return docs

        return asyncio.run(_run())

    def embed_docs(self, docs: List[str]) -> List[Optional[list]]:
        """
        对假设性文档做 embedding（批量）。

        Args:
            docs: 假设性文档列表

        Returns:
            embedding 列表（numpy array → list）
        """
        embeddings = self.embedder.encode_batch(docs)
        return [
            emb.tolist() if emb is not None else None for emb in embeddings
        ]

    @staticmethod
    def fuse_embeddings(embeddings: List[list]) -> list:
        """
        将多个 embedding 融合为单一向量（取平均）。

        Args:
            embeddings: 有效的 embedding 列表

        Returns:
            融合后的 embedding（list）
        """
        if not embeddings:
            return []
        if len(embeddings) == 1:
            return embeddings[0]

        dim = len(embeddings[0])
        fused = [0.0] * dim
        for emb in embeddings:
            for j in range(dim):
                fused[j] += emb[j]
        count = len(embeddings)
        fused = [v / count for v in fused]
        return fused

    def transform(self, query: str) -> HyDEResult:
        """
        完整的 HyDE 变换流程：query → 假设性文档 → embedding → 融合。

        Args:
            query: 用户原始查询

        Returns:
            HyDEResult 包含假设性文档、各文档 embedding 和融合 embedding
        """
        logger.info(f"HyDE 变换: query='{query}', num_hypotheses={self.num_hypotheses}")

        # 1. 生成假设性文档（并发）
        docs = self.generate_hypothetical_docs(query)

        # 2. 批量向量化
        raw_embeddings = self.embed_docs(docs)

        # 3. 融合
        valid_embeddings = [e for e in raw_embeddings if e is not None]
        fused = self.fuse_embeddings(valid_embeddings) if valid_embeddings else None

        result = HyDEResult(
            original_query=query,
            hypothetical_docs=docs,
            hypothetical_embeddings=raw_embeddings,
            fused_embedding=fused,
        )

        logger.info(
            f"HyDE 变换完成: {len(docs)} 篇假设性文档, "
            f"fused_embedding={'ok' if fused else 'failed'}"
        )
        return result

    def get_query_embedding(self, query: str) -> Optional[list]:
        """
        快捷方法：返回融合后的 embedding，直接用于向量检索。

        Args:
            query: 用户原始查询

        Returns:
            融合后的 embedding list，失败返回 None
        """
        result = self.transform(query)
        return result.fused_embedding
