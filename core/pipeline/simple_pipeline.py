"""
Simple Pipeline — BM25 + Raw Query Vector + RRF + HyDE 开关 + Keyword Spec

替换旧 RAGPipeline 的简化版：
- 不使用 QueryUnderstandingService / QueryRewriteService，直接用原始 query
- BM25 检索仍用关键词提取+同义词扩展
- Vector 检索用 encode(query) 的原始 embedding（或 HyDE embedding）
- Spec 查询用 build_specs_context(query) 关键词匹配
- 生成统一用 RAG_SYSTEM_PROMPT + RAG_USER_PROMPT_TEMPLATE

流程：Embedding → BM25 → Vector → RRF 融合 → 低质量过滤 → Rerank → Parent Expand → Spec 查询
"""

import time
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from config import settings
from core.client.embedder import encode
from core.generation.generation import GenerationService, get_generation_service
from core.ingestion.cleaner import is_low_quality_content
from core.pipeline.parent_expand import expand_to_parent_chunks
from core.products.specs_service import build_specs_context
from core.query_engineer.hyde import HyDEQueryEngine
from core.retrieve.retrieval import RetrievalService, get_retrieval_service
from core.retrieve.retrieval_models import RetrievedChunk, RetrievalOptions, TokenUsage
from store import get_store


class SimplePipelineResult:
    """SimplePipeline 执行结果"""

    def __init__(
        self,
        original_query: str,
        chunks: List[RetrievedChunk],
        spec_context: str,
        timing: dict,
        bm25_chunks: Optional[List[RetrievedChunk]] = None,
        vector_chunks: Optional[List[RetrievedChunk]] = None,
        generation_answer: Optional[str] = None,
        generation_usage: Optional[TokenUsage] = None,
    ):
        self.original_query = original_query
        self.chunks = chunks
        self.spec_context = spec_context
        self.timing = timing
        self.bm25_chunks = bm25_chunks or []
        self.vector_chunks = vector_chunks or []
        self.generation_answer = generation_answer
        self.generation_usage = generation_usage

    def __repr__(self):
        return (
            f"SimplePipelineResult(query={self.original_query[:30]!r}..., "
            f"chunks={len(self.chunks)}, "
            f"spec_context={len(self.spec_context)}chars)"
        )


class SimplePipeline:
    """BM25 + Raw Query Vector + RRF 融合 Pipeline"""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        store=None,
    ):
        self.retrieval = retrieval_service or get_retrieval_service()
        self._store = store or get_store()

    def run(
        self,
        query: str,
        top_k: int = 20,
        use_hyde: bool = False,
        use_rerank: bool = True,
        rerank_top_k: Optional[int] = None,
    ) -> SimplePipelineResult:
        """
        执行 Simple Pipeline

        流程：
        1. Embedding — encode(query)；若 use_hyde=True，用 HyDE 获取 fused_embedding
        2. BM25 检索
        3. Vector 检索
        4. RRF 融合
        5. 低质量过滤
        6. Rerank（可选）
        7. Parent Expand
        8. Spec 查询

        Args:
            query: 用户原始查询
            top_k: 检索返回数量
            use_hyde: 是否使用 HyDE embedding 替代 raw query embedding
            use_rerank: 是否使用 Rerank
            rerank_top_k: Rerank 后保留数量

        Returns:
            SimplePipelineResult
        """
        timing = {}

        # 1. Embedding
        t0 = time.time()
        if use_hyde:
            hyde_engine = HyDEQueryEngine(num_hypotheses=1)
            hyde_result = hyde_engine.transform(query)
            if hyde_result.fused_embedding:
                query_vector = np.array(hyde_result.fused_embedding, dtype=np.float32)
                logger.info(
                    f"HyDE 变换完成: {len(hyde_result.hypothetical_docs)} 篇假设性文档"
                )
            else:
                logger.warning("HyDE 融合 embedding 失败，回退到原始 query embedding")
                query_vector = encode(query)
        else:
            query_vector = encode(query)
        timing["embedding"] = time.time() - t0

        # 2. 检索
        options = RetrievalOptions(top_k=top_k, use_rerank=False)
        candidate_k = top_k * 2

        # BM25: 先构建 query_string
        t0 = time.time()
        query_string = self.retrieval._build_bm25_query(query, options)
        bm25_chunks = self.retrieval._execute_bm25(query_string, options, candidate_k)
        timing["bm25"] = time.time() - t0

        # Vector
        t0 = time.time()
        vector_chunks = []
        if query_vector is not None:
            vector_options = RetrievalOptions(top_k=candidate_k)
            vector_chunks = self.retrieval._execute_vector_search(
                query_vector, vector_options, candidate_k
            )
        timing["vector"] = time.time() - t0

        # 3. RRF 融合
        t0 = time.time()
        rrf_results = self.retrieval.rrf.fuse(
            bm25_results=[(c, i) for i, c in enumerate(bm25_chunks)],
            vector_results=[(c, i) for i, c in enumerate(vector_chunks)],
            bm25_weight=0.3,
            vector_weight=0.7,
        )
        chunks = []
        for chunk, score in rrf_results:
            chunk.score = score
            if not is_low_quality_content(chunk.content):
                chunks.append(chunk)
        timing["rrf"] = time.time() - t0

        # 4. Rerank（构建增强 rerank query）
        if use_rerank and chunks:
            t0 = time.time()
            from core.query_engineer.rerank_query import build_rerank_query
            rerank_query = build_rerank_query(query)
            chunks = self.retrieval._rerank(rerank_query, chunks, options)
            timing["rerank"] = time.time() - t0

        # 5. 截断
        k = rerank_top_k or top_k
        chunks = chunks[:k]

        # 6. Parent Expand
        t0 = time.time()
        chunks = expand_to_parent_chunks(chunks, self._store)
        timing["parent_expand"] = time.time() - t0

        # 7. Spec 查询
        t0 = time.time()
        spec_context = build_specs_context(query)
        timing["spec"] = time.time() - t0

        logger.info(
            f"[SimplePipeline] 完成: query='{query}', "
            f"bm25={len(bm25_chunks)}, vector={len(vector_chunks)}, "
            f"chunks={len(chunks)}, spec={len(spec_context)}chars"
        )

        return SimplePipelineResult(
            original_query=query,
            chunks=chunks,
            spec_context=spec_context,
            timing=timing,
            bm25_chunks=bm25_chunks,
            vector_chunks=vector_chunks,
        )
