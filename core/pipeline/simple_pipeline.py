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

        # BM25
        t0 = time.time()
        bm25_chunks = self.retrieval._execute_bm25(query, options, candidate_k)
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

        # 4. Rerank
        if use_rerank and chunks:
            t0 = time.time()
            chunks = self.retrieval._rerank(query, chunks, options)
            timing["rerank"] = time.time() - t0

        # 5. 截断
        k = rerank_top_k or top_k
        chunks = chunks[:k]

        # 6. Parent Expand
        t0 = time.time()
        chunks = self._expand_to_parent_chunks(chunks)
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

    def _expand_to_parent_chunks(
        self,
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        将 child/summary chunks 做智能展开。

        逻辑（复用 RAGPipeline._expand_to_parent_chunks）：
        - summary chunk → 展开为 parent chunk
        - 同一个 parent 有 >= 2 个 child chunk → 展开为 parent chunk
        - 同一个 parent 只有 1 个 child chunk → 保留该 child + 前后各 1 个 sibling
        - 无 parent_id 的 chunk → 保持原样
        """
        if not chunks:
            return chunks

        # 1. 按 parent_id 分组
        parent_children: Dict[str, List[RetrievedChunk]] = {}
        for chunk in chunks:
            if chunk.parent_id:
                parent_children.setdefault(chunk.parent_id, []).append(chunk)

        if not parent_children:
            return chunks

        # 2. 收集需要展开的 parent_id
        parent_ids_to_expand = []
        single_child_ids = []
        for pid, children in parent_children.items():
            if any(c.chunk_type == "summary" for c in children):
                parent_ids_to_expand.append(pid)
            elif len(children) >= 2:
                parent_ids_to_expand.append(pid)
            else:
                single_child_ids.append(pid)

        # 3. 批量拉取需要展开的 parent chunks
        parent_map: Dict[str, Dict] = {}
        if parent_ids_to_expand:
            try:
                resp = self._store.es.mget(
                    index=settings.es_index_chunks,
                    body={"ids": parent_ids_to_expand},
                )
                for doc in resp.get("docs", []):
                    if doc.get("found") and doc.get("_source"):
                        parent_map[doc["_id"]] = doc["_source"]
            except Exception as e:
                logger.warning(f"[SimplePipeline] 批量获取 parent chunk 失败: {e}")

        # 4. 构建结果
        result: List[RetrievedChunk] = []
        seen_ids = set()

        for chunk in chunks:
            if chunk.chunk_id in seen_ids:
                continue

            if not chunk.parent_id:
                result.append(chunk)
                seen_ids.add(chunk.chunk_id)
                continue

            children = parent_children[chunk.parent_id]

            # 4a. summary → 展开为 parent
            if chunk.chunk_type == "summary":
                if chunk.parent_id in parent_map:
                    parent_src = parent_map[chunk.parent_id]
                    parent_chunk = RetrievedChunk(
                        chunk_id=parent_src.get("chunk_id", chunk.parent_id),
                        doc_id=parent_src.get("doc_id", ""),
                        content=parent_src.get("content", ""),
                        score=chunk.score,
                        doc_title=parent_src.get("doc_title"),
                        dataset_id=parent_src.get("dataset_id"),
                        chunk_type="parent",
                        doc_hash=parent_src.get("doc_hash"),
                        section_title=parent_src.get("section_title"),
                        parent_id=None,
                    )
                    result.append(parent_chunk)
                    seen_ids.add(chunk.parent_id)
                else:
                    result.append(chunk)
                    seen_ids.add(chunk.chunk_id)
                continue

            # 4b. >= 2 个 child from same parent → 展开为 parent
            if len(children) >= 2:
                if chunk.parent_id in parent_map:
                    parent_src = parent_map[chunk.parent_id]
                    max_score = max(c.score for c in children)
                    parent_chunk = RetrievedChunk(
                        chunk_id=parent_src.get("chunk_id", chunk.parent_id),
                        doc_id=parent_src.get("doc_id", ""),
                        content=parent_src.get("content", ""),
                        score=max_score,
                        doc_title=parent_src.get("doc_title"),
                        dataset_id=parent_src.get("dataset_id"),
                        chunk_type="parent",
                        doc_hash=parent_src.get("doc_hash"),
                        section_title=parent_src.get("section_title"),
                        parent_id=None,
                    )
                    if chunk.parent_id not in seen_ids:
                        result.append(parent_chunk)
                        seen_ids.add(chunk.parent_id)
                continue

            # 4c. 只有 1 个 child → 保留 child + 前后各 1 个 sibling
            result.append(chunk)
            seen_ids.add(chunk.chunk_id)

            siblings = self._get_sibling_chunks(chunk.parent_id, chunk.chunk_id)
            for sib in siblings:
                if sib.chunk_id not in seen_ids:
                    result.append(sib)
                    seen_ids.add(sib.chunk_id)

        return result

    def _get_sibling_chunks(
        self,
        parent_id: str,
        current_chunk_id: str,
        limit: int = 1,
    ) -> List[RetrievedChunk]:
        """获取指定 child chunk 的前后各 N 个 sibling"""
        try:
            resp = self._store.es.search(
                index=settings.es_index_chunks,
                body={
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"parent_id": parent_id}},
                                {"term": {"chunk_type": "child"}},
                                {"term": {"is_latest": True}},
                            ],
                            "must_not": [{"term": {"chunk_id": current_chunk_id}}],
                        }
                    },
                    "sort": [{"created_at": "asc"}],
                    "size": 100,
                },
            )
        except Exception as e:
            logger.warning(f"[SimplePipeline] 获取 sibling chunks 失败: {e}")
            return []

        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            return []

        current_idx = -1
        for i, hit in enumerate(hits):
            if hit.get("_source", {}).get("chunk_id") == current_chunk_id:
                current_idx = i
                break

        if current_idx < 0:
            return []

        siblings: List[RetrievedChunk] = []
        for i in range(current_idx - limit, current_idx):
            if i >= 0:
                src = hits[i].get("_source", {})
                siblings.append(
                    self._make_retrieved_chunk(src, hits[i].get("_score", 0.0))
                )

        for i in range(current_idx + 1, current_idx + 1 + limit):
            if i < len(hits):
                src = hits[i].get("_source", {})
                siblings.append(
                    self._make_retrieved_chunk(src, hits[i].get("_score", 0.0))
                )

        return siblings

    def _make_retrieved_chunk(self, source: Dict, score: float = 0.0) -> RetrievedChunk:
        """从 ES _source 构建 RetrievedChunk"""
        return RetrievedChunk(
            chunk_id=source.get("chunk_id", ""),
            doc_id=source.get("doc_id", ""),
            content=source.get("content", ""),
            score=score,
            doc_title=source.get("doc_title"),
            dataset_id=source.get("dataset_id"),
            chunk_type=source.get("chunk_type"),
            doc_hash=source.get("doc_hash"),
            section_title=source.get("section_title"),
            parent_id=source.get("parent_id"),
        )
