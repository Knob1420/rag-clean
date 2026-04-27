"""
混合检索服务 — BM25 + 向量检索 + RRF 融合

适配 rag-clean 实际 ES 字段：
- chunk_id, doc_id, doc_hash, doc_title, dataset_id, chunk_type
- content, parent_id, embedding_vector, is_latest, created_at
"""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config import settings
from core.client.embedder import encode
from core.client.rerank_client import rerank_documents
from core.retrieve.retrieval_models import (
    RetrievedChunk,
    RetrievalOptions,
    RetrievalResult,
)
from store import DocumentStore, get_store


# ============================================================
# RRF 融合
# ============================================================


class RRFFusion:
    """Reciprocal Rank Fusion 融合算法"""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        bm25_results: List[Tuple[RetrievedChunk, int]],
        vector_results: List[Tuple[RetrievedChunk, int]],
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> List[Tuple[RetrievedChunk, float]]:
        """融合两个检索结果，按融合得分降序排列"""
        scores: Dict[str, float] = defaultdict(float)
        chunks: Dict[str, RetrievedChunk] = {}

        for chunk, rank in bm25_results:
            scores[chunk.chunk_id] += bm25_weight * (1.0 / (self.k + rank))
            chunks[chunk.chunk_id] = chunk

        for chunk, rank in vector_results:
            if chunk.chunk_id not in chunks:
                chunks[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] += vector_weight * (1.0 / (self.k + rank))

        ranked = [(chunks[cid], score) for cid, score in scores.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


# ============================================================
# 检索服务
# ============================================================


class RetrievalService:
    """混合检索服务类"""

    def __init__(self, store: Optional[DocumentStore] = None):
        self._store = store or get_store()
        self.chunks_index = settings.es_index_chunks
        self.default_top_k = settings.default_top_k
        self.rrf = RRFFusion(k=60)

    @property
    def es(self):
        return self._store.es

    # ============================================================
    # 主入口
    # ============================================================

    def search(
        self,
        query: str,
        options: Optional[RetrievalOptions] = None,
        use_hybrid: bool = True,
    ) -> RetrievalResult:
        """执行混合检索（BM25 + 向量检索 + RRF 融合）"""
        timing = {}

        if options is None:
            options = RetrievalOptions()

        # 检索
        if use_hybrid:
            t0 = time.time()
            chunks = self._hybrid_search(query, options)
            timing["retrieve"] = time.time() - t0
        else:
            t0 = time.time()
            chunks = self._bm25_search(query, options)
            timing["retrieve"] = time.time() - t0

        # 父子块召回

        # Rerank
        if options.use_rerank and chunks:
            t0 = time.time()
            chunks = self._rerank(query, chunks, options)
            timing["rerank"] = time.time() - t0

        # 过滤低分
        if options.min_score > 0:
            chunks = [c for c in chunks if c.score >= options.min_score]

        # # 截断
        # chunks = chunks[: options.top_k]

        timing_str = ", ".join([f"{k}={v:.0f}s" for k, v in timing.items()])
        logger.info(
            f"检索完成: query='{query}', "
            f"results={len(chunks)}, "
            f"hybrid={use_hybrid}, "
            f"rerank={options.use_rerank}, "
            f"({timing_str})"
        )

        return RetrievalResult(
            query=query,
            total=len(chunks),
            chunks=chunks,
            timing=timing,
        )

    # ============================================================
    # 混合检索
    # ============================================================

    def _hybrid_search(
        self,
        query: str,
        options: RetrievalOptions,
    ) -> List[RetrievedChunk]:
        """混合检索（BM25 + 向量 + RRF 融合）"""
        query_vector = encode(query)
        candidate_k = options.top_k

        # 1. BM25
        bm25_results = self._execute_bm25(query, options, candidate_k)

        # 2. 向量
        vector_results = []
        if query_vector is not None:
            vector_results = self._execute_vector_search(
                query_vector, options, candidate_k
            )

        # 3. RRF 融合
        rrf_results = self.rrf.fuse(
            bm25_results=[(c, i) for i, c in enumerate(bm25_results)],
            vector_results=[(c, i) for i, c in enumerate(vector_results)],
        )

        final_chunks = []
        for chunk, fused_score in rrf_results:
            chunk.score = fused_score
            final_chunks.append(chunk)

        return final_chunks

    def _bm25_search(
        self,
        query: str,
        options: RetrievalOptions,
    ) -> List[RetrievedChunk]:
        """仅 BM25 检索"""
        return self._execute_bm25(query, options, options.top_k)

    def _vector_search(
        self,
        query: str,
        options: RetrievalOptions,
    ) -> List[RetrievedChunk]:
        """仅向量检索"""
        query_vector = encode(query)
        if query_vector is None:
            return []
        return self._execute_vector_search(query_vector, options, options.top_k)

    # ============================================================
    # ES 查询执行
    # ============================================================

    def _execute_bm25(
        self,
        query: str,
        options: RetrievalOptions,
        top_k: int,
    ) -> List[RetrievedChunk]:
        """执行 BM25 检索"""
        bool_query = self._build_bm25_query(query, options)

        dsl: Dict[str, Any] = {
            "query": bool_query,
            "size": top_k,
        }

        try:
            response = self.es.search(index=self.chunks_index, body=dsl)
        except Exception as e:
            logger.error(f"BM25 检索失败: {e}")
            return []

        return self._parse_results_with_score(response)

    def _execute_vector_search(
        self,
        query_vector: np.ndarray,
        options: RetrievalOptions,
        top_k: int,
    ) -> List[RetrievedChunk]:
        """执行向量检索"""
        vector_query = self._build_vector_query(query_vector, options, top_k)

        dsl = {**vector_query, "size": top_k}

        try:
            response = self.es.search(index=self.chunks_index, body=dsl)
        except Exception as e:
            if "different number of dimensions" in str(e):
                logger.warning(f"向量维度不匹配，向量检索降级: {e}")
            else:
                logger.error(f"向量检索失败: {e}")
            return []

        return self._parse_results_with_score(response)

    # ============================================================
    # BM25 查询构建
    # ============================================================

    def _build_bm25_query(
        self,
        query: str,
        options: RetrievalOptions,
    ) -> Dict[str, Any]:
        """
        构建 BM25 查询

        filter: is_latest=True, chunk_type=child, doc_ids, dataset_ids
        """
        filter_conditions = [
            {"term": {"is_latest": True}},
            {"terms": {"chunk_type": ["child", "summary"]}},
        ]
        if options.doc_ids:
            filter_conditions.append({"terms": {"doc_id": options.doc_ids}})
        if options.dataset_ids:
            filter_conditions.append({"terms": {"dataset_id": options.dataset_ids}})

        return {
            "bool": {
                "filter": filter_conditions,
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["doc_title^3", "content"],
                            "type": "best_fields",
                        }
                    }
                ],
                "minimum_should_match": 1,
            }
        }

    # ============================================================
    # 向量查询构建
    # ============================================================

    def _build_vector_query(
        self,
        query_vector: np.ndarray,
        options: RetrievalOptions,
        top_k: int,
    ) -> Dict[str, Any]:
        """
        构建向量 kNN 查询

        filter: is_latest=True, chunk_type=child, doc_ids, dataset_ids
        """
        filter_conditions = [
            {"term": {"is_latest": True}},
            {"terms": {"chunk_type": ["child", "summary"]}},
        ]
        if options.doc_ids:
            filter_conditions.append({"terms": {"doc_id": options.doc_ids}})
        if options.dataset_ids:
            filter_conditions.append({"terms": {"dataset_id": options.dataset_ids}})

        num_candidates = max(top_k * 4, 50)

        return {
            "knn": {
                "field": "embedding_vector",
                "query_vector": query_vector.tolist(),
                "k": top_k,
                "num_candidates": num_candidates,
                "filter": {"bool": {"filter": filter_conditions}},
            }
        }

    # ============================================================
    # 结果解析
    # ============================================================

    def _parse_results_with_score(
        self, response: Dict[str, Any]
    ) -> List[RetrievedChunk]:
        """解析 ES 结果并保留原始得分"""
        hits = response.get("hits", {}).get("hits", [])

        chunks = []
        for hit in hits:
            source = hit.get("_source", {})

            chunk = RetrievedChunk(
                chunk_id=source.get("chunk_id", ""),
                doc_id=source.get("doc_id", ""),
                content=source.get("content", ""),
                score=hit.get("_score", 0.0),
                # 文档级字段
                doc_title=source.get("doc_title"),
                dataset_id=source.get("dataset_id"),
                # chunk 级字段
                chunk_type=source.get("chunk_type"),
                doc_hash=source.get("doc_hash"),
                # 父子导航
                parent_id=source.get("parent_id"),
            )
            chunks.append(chunk)

        return chunks

    # ============================================================
    # Rerank
    # ============================================================

    def _rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        options: RetrievalOptions,
    ) -> List[RetrievedChunk]:
        """使用 Rerank 对检索结果重排序"""
        logger.info(f"Rerank 重排序: {len(chunks)} 个候选结果")

        documents = []
        doc_to_chunk_idx: Dict[str, int] = {}
        for i, chunk in enumerate(chunks):
            text = chunk.content
            documents.append(text)
            doc_to_chunk_idx[text] = i

        rerank_results = rerank_documents(
            query=query,
            documents=documents,
            top_k=options.rerank_top_k,
        )

        reranked_chunks = []
        seen_indices = set()
        for doc_text, score in rerank_results:
            idx = doc_to_chunk_idx.get(doc_text)
            if idx is not None and idx not in seen_indices:
                chunk = chunks[idx]
                chunk.score = score
                reranked_chunks.append(chunk)
                seen_indices.add(idx)

        logger.info(f"  Rerank 完成: 返回 {len(reranked_chunks)} 个结果")
        return reranked_chunks


# ── 全局实例 ──────────────────────────────────────────

_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """获取检索服务单例"""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
