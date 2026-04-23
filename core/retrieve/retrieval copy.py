"""
混合检索服务 — BM25 + 向量检索 + RRF 融合

基于 rag-knowledge-base 的 retrieval.py，适配 rag-clean 扁平字段：
- get_es() → get_store().es
- encode_via_client() → encode()
- 移除所有 user_roles / access_roles 过滤
- _parse_results 适配 doc_type/domain/filter_terms 等扁平字段
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
    HighlightOptions,
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
        highlight: Optional[HighlightOptions] = None,
        use_hybrid: bool = True,
    ) -> RetrievalResult:
        """执行混合检索（BM25 + 向量检索 + RRF 融合）"""
        timing = {}

        if options is None:
            options = RetrievalOptions()
        if highlight is None:
            highlight = HighlightOptions()

        # 检索
        if use_hybrid:
            t0 = time.time()
            chunks = self._hybrid_search(query, options, highlight)
            timing["retrieve"] = time.time() - t0
        else:
            t0 = time.time()
            chunks = self._bm25_search(query, options, highlight)
            timing["retrieve"] = time.time() - t0

        # Rerank
        if options.use_rerank and chunks:
            t0 = time.time()
            chunks = self._rerank(query, chunks, options)
            timing["rerank"] = time.time() - t0

        # 过滤低分
        if options.min_score > 0:
            chunks = [c for c in chunks if c.score >= options.min_score]

        # 截断
        chunks = chunks[: options.top_k]

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
        highlight: HighlightOptions,
    ) -> List[RetrievedChunk]:
        """混合检索（BM25 + 向量 + RRF 融合）"""
        query_vector = encode(query)
        candidate_k = options.top_k * 2

        # 1. BM25
        bm25_results = self._execute_bm25(query, options, highlight, candidate_k)

        # 2. 向量
        vector_results = []
        if query_vector is not None:
            vector_options = options.model_copy(update={"top_k": candidate_k})
            vector_results = self._execute_vector_search(
                query_vector, vector_options, candidate_k
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
        highlight: HighlightOptions,
    ) -> List[RetrievedChunk]:
        """仅 BM25 检索"""
        return self._execute_bm25(query, options, highlight, options.top_k)

    # ============================================================
    # ES 查询执行
    # ============================================================

    def _execute_bm25(
        self,
        query: str,
        options: RetrievalOptions,
        highlight: HighlightOptions,
        top_k: int,
    ) -> List[RetrievedChunk]:
        """执行 BM25 检索"""
        bool_query = self._build_bm25_query(query, options)
        highlight_config = self._build_highlight(highlight)

        dsl: Dict[str, Any] = {
            "query": bool_query,
            "size": top_k,
            "_source": True,
        }

        if highlight and highlight.pre_tags:
            dsl["highlight"] = highlight_config

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
            response = self.es.search(index=self.chunks_index, body=dsl, _source=True)
        except Exception as e:
            # 维度不匹配时优雅降级
            if "different number of dimensions" in str(e):
                logger.warning(f"向量维度不匹配，向量检索降级: {e}")
            else:
                logger.error(f"向量检索失败: {e}")
            return []

        return self._parse_results_with_score(response)

    # ============================================================
    # BM25 查询构建（简化版）
    # ============================================================

    def _build_bm25_query(
        self,
        query: str,
        options: RetrievalOptions,
    ) -> Dict[str, Any]:
        """
        构建 BM25 查询

        核心字段参与检索增强：
        1. query: 主查询文本
        2. chunk_type(s): 加权 boost（偏好提示）
        3. keywords: 加权 boost
        4. target_models: should 加权偏好（非硬过滤）

        filter: is_latest=True, doc_ids
        """
        # ========== 可调整参数 ==========
        QUERY_BOOST = 1.0  # 主查询权重
        KEYWORDS_BOOST = 2.0  # 关键词权重
        CHUNK_TYPE_BOOST = 2.0  # chunk_type 加权（偏好提示，非硬过滤）
        ENTITY_BOOST = 3.0  # 实体加权（偏好提示，非硬过滤）
        # =================================

        # filter 条件（仅保留硬过滤）
        filter_conditions = [
            {"term": {"is_latest": True}},
        ]
        if options.doc_ids:
            filter_conditions.append({"terms": {"doc_id": options.doc_ids}})

        # should 条件
        should_conditions = []

        # target_models → should 加权偏好（keywords + context_summary + content）
        if options.target_models:
            for entity in options.target_models:
                should_conditions.append(
                    {"multi_match": {
                        "query": entity,
                        "fields": ["keywords", "context_summary", "content"],
                        "type": "best_fields",
                        "boost": ENTITY_BOOST,
                    }}
                )
            logger.info(f"  [target_models] should boost: {options.target_models}, boost={ENTITY_BOOST}")

        # 1. 主查询
        should_conditions.append(
            {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "entities_text", "context_summary"],
                    "type": "best_fields",
                    "boost": QUERY_BOOST,
                }
            }
        )

        # 2. chunk_type 加权
        if options.chunk_types:
            valid_types = [t for t in options.chunk_types if t and t != "other"]
            if valid_types:
                if len(valid_types) == 1:
                    should_conditions.append(
                        {
                            "term": {
                                "chunk_type": {
                                    "value": valid_types[0],
                                    "boost": CHUNK_TYPE_BOOST,
                                }
                            }
                        }
                    )
                else:
                    should_conditions.append(
                        {
                            "terms": {
                                "chunk_type": valid_types,
                                "boost": CHUNK_TYPE_BOOST,
                            }
                        }
                    )
                logger.info(f"  [chunk_type] {valid_types}, boost={CHUNK_TYPE_BOOST}")

        # 3. keywords 加权
        if options.keywords:
            keywords_query = " ".join(options.keywords)
            should_conditions.append(
                {
                    "multi_match": {
                        "query": keywords_query,
                        "fields": ["content", "entities_text"],
                        "type": "cross_fields",
                        "operator": "or",
                        "boost": KEYWORDS_BOOST,
                    }
                }
            )
            logger.info(f"  [keywords] {options.keywords}, boost={KEYWORDS_BOOST}")

        return {
            "bool": {
                "filter": filter_conditions,
                "should": should_conditions,
                "minimum_should_match": 2,
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
        构建向量 kNN 查询 + rescore

        向量搜索做语义召回，rescore 用文本条件（实体/关键词/chunk_type）
        对 top 候选重排序，解决"结构相同、产品不同"的区分问题。
        """
        filter_conditions = [
            {"term": {"is_latest": True}},
        ]
        if options.doc_ids:
            filter_conditions.append({"terms": {"doc_id": options.doc_ids}})

        # rescore 条件（文本加权，在语义召回后对 top 候选重评分）
        rescore_should = []

        # target_models → rescore 加权
        ENTITY_BOOST_VEC = 3.0
        if options.target_models:
            for entity in options.target_models:
                rescore_should.append(
                    {"multi_match": {
                        "query": entity,
                        "fields": ["keywords", "context_summary", "content"],
                        "type": "best_fields",
                        "boost": ENTITY_BOOST_VEC,
                    }}
                )
            logger.info(f"  [vec target_models] rescore boost: {options.target_models}, boost={ENTITY_BOOST_VEC}")

        num_candidates = max(top_k * 4, 50)

        # chunk_type → rescore 加权
        if options.chunk_types:
            valid_types = [t for t in options.chunk_types if t and t != "other"]
            for vt in valid_types:
                rescore_should.append({"term": {"chunk_type": {"value": vt, "boost": 2.0}}})
            if valid_types:
                logger.info(f"  [vec chunk_type] rescore {valid_types}, boost=2.0")

        # keywords → rescore 加权
        if options.keywords:
            keywords_text = " ".join(options.keywords)
            rescore_should.append({
                "multi_match": {
                    "query": keywords_text,
                    "fields": ["content", "entities_text"],
                    "type": "best_fields",
                    "boost": 1.5,
                }
            })
            logger.info(f"  [vec keywords] rescore {options.keywords}, boost=1.5")

        # 基础 knn 查询（纯语义召回）
        knn_query = {
            "knn": {
                "field": "embedding_vector",
                "query_vector": query_vector.tolist(),
                "k": top_k,
                "num_candidates": num_candidates,
                "filter": {"bool": {"filter": filter_conditions}},
            }
        }

        # 如果有文本加权条件，添加 rescore
        if rescore_should:
            knn_query["rescore"] = {
                "window_size": min(top_k * 2, num_candidates),
                "query": {
                    "rescore_query": {
                        "bool": {"should": rescore_should}
                    },
                    "query_weight": 0.6,
                    "rescore_query_weight": 0.4,
                }
            }

        return knn_query

    # ============================================================
    # 通用筛选
    # ============================================================
    # 高亮
    # ============================================================

    def _build_highlight(self, highlight: HighlightOptions) -> Dict[str, Any]:
        return {
            "fields": {
                "content": {
                    "pre_tags": highlight.pre_tags,
                    "post_tags": highlight.post_tags,
                    "fragment_size": highlight.fragment_size,
                    "number_of_fragments": highlight.number_of_fragments,
                },
                "context_summary": {
                    "pre_tags": highlight.pre_tags,
                    "post_tags": highlight.post_tags,
                    "fragment_size": highlight.fragment_size,
                    "number_of_fragments": 1,
                },
            },
            "require_field_match": False,
        }

    # ============================================================
    # 结果解析 — 适配 rag-clean 扁平字段
    # ============================================================

    def _parse_results_with_score(
        self, response: Dict[str, Any]
    ) -> List[RetrievedChunk]:
        """解析 ES 结果并保留原始得分"""
        hits = response.get("hits", {}).get("hits", [])

        chunks = []
        for hit in hits:
            source = hit.get("_source", {})
            highlights = hit.get("highlight", {})

            chunk = RetrievedChunk(
                chunk_id=source.get("chunk_id", ""),
                doc_id=source.get("doc_id", ""),
                content=source.get("content", ""),
                section_title=source.get("section_title"),
                score=hit.get("_score", 0.0),
                # 文档级扁平字段
                doc_type=source.get("doc_type"),
                domain=source.get("domain"),
                filter_terms=source.get("filter_terms") or [],
                # chunk 级字段
                chunk_type=source.get("chunk_type"),
                spec_table=source.get("spec_table"),
                spec_rows=source.get("spec_rows"),
                # Enrichment
                entities_text=source.get("entities_text"),
                keywords=source.get("keywords") or [],
                context_summary=source.get("context_summary"),
                # 父子导航
                parent_id=source.get("parent_id"),
                # 高亮
                highlight=highlights if highlights else {},
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

        # 构建文档文本 + 反向索引
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

        # 通过文本匹配回原始 chunk
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

    # ============================================================
    # Complex 路由检索
    # ============================================================

    def search_routed(
        self,
        rewritten_query: str,
        sub_queries: List[dict],
        options: Optional[RetrievalOptions] = None,
        highlight: Optional[HighlightOptions] = None,
    ) -> RetrievalResult:
        """Complex 查询路由：对每个 sub_query 分别检索，合并去重后统一 rerank"""
        timing = {}

        if options is None:
            options = RetrievalOptions()
        if highlight is None:
            highlight = HighlightOptions()

        sub_top_k = max(options.top_k // len(sub_queries), 8)

        all_chunks: List[RetrievedChunk] = []
        seen_ids: set = set()
        t0 = time.time()
        for sub in sub_queries:
            sub_query = sub.get("query", rewritten_query)
            sub_entity = sub.get("entity")
            sub_intent = sub.get("intent")

            sub_options = options.model_copy(
                update={
                    "top_k": sub_top_k,
                    "target_models": [sub_entity] if sub_entity else None,
                    "chunk_types": (
                        [sub_intent] if sub_intent and sub_intent != "other" else None
                    ),
                    "use_rerank": False,
                }
            )

            logger.info(
                f"  [sub_query] query={sub_query}, entity={sub_entity}, intent={sub_intent}"
            )

            chunks = self._hybrid_search(sub_query, sub_options, highlight)

            for c in chunks:
                if c.chunk_id not in seen_ids:
                    all_chunks.append(c)
                    seen_ids.add(c.chunk_id)

        logger.info(f"  [complex] 合并 {len(all_chunks)} 个去重结果")
        timing["retrieve"] = time.time() - t0

        # 统一 rerank
        t0 = time.time()
        if options.use_rerank and all_chunks:
            all_chunks = self._rerank(rewritten_query, all_chunks, options)
            timing["rerank"] = time.time() - t0

        all_chunks = all_chunks[: options.top_k]

        logger.info(
            f"  [complex] 完成: {len(all_chunks)} 结果, sub_queries={len(sub_queries)}"
        )

        return RetrievalResult(
            query=rewritten_query,
            total=len(all_chunks),
            chunks=all_chunks,
            timing=timing,
        )


# ── 全局实例 ──────────────────────────────────────────

_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """获取检索服务单例"""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
