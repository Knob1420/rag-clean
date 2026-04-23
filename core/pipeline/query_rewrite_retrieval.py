"""
Query Rewrite + Retrieval Pipeline

基于 QueryRewriteServiceV2 + RetrievalService 的串联 pipeline
支持 Intent 路由：high confidence 时使用不同策略
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from core.query_engineer.query_rewrite import QueryRewriteServiceV2, RewrittenQueryV2
from core.retrieve.retrieval import RetrievalService, RetrievalOptions
from core.retrieve.retrieval_models import RetrievedChunk
from core.router import RoutingResult
from core.products.spec_matcher import query_products, format_spec_context


class PipelineResult:
    """Pipeline 执行结果"""

    def __init__(
        self,
        original_query: str,
        rewritten_queries: List[str],
        chunks: List[RetrievedChunk],
        per_query_chunks: Dict[str, List[RetrievedChunk]],
        total: int,
        timing: dict,
        routing_result: Optional[RoutingResult] = None,
        per_query_bm25_chunks: Optional[Dict[str, List[RetrievedChunk]]] = None,
        per_query_vector_chunks: Optional[Dict[str, List[RetrievedChunk]]] = None,
        extracted_params: Optional[Dict[str, Any]] = None,
        spec_context: str = "",
        intent: str = "",
    ):
        self.original_query = original_query
        self.rewritten_queries = rewritten_queries
        self.chunks = chunks
        self.per_query_chunks = per_query_chunks
        self.total = total
        self.timing = timing
        self.routing_result = routing_result
        self.per_query_bm25_chunks = per_query_bm25_chunks or {}
        self.per_query_vector_chunks = per_query_vector_chunks or {}
        self.extracted_params = extracted_params or {}
        self.spec_context = spec_context
        self.intent = intent

    @property
    def intent_confidence(self) -> float:
        """获取路由置信度（LLM 路由无置信度，返回 1.0）"""
        return 1.0 if self.intent else 0.0

    def __repr__(self):
        return (
            f"PipelineResult(queries={self.rewritten_queries}, "
            f"chunks={len(self.chunks)}, total={self.total}, "
            f"intent={self.intent}, conf={self.intent_confidence:.2f})"
        )


class QueryRewriteRetrievalPipeline:
    """Query Rewrite + Retrieval 串联 pipeline"""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        query_rewrite_service: Optional[QueryRewriteServiceV2] = None,
    ):
        self.retrieval = retrieval_service or RetrievalService()
        self.query_rewrite = query_rewrite_service or QueryRewriteServiceV2()

    def run(
        self,
        query: str,
        top_k: int = 20,
        use_rewrite: bool = True,
        use_rerank: bool = True,
        rerank_top_k: Optional[int] = None,
        query_rewrite_only: bool = False,
        dataset_ids: Optional[List[str]] = None,
    ) -> PipelineResult:
        """
        执行 Query Rewrite + Retrieval pipeline

        Args:
            query: 用户原始查询
            top_k: 检索返回数量
            use_rewrite: 是否使用 Query Rewrite（默认 True，LLM 判断意图）
            use_rerank: 是否使用 Rerank
            rerank_top_k: Rerank 后保留数量
            dataset_ids: 按数据集 ID 列表筛选（如 ["products", "contracts"]）
            use_rewrite: 是否使用 Query Rewrite（默认 True，LLM 判断意图）
            use_rerank: 是否使用 Rerank
            rerank_top_k: Rerank 后保留数量

        Returns:
            PipelineResult: 包含原始查询、重写后的查询、检索结果等
        """
        import time

        timing = {}

        # 0. Query Rewrite（LLM 判断意图 + 提取参数）
        t0 = time.time()
        if use_rewrite:
            rewrite_result = self.query_rewrite.rewrite(query)
        else:
            rewrite_result = RewrittenQueryV2(
                original_query=query,
                intent="lookup",
                transform_strategy="direct",
                rewritten_queries=[query],
            )
        timing["rewrite"] = time.time() - t0

        # 记录 intent（如果有）
        routing_result = None

        # 如果只运行 query rewrite 阶段，跳过检索
        if query_rewrite_only:
            return PipelineResult(
                original_query=query,
                rewritten_queries=rewrite_result.rewritten_queries,
                chunks=[],
                per_query_chunks={},
                total=0,
                timing=timing,
                routing_result=routing_result,
                per_query_bm25_chunks={},
                per_query_vector_chunks={},
                extracted_params=rewrite_result.extracted_params,
                spec_context="",
                intent=rewrite_result.intent,
            )

        # 2. Retrieval（处理多个 rewritten_queries）
        t0 = time.time()
        all_chunks: List[RetrievedChunk] = []
        seen_chunk_ids = set()
        per_query_chunks: Dict[str, List[RetrievedChunk]] = {}  # 每个 query 的检索结果
        per_query_bm25_chunks: Dict[str, List[RetrievedChunk]] = {}
        per_query_vector_chunks: Dict[str, List[RetrievedChunk]] = {}

        # 先收集各查询的 hybrid+RRF 结果（不 rerank）
        query_results: List[Tuple[str, List[RetrievedChunk]]] = []
        for rewritten_query in rewrite_result.rewritten_queries:
            options = RetrievalOptions(
                top_k=top_k,
                use_rerank=False,
                dataset_ids=dataset_ids,
            )
            # 分别执行 BM25 和 vector，保留分开的结果
            bm25_chunks = self.retrieval._bm25_search(rewritten_query, options)
            vector_chunks = self.retrieval._vector_search(rewritten_query, options)

            # 融合
            rrf_results = self.retrieval.rrf.fuse(
                bm25_results=[(c, i) for i, c in enumerate(bm25_chunks)],
                vector_results=[(c, i) for i, c in enumerate(vector_chunks)],
            )
            fused_chunks = []
            for chunk, fused_score in rrf_results:
                chunk.score = fused_score
                fused_chunks.append(chunk)

            query_results.append((rewritten_query, fused_chunks))
            per_query_chunks[rewritten_query] = fused_chunks
            per_query_bm25_chunks[rewritten_query] = bm25_chunks
            per_query_vector_chunks[rewritten_query] = vector_chunks

        # 合并所有查询结果（去重，保持顺序）
        # 确保每个 sub_query 至少有 top_k_per_query 个候选
        top_k_per_query = max(1, rerank_top_k // len(query_results) if query_results else 1)
        for query, chunks in query_results:
            for chunk in chunks[:top_k_per_query]:
                if chunk.chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)
            # 如果 top_k_per_query 之外还有不重复的，也加入（保证更多候选）
            for chunk in chunks[top_k_per_query:]:
                if chunk.chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)

        # 最后统一做一次 rerank（使用原始 query）
        if use_rerank and all_chunks:
            all_chunks = self.retrieval._rerank(
                query,
                all_chunks,
                RetrievalOptions(top_k=top_k, rerank_top_k=rerank_top_k or top_k, dataset_ids=dataset_ids),
            )

        # TODO: 暂时不使用 parent context injection，直接用 child chunk
        # store = get_store()
        # all_chunks = _inject_parent_context(all_chunks, store)

        timing["retrieve"] = time.time() - t0

        # 3. 结构化参数查询（如果需要）
        spec_context = ""
        if rewrite_result.target_models or rewrite_result.required_fields:
            spec_results = query_products(
                target_models=rewrite_result.target_models,
                required_fields=rewrite_result.required_fields,
                numerical_constraints=rewrite_result.numerical_constraints,
            )
            spec_context = format_spec_context(spec_results, rewrite_result.intent)
            logger.info(f"[Pipeline] 结构化查询结果: {len(spec_results)} 个产品匹配")

        logger.info(
            f"[Pipeline] 完成: original_query='{query}', "
            f"rewritten_queries={rewrite_result.rewritten_queries}, "
            f"total_chunks={len(all_chunks)}"
        )

        return PipelineResult(
            original_query=query,
            rewritten_queries=rewrite_result.rewritten_queries,
            chunks=all_chunks,
            per_query_chunks=per_query_chunks,
            total=len(all_chunks),
            timing=timing,
            routing_result=routing_result,
            per_query_bm25_chunks=per_query_bm25_chunks,
            per_query_vector_chunks=per_query_vector_chunks,
            extracted_params=rewrite_result.extracted_params,
            spec_context=spec_context,
            intent=rewrite_result.intent,
        )
