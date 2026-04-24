"""
Query Rewrite + Retrieval Pipeline

基于 QueryRewriteServiceV2 + RetrievalService 的串联 pipeline
支持 Intent 路由：high confidence 时使用不同策略

流程：Query Understanding → Query Rewrite → 混合检索 → Rerank
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from core.query_engineer.query_rewrite import QueryRewriteServiceV2, RewrittenQueryV2
from core.query_engineer.query_understanding import (
    QueryUnderstandingService,
    QueryUnderstandingResult,
)
from core.retrieve.retrieval import RetrievalService, RetrievalOptions
from core.retrieve.retrieval_models import RetrievedChunk
from core.router import RoutingResult
from core.products.spec_matcher import query_products, format_spec_context


class PipelineResult:
    """Pipeline 执行结果"""

    def __init__(
        self,
        original_query: str,
        understanding_result: Optional[QueryUnderstandingResult],
        rewritten_queries: List[str],
        chunks: List[RetrievedChunk],
        per_sub_question_chunks: Dict[str, List[RetrievedChunk]],
        rq_intent_map: Dict[str, str],
        total: int,
        timing: dict,
        routing_result: Optional[RoutingResult] = None,
        extracted_params: Optional[Dict[str, Any]] = None,
        spec_context: str = "",
        per_sub_question_spec_context: Optional[Dict[str, str]] = None,
        per_sub_question_generation_constraints: Optional[Dict[str, List[str]]] = None,
        intent: str = "",
    ):
        self.original_query = original_query
        self.understanding_result = understanding_result
        self.rewritten_queries = rewritten_queries
        self.chunks = chunks
        # per_sub_question_chunks: key = sub_question 字符串, value = 该 sub_question 合并去重 + 独立 rerank 后的 chunks
        self.per_sub_question_chunks = per_sub_question_chunks
        # rq_intent_map: key = rewritten_query 字符串, value = 所属 sub_question 的 intent
        self.rq_intent_map = rq_intent_map
        self.total = total
        self.timing = timing
        self.routing_result = routing_result
        self.extracted_params = extracted_params or {}
        # spec_context: 兼容旧代码，取第一条 sub_question 的结构化上下文
        self.spec_context = spec_context
        # per_sub_question_spec_context: key = sub_question 字符串, value = 该 sub_question 的结构化上下文
        self.per_sub_question_spec_context = per_sub_question_spec_context or {}
        # per_sub_question_generation_constraints: key = sub_question 字符串, value = 该 sub_question 的生成约束
        self.per_sub_question_generation_constraints = per_sub_question_generation_constraints or {}
        self.intent = intent

    @property
    def intent_confidence(self) -> float:
        """获取路由置信度（LLM 路由无置信度，返回 1.0）"""
        return 1.0 if self.intent else 0.0

    def __repr__(self):
        return (
            f"PipelineResult(query={self.original_query[:30]!r}..., "
            f"sub_questions={len(self.understanding_result.sub_queries) if self.understanding_result else 0}, "
            f"chunks={len(self.chunks)}, intent={self.intent})"
        )


class QueryRewriteRetrievalPipeline:
    """Query Rewrite + Retrieval 串联 pipeline"""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        query_rewrite_service: Optional[QueryRewriteServiceV2] = None,
        query_understanding_service: Optional[QueryUnderstandingService] = None,
    ):
        self.retrieval = retrieval_service or RetrievalService()
        self.query_rewrite = query_rewrite_service or QueryRewriteServiceV2()
        self.query_understanding = (
            query_understanding_service or QueryUnderstandingService()
        )

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

        流程：
        1. Query Understanding（多问句拆分 + 意图分类）
        2. Query Rewrite（每条子问句重写）
        3. 混合检索 + Rerank（每条 sub_question 独立 rerank）

        Args:
            query: 用户原始查询
            top_k: 检索返回数量
            use_rewrite: 是否使用 Query Rewrite（默认 True）
            use_rerank: 是否使用 Rerank
            rerank_top_k: Rerank 后保留数量
            dataset_ids: 按数据集 ID 列表筛选

        Returns:
            PipelineResult
        """
        import time

        timing = {}

        # 0. Query Understanding（前置：多问句拆分 + 意图分类）
        t0 = time.time()
        understanding = self.query_understanding.parse(query)
        timing["understanding"] = time.time() - t0

        # 1. Query Rewrite（每条子问句单独 rewrite，建立 sub_question → rewrite 结果映射）
        t0 = time.time()
        sub_questions = understanding.sub_queries

        rewritten_queries: List[str] = []
        sub_question_rewrite_map: Dict[str, RewrittenQueryV2] = {}  # key = sub_question.query

        for sq in sub_questions:
            if use_rewrite:
                rr = self.query_rewrite.rewrite(sq.query, sq.intent)
            else:
                rr = RewrittenQueryV2(
                    original_query=sq.query,
                    intent=sq.intent,
                    transform_strategy="direct",
                    rewritten_queries=[sq.query],
                )
            rewritten_queries.extend(rr.rewritten_queries)
            sub_question_rewrite_map[sq.query] = rr

        timing["rewrite"] = time.time() - t0

        # 构建 rq → intent 映射（用于 generation 时按 intent 选模板）
        rq_intent_map: Dict[str, str] = {}
        for sq in sub_questions:
            rr = sub_question_rewrite_map.get(sq.query)
            for rq in (rr.rewritten_queries if rr else [sq.query]):
                rq_intent_map[rq] = sq.intent

        # 记录 intent（取第一条子问句的 intent 作为主 intent）
        main_intent = sub_questions[0].intent if sub_questions else "lookup"
        routing_result = None

        # 如果只运行 query rewrite 阶段，跳过检索
        if query_rewrite_only:
            return PipelineResult(
                original_query=query,
                understanding_result=understanding,
                rewritten_queries=rewritten_queries,
                chunks=[],
                per_sub_question_chunks={},
                rq_intent_map=rq_intent_map,
                total=0,
                timing=timing,
                routing_result=routing_result,
                extracted_params={},
                spec_context="",
                per_sub_question_spec_context={},
                per_sub_question_generation_constraints={},
                intent=main_intent,
            )

        # 2. Retrieval（每条 sub_question 独立 retrieve → merge → rerank）
        t0 = time.time()
        all_chunks: List[RetrievedChunk] = []
        seen_chunk_ids = set()
        per_sub_question_chunks: Dict[str, List[RetrievedChunk]] = {}

        for sq in sub_questions:
            rr = sub_question_rewrite_map.get(sq.query)
            rq_list = rr.rewritten_queries if rr else [sq.query]

            # 该 sub_question 下所有 rewritten_query 的 chunks 合并去重（rerank 前）
            sq_chunks: List[RetrievedChunk] = []
            sq_seen = set()

            for rq in rq_list:
                options = RetrievalOptions(
                    top_k=top_k,
                    use_rerank=False,
                    dataset_ids=dataset_ids,
                )
                bm25_chunks = self.retrieval._bm25_search(rq, options)
                vector_chunks = self.retrieval._vector_search(rq, options)

                # RRF 融合
                rrf_results = self.retrieval.rrf.fuse(
                    bm25_results=[(c, i) for i, c in enumerate(bm25_chunks)],
                    vector_results=[(c, i) for i, c in enumerate(vector_chunks)],
                )
                fused_chunks = []
                for chunk, fused_score in rrf_results:
                    chunk.score = fused_score
                    fused_chunks.append(chunk)

                # 合并去重（该 sub_question 内）
                top_k_per_rq = max(1, (rerank_top_k or top_k) // max(len(rq_list), 1))
                for chunk in fused_chunks[:top_k_per_rq]:
                    if chunk.chunk_id not in sq_seen:
                        sq_chunks.append(chunk)
                        sq_seen.add(chunk.chunk_id)

            # 该 sub_question 独立 rerank
            if use_rerank and sq_chunks:
                sq_chunks = self.retrieval._rerank(
                    sq.query,
                    sq_chunks,
                    RetrievalOptions(
                        top_k=top_k,
                        rerank_top_k=rerank_top_k or top_k,
                        dataset_ids=dataset_ids,
                    ),
                )

            per_sub_question_chunks[sq.query] = sq_chunks

            # 汇总到 all_chunks
            for chunk in sq_chunks:
                if chunk.chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)

        timing["retrieve"] = time.time() - t0

        # 3. 结构化参数查询（每条 sub_question 独立查询）
        spec_context = ""
        per_sub_question_spec_context: Dict[str, str] = {}
        for sq in sub_questions:
            rr = sub_question_rewrite_map.get(sq.query)
            if rr and (rr.entities or rr.required_fields):
                spec_results = query_products(
                    target_models=rr.entities,
                    required_fields=rr.required_fields,
                    numerical_constraints=rr.numerical_constraints,
                )
                sq_spec = format_spec_context(spec_results, sq.intent)
                per_sub_question_spec_context[sq.query] = sq_spec
                logger.info(
                    f"[Pipeline] sub_question='{sq.query}' 结构化查询: {len(spec_results)} 个产品匹配"
                )
        # 兼容旧代码：spec_context 取第一条 sub_question 的结果
        if per_sub_question_spec_context and sub_questions:
            spec_context = per_sub_question_spec_context.get(sub_questions[0].query, "")

        # 构建 per_sub_question_generation_constraints
        per_sub_question_generation_constraints: Dict[str, List[str]] = {}
        for sq in sub_questions:
            per_sub_question_generation_constraints[sq.query] = sq.generation_constraints

        logger.info(
            f"[Pipeline] 完成: original_query='{query}', "
            f"sub_questions={len(sub_questions)}, rewritten={len(rewritten_queries)}, "
            f"total_chunks={len(all_chunks)}"
        )

        return PipelineResult(
            original_query=query,
            understanding_result=understanding,
            rewritten_queries=rewritten_queries,
            chunks=all_chunks,
            per_sub_question_chunks=per_sub_question_chunks,
            rq_intent_map=rq_intent_map,
            total=len(all_chunks),
            timing=timing,
            routing_result=routing_result,
            extracted_params=sub_question_rewrite_map.get(sub_questions[0].query, RewrittenQueryV2()).extracted_params if sub_questions else {},
            spec_context=spec_context,
            per_sub_question_spec_context=per_sub_question_spec_context,
            per_sub_question_generation_constraints=per_sub_question_generation_constraints,
            intent=main_intent,
        )
