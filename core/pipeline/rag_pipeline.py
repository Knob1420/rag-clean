"""
Query Rewrite + Retrieval Pipeline

基于 QueryRewriteService + RetrievalService 的串联 pipeline
支持 Intent 路由：high confidence 时使用不同策略

流程：Query Understanding → Query Rewrite → 混合检索 → Rerank → Parent 展开 → Generation
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from config import settings
from core.generation.generation import GenerationService, get_generation_service
from core.query_engineer.query_rewrite import QueryRewriteService, RewrittenQuery
from core.query_engineer.query_understanding import (
    QueryUnderstandingService,
    QueryUnderstandingResult,
    SubQuery,
)
from core.retrieve.retrieval import RetrievalService, RetrievalOptions
from core.retrieve.retrieval_models import RetrievedChunk, TokenUsage
from core.products.spec_matcher import query_products, format_spec_context
from store import get_store


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
        spec_context: str = "",
        per_sub_question_spec_context: Optional[Dict[str, str]] = None,
        per_sub_question_generation_constraints: Optional[Dict[str, List[str]]] = None,
        intent: str = "",
        per_query_bm25_chunks: Optional[Dict[str, List[RetrievedChunk]]] = None,
        per_query_vector_chunks: Optional[Dict[str, List[RetrievedChunk]]] = None,
        generation_answer: Optional[str] = None,
        generation_usage: Optional[TokenUsage] = None,
    ):
        self.original_query = original_query
        self.understanding_result = understanding_result
        self.rewritten_queries = rewritten_queries
        self.chunks = chunks
        # per_sub_question_chunks: key = sub_question 字符串, value = 该 sub_question 合并去重 + 独立 rerank 后的 chunks
        self.per_sub_question_chunks = per_sub_question_chunks
        # rq_intent_map: key = rewritten_query 字符串, value = 所属 sub_question 的 intent
        self.rq_intent_map = rq_intent_map
        self.per_query_bm25_chunks = per_query_bm25_chunks or {}
        self.per_query_vector_chunks = per_query_vector_chunks or {}
        self.total = total
        self.timing = timing
        # spec_context: 兼容旧代码，取第一条 sub_question 的结构化上下文
        self.spec_context = spec_context
        # per_sub_question_spec_context: key = sub_question 字符串, value = 该 sub_question 的结构化上下文
        self.per_sub_question_spec_context = per_sub_question_spec_context or {}
        # per_sub_question_generation_constraints: key = sub_question 字符串, value = 该 sub_question 的生成约束
        self.per_sub_question_generation_constraints = (
            per_sub_question_generation_constraints or {}
        )
        self.intent = intent
        self.generation_answer = generation_answer
        self.generation_usage = generation_usage

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


class RAGPipeline:
    """Query Rewrite + Retrieval + Generation 串联 pipeline"""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        query_rewrite_service: Optional[QueryRewriteService] = None,
        query_understanding_service: Optional[QueryUnderstandingService] = None,
        store=None,
    ):
        self.retrieval = retrieval_service or RetrievalService()
        self.query_rewrite = query_rewrite_service or QueryRewriteService()
        self.query_understanding = (
            query_understanding_service or QueryUnderstandingService()
        )
        self._store = store or get_store()

    def _expand_to_parent_chunks(
        self,
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        将 rerank 后的 child/summary chunks 做智能展开。

        逻辑：
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
        single_child_ids = []  # parent_id: only 1 child → 只需 fetch siblings
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
                logger.warning(f"[Pipeline] 批量获取 parent chunk 失败: {e}")

        # 4. 构建结果
        result: List[RetrievedChunk] = []
        seen_ids = set()  # 用于去重

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
            # 先加当前 child
            result.append(chunk)
            seen_ids.add(chunk.chunk_id)

            # fetch siblings: child chunks of same parent, sorted by created_at
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
        """
        获取指定 child chunk 的前后各 N 个 sibling（同一 parent 的其他 child chunks）。
        按 created_at 排序以确定位置关系。
        """
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
            logger.warning(f"[Pipeline] 获取 sibling chunks 失败: {e}")
            return []

        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            return []

        # 找到当前 chunk 在排序列表中的位置
        current_idx = -1
        for i, hit in enumerate(hits):
            if hit.get("_source", {}).get("chunk_id") == current_chunk_id:
                current_idx = i
                break

        if current_idx < 0:
            return []

        siblings: List[RetrievedChunk] = []
        # 前 N 个
        for i in range(current_idx - limit, current_idx):
            if i >= 0:
                src = hits[i].get("_source", {})
                siblings.append(
                    self._make_retrieved_chunk(src, hits[i].get("_score", 0.0))
                )

        # 后 N 个
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

    def _generate(
        self,
        query: str,
        result: "PipelineResult",
        generation_svc: GenerationService,
    ) -> Tuple[Optional[str], Optional[TokenUsage]]:
        """
        执行 LLM 生成。

        流程：
        - 单子问句：直接生成
        - 多子问句：逐条生成 → 合并 → 整合生成
        """
        understanding = result.understanding_result
        sub_queries = understanding.sub_queries if understanding else []

        if not sub_queries:
            return None, None

        if len(sub_queries) == 1:
            # 单子问句：直接生成
            sq = sub_queries[0]
            sq_chunks = result.per_sub_question_chunks.get(sq.query, result.chunks)
            sq_spec = result.per_sub_question_spec_context.get(sq.query, "")
            sq_constraints = result.per_sub_question_generation_constraints.get(
                sq.query, []
            )
            answer, usage = generation_svc.generate(
                query=query,
                chunks=sq_chunks,
                query_intent=sq.intent,
                spec_context=sq_spec,
                generation_constraints=sq_constraints,
            )
            return answer, usage
        else:
            # 多子问句：逐条生成 → 合并 → 整合生成
            merged_answers = []
            total_usage = None

            for sq in sub_queries:
                sq_chunks = result.per_sub_question_chunks.get(sq.query, [])
                if not sq_chunks:
                    continue
                sq_spec = result.per_sub_question_spec_context.get(sq.query, "")
                sq_constraints = result.per_sub_question_generation_constraints.get(
                    sq.query, []
                )
                sub_answer, sub_usage = generation_svc.generate(
                    query=sq.query,
                    chunks=sq_chunks,
                    query_intent=sq.intent,
                    spec_context=sq_spec,
                    generation_constraints=sq_constraints,
                )
                merged_answers.append(f"【{sq.query}】\n{sub_answer}")
                total_usage = sub_usage

            if not merged_answers:
                return None, None

            # 整合生成
            integrated = "\n\n---\n\n".join(merged_answers)
            system_prompt, user_prompt = generation_svc._build_integration_prompt(
                original_query=query,
                merged_answers=integrated,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            import openai
            from config import settings

            if settings.deepseek_api_key:
                client = openai.OpenAI(
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                )
                resp = client.chat.completions.create(
                    model=settings.deepseek_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000,
                )
                answer = resp.choices[0].message.content
            else:
                answer = integrated

            return answer, total_usage

    def run(
        self,
        query: str,
        top_k: int = 20,
        use_understand: bool = True,
        use_rewrite: bool = True,
        use_rerank: bool = True,
        rerank_top_k: Optional[int] = None,
        query_rewrite_only: bool = False,
        dataset_ids: Optional[List[str]] = None,
        use_generation: bool = True,
    ) -> PipelineResult:
        """
        执行 Query Rewrite + Retrieval + Generation pipeline

        流程：
        1. Query Understanding（多问句拆分 + 意图分类，可跳过）
        2. Query Rewrite（每条子问句重写）
        3. 混合检索 + Rerank（每条 sub_question 独立 rerank）
        4. Parent 展开
        5. Generation（多子问句合并生成）

        Args:
            query: 用户原始查询
            top_k: 检索返回数量
            use_understand: 是否使用 Query Understand（默认 True，关闭时用原始 query 作为单条子问句）
            use_rewrite: 是否使用 Query Rewrite（默认 True）
            use_rerank: 是否使用 Rerank
            rerank_top_k: Rerank 后保留数量
            query_rewrite_only: 如果为 True，跳过检索和生成
            dataset_ids: 按数据集 ID 列表筛选
            use_generation: 是否使用 LLM 生成（默认 True）

        Returns:
            PipelineResult
        """
        import time

        timing = {}

        # 0. Query Understanding（前置：多问句拆分 + 意图分类）
        t0 = time.time()
        if use_understand:
            understanding = self.query_understanding.parse(query)
        else:
            # 关闭时，用原始 query 构造一个简单的理解结果
            understanding = QueryUnderstandingResult(
                original_query=query,
                sub_queries=[SubQuery(query=query, intent="lookup")],
                generation_constraints=[],
            )
        timing["understanding"] = time.time() - t0

        # 1. Query Rewrite（每条子问句单独 rewrite，建立 sub_question → rewrite 结果映射）
        t0 = time.time()
        sub_questions = understanding.sub_queries

        rewritten_queries: List[str] = []
        sub_question_rewrite_map: Dict[str, RewrittenQuery] = (
            {}
        )  # key = sub_question.query

        for sq in sub_questions:
            if use_rewrite:
                rr = self.query_rewrite.rewrite(sq.query, sq.intent)
                # SemanticRouter 纠错后，intent 以 rr.intent 为准（rewrite 内部已处理）
            else:
                rr = RewrittenQuery(
                    original_query=sq.query,
                    intent=sq.intent,
                    rewritten_queries=[sq.query],
                )
            rewritten_queries.extend(rr.rewritten_queries)
            sub_question_rewrite_map[sq.query] = rr

        timing["rewrite"] = time.time() - t0

        # 构建 rq → intent 映射（用于 generation 时按 intent 选模板）
        # 使用 rr.intent 而非 sq.intent：SemanticRouter 纠错后的值
        rq_intent_map: Dict[str, str] = {}
        for sq in sub_questions:
            rr = sub_question_rewrite_map.get(sq.query)
            rr_intent = rr.intent if rr else sq.intent
            for rq in rr.rewritten_queries if rr else [sq.query]:
                rq_intent_map[rq] = rr_intent

        # 记录 intent（取第一条子问句 rewrite 后的 intent 作为主 intent）
        main_intent = (
            sub_question_rewrite_map.get(sub_questions[0].query).intent
            if sub_questions and sub_question_rewrite_map.get(sub_questions[0].query)
            else "lookup"
        )
        # 如果只运行 query rewrite 阶段，跳过检索和生成
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
                spec_context="",
                per_sub_question_spec_context={},
                per_sub_question_generation_constraints={},
                intent=main_intent,
                per_query_bm25_chunks={},
                per_query_vector_chunks={},
                generation_answer=None,
                generation_usage=None,
            )

        # 2. Retrieval（每条 sub_question 独立 retrieve → merge → rerank）
        t0 = time.time()
        all_chunks: List[RetrievedChunk] = []
        seen_chunk_ids = set()
        per_sub_question_chunks: Dict[str, List[RetrievedChunk]] = {}
        per_query_bm25_chunks: Dict[str, List[RetrievedChunk]] = {}
        per_query_vector_chunks: Dict[str, List[RetrievedChunk]] = {}

        # 细粒度计时
        t_rerank_total = 0.0
        t_parent_expand_total = 0.0
        # 汇总 _hybrid_search 内部的细粒度 timing
        agg_hybrid_timing: Dict[str, float] = {}

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
                # 预处理：构建 query_string + query_vector
                query_string = self.retrieval._build_bm25_query(rq, options)
                from core.client.embedder import encode
                query_vector = encode(rq)
                # 一次 hybrid 拿到 bm25 / vector / hybrid 三组结果
                fused_chunks, bm25_chunks, vector_chunks, ht = self.retrieval._hybrid_search(
                    query_string, query_vector, options, return_intermediates=True
                )
                per_query_bm25_chunks[rq] = bm25_chunks
                per_query_vector_chunks[rq] = vector_chunks

                # 汇总 hybrid_timing
                for k, v in ht.items():
                    agg_hybrid_timing[k] = agg_hybrid_timing.get(k, 0.0) + v

                # 合并去重（该 sub_question 内）
                top_k_per_rq = max(1, (top_k or rerank_top_k) // max(len(rq_list), 1))
                for chunk in fused_chunks[:top_k_per_rq]:
                    if chunk.chunk_id not in sq_seen:
                        sq_chunks.append(chunk)
                        sq_seen.add(chunk.chunk_id)

            # 该 sub_question 独立 rerank
            if use_rerank and sq_chunks:
                t_rr = time.time()
                from core.query_engineer.rerank_query import build_rerank_query
                rerank_query = build_rerank_query(sq.query)
                sq_chunks = self.retrieval._rerank(
                    rerank_query,
                    sq_chunks,
                    RetrievalOptions(
                        top_k=top_k,
                        rerank_top_k=rerank_top_k or top_k,
                        dataset_ids=dataset_ids,
                    ),
                )
                t_rerank_total += time.time() - t_rr

            # 展开 parent：child/summary → parent（按 parent_id 去重取 max rerank score）
            t_pe = time.time()
            sq_chunks = self._expand_to_parent_chunks(sq_chunks)
            t_parent_expand_total += time.time() - t_pe

            logger.info(
                f"[Pipeline] sub_question='{sq.query}' parent展开后: {len(sq_chunks)} 条"
            )

            per_sub_question_chunks[sq.query] = sq_chunks

            # 汇总到 all_chunks
            for chunk in sq_chunks:
                if chunk.chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)

        t_retrieve_total = time.time() - t0
        timing["retrieve"] = round(t_retrieve_total, 3)
        timing["rerank"] = round(t_rerank_total, 3)
        timing["parent_expand"] = round(t_parent_expand_total, 3)
        # 从 _hybrid_search 汇总的细粒度 timing
        for k, v in agg_hybrid_timing.items():
            timing[k] = round(v, 3)

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
                sq_spec = format_spec_context(spec_results, rr.intent)
                per_sub_question_spec_context[sq.query] = sq_spec
                logger.info(
                    f"[Pipeline] sub_question='{sq.query}' 结构化查询: {len(spec_results)} 个产品匹配"
                )
        # 兼容旧代码：spec_context 取第一条 sub_question 的结果
        if per_sub_question_spec_context and sub_questions:
            spec_context = per_sub_question_spec_context.get(sub_questions[0].query, "")

        # 构建 per_sub_question_generation_constraints
        # 顶层 generation_constraints 来自 _extract_constraints_from_query（如"翻译成英文"、"不超过50字"）
        # 所有 sub_question 共用同一份顶层约束
        top_constraints = understanding.generation_constraints if understanding else []
        per_sub_question_generation_constraints: Dict[str, List[str]] = {}
        for sq in sub_questions:
            per_sub_question_generation_constraints[sq.query] = top_constraints

        logger.info(
            f"[Pipeline] 完成: original_query='{query}', "
            f"sub_questions={len(sub_questions)}, rewritten={len(rewritten_queries)}, "
            f"total_chunks={len(all_chunks)}"
        )

        # 构建中间结果（用于 generation）
        result = PipelineResult(
            original_query=query,
            understanding_result=understanding,
            rewritten_queries=rewritten_queries,
            chunks=all_chunks,
            per_sub_question_chunks=per_sub_question_chunks,
            rq_intent_map=rq_intent_map,
            total=len(all_chunks),
            timing=timing,
            spec_context=spec_context,
            per_sub_question_spec_context=per_sub_question_spec_context,
            per_sub_question_generation_constraints=per_sub_question_generation_constraints,
            intent=main_intent,
            per_query_bm25_chunks=per_query_bm25_chunks,
            per_query_vector_chunks=per_query_vector_chunks,
            generation_answer=None,
            generation_usage=None,
        )

        # 5. Generation
        generation_answer = None
        generation_usage = None
        if use_generation and all_chunks:
            t0 = time.time()
            try:
                generation_svc = get_generation_service()
                generation_answer, generation_usage = self._generate(
                    query, result, generation_svc
                )
            except Exception as e:
                logger.warning(f"[Pipeline] Generation 失败: {e}")
            timing["generation"] = time.time() - t0

        result.generation_answer = generation_answer
        result.generation_usage = generation_usage

        return result
