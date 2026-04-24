"""
查询理解与重写服务

基于 rag-knowledge-base query_rewrite.py，适配 rag-clean：
- ES 聚合用 doc_type（扁平字段）而非 metadata.doc_type.keyword
- 复用 llm.py 的 LLMClient.call() + parse_json_response()
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from config import settings
from core.generation.llm import LLMClient, get_llm_client, parse_json_response
from prompt import INTENT_ROUTING_PROMPT


class RewrittenQuery(BaseModel):
    """重写后的查询"""

    original_query: str = ""
    rewritten_query: str
    intent_type: str = (
        "other"  # → ES chunk_type should boost + generation intent context
    )
    target_entities: List[str] = Field(default_factory=list)  # → ES entity boost
    keywords: List[str] = Field(default_factory=list)  # → ES keywords boost
    strategy: str = "direct"  # direct | parallel | sequential → 路由决策
    sub_queries: List[dict] = Field(default_factory=list)  # → parallel path 输入

    @staticmethod
    def _validate_strategy(v: str) -> str:
        if v not in ("direct", "parallel", "sequential"):
            return "direct"
        return v

    def model_post_init(self, __context):
        self.strategy = self._validate_strategy(self.strategy)

    @property
    def intent(self) -> str:
        """兼容旧代码：返回 intent_type"""
        return self.intent_type

    @property
    def entities(self) -> Dict[str, str]:
        """兼容旧代码：将 target_entities 转换为字典"""
        if not self.target_entities:
            return {}
        return {"实体": ",".join(self.target_entities)}

    @property
    def target_models(self) -> List[str]:
        """兼容旧代码"""
        return self.target_entities


class QueryRewriteService:
    """查询理解与重写服务"""

    def __init__(self):
        self.llm = get_llm_client()
        self._store = get_store()
        self.chunks_index = settings.es_index_chunks
        self._chunk_types_cache: Optional[List[str]] = None

    @property
    def es(self):
        return self._store.es

    # ============================================================
    # ES chunk_type 聚合
    # ============================================================

    def refresh_type_cache(self):
        """从 ES 聚合获取 chunk 类型，刷新缓存"""
        try:
            agg = self.es.search(
                index=self.chunks_index,
                body={
                    "size": 0,
                    "aggs": {
                        "chunk_types": {"terms": {"field": "chunk_type", "size": 50}},
                    },
                },
            )

            chunk_buckets = (
                agg.get("aggregations", {}).get("chunk_types", {}).get("buckets", [])
            )
            self._chunk_types_cache = [b["key"] for b in chunk_buckets]

            logger.info(
                f"[Query Rewrite] chunk_type 缓存已刷新: {len(chunk_buckets)} 种"
            )

        except Exception as e:
            logger.warning(f"[Query Rewrite] ES 聚合失败，使用默认缓存: {e}")
            self._chunk_types_cache = []

    def _get_chunk_types_list(self) -> str:
        """获取 chunk_type 列表文本（懒加载）"""
        if self._chunk_types_cache is None:
            self.refresh_type_cache()
        if self._chunk_types_cache:
            return ", ".join(self._chunk_types_cache)
        return "intro, spec_data, feature, procedure, faq, profile, other"

    # ============================================================
    # 核心入口
    # ============================================================

    def rewrite(self, query: str) -> RewrittenQuery:
        """重写查询"""
        logger.info(f"[Query Rewrite] 原始查询: {query}")

        prompt = self._build_rewrite_prompt(query)

        try:
            response = self.llm.call(
                [
                    {
                        "role": "system",
                        "content": QUERY_REWRITE_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            result = parse_json_response(response)
            result["original_query"] = query

            # 兼容旧字段名 → 新字段名
            if "target_models" in result and "target_entities" not in result:
                models = result.pop("target_models", [])
                others = result.pop("other_entities", [])
                seen = set()
                merged = []
                for e in models + others:
                    if e and e not in seen:
                        merged.append(e)
                        seen.add(e)
                result["target_entities"] = merged

            rewritten = RewrittenQuery(**result)

            logger.info(f"[Query Rewrite] 完成:")
            logger.info(
                f"  strategy={rewritten.strategy}, intent={rewritten.intent_type}"
            )
            logger.info(f"  entities={rewritten.target_entities}")
            logger.info(f"  rewritten: {rewritten.rewritten_query}")

            return rewritten

        except Exception as e:
            logger.error(f"[Query Rewrite] 失败: {e}")
            return RewrittenQuery(
                original_query=query,
                rewritten_query=query,
            )

    # ============================================================
    # Prompt 构建
    # ============================================================

    def _build_rewrite_prompt(self, query: str) -> str:
        """构建重写提示词（动态注入 ES chunk_type 列表）"""
        chunk_types_list = self._get_chunk_types_list()
        return build_rewrite_prompt(query, chunk_types_list)


# ── 全局实例 ──────────────────────────────────────────

_query_rewrite_service: Optional[QueryRewriteService] = None


def get_query_rewrite_service() -> QueryRewriteService:
    """获取查询重写服务单例"""
    global _query_rewrite_service
    if _query_rewrite_service is None:
        _query_rewrite_service = QueryRewriteService()
    return _query_rewrite_service


# ════════════════════════════════════════════════════════════
# V2 版本：基于意图的查询转换（Intent-Driven Query Transform）
# ════════════════════════════════════════════════════════════


class RewrittenQueryV2(BaseModel):
    """V2 版本重写后的查询"""

    original_query: str = ""
    intent: str = ""  # lookup | compare | recommend | aggregate
    transform_strategy: str = ""  # clarify | decompose | generalize（保留兼容）

    # 重写后的查询列表（compare 时可能有多个）
    rewritten_queries: List[str] = Field(default_factory=list)

    # LLM 判断后的结构化结果
    entities: List[str] = Field(default_factory=list)  # 主要实体（产品型号/文档名/合作单位等）
    required_fields: List[str] = Field(default_factory=list)  # 关心的参数字段
    numerical_constraints: Dict[str, str] = Field(default_factory=dict)  # 数值约束，格式 {"重量": "<=3.0"}

    # 兼容旧字段
    extracted_params: Dict[str, Any] = Field(default_factory=dict)


class QueryRewriteServiceV2:
    """
    V2 版本查询理解与重写服务（基于意图的查询转换）

    转换策略：
    - simple_lookup → clarify（澄清查询，使检索更精准）
    - compare / reasoning → decompose（分解为多个子查询，并行检索后合并）
    - recommend → generalize（泛化查询，step back 获取更宽视角）
    - aggregate → decompose（分解为统计子查询）

    所有路径均会提取结构化参数，置于检索上下文最前方。
    """

    # ── Prompt 模板 ──────────────────────────────────────────

    CLARIFY_SYSTEM_PROMPT = """You are an expert query analyzer for a space satellite product knowledge base.

Your task is to CLARIFY the user's query for better document retrieval.

Rules:
1. ONLY make explicit what is IMPLICIT in the user's query (e.g., "它的重量" → "星载智算机NX1的重量")
   - Do NOT assume or infer specific product names, model numbers unless explicitly stated
   - If the user says "首发之前的智算机", do NOT assume it means NX1 — keep the query general
2. Preserve all product names, model numbers, and technical terms exactly
3. Fix grammar and remove conversational filler words
4. If multiple distinct products are mentioned, clarify which one matters
5. Keep the query focused — do not expand scope

Output JSON:
{
    "rewritten_query": "clarified query string (if already clear, use original)",
    "extracted_params": {"product_type": "...", "model": "...", "param_names": [...], "constraints": {...}}
}"""

    DECOMPOSE_SYSTEM_PROMPT = """You are an expert query analyzer for a space satellite product knowledge base.

Your task is to DECOMPOSE a comparison or reasoning query into parallel sub-queries.

Rules:
1. Each sub-query should focus on ONE product/model or ONE comparison dimension
2. Preserve all specific product names, model numbers, and technical terms
3. Sub-queries should be self-contained for independent retrieval
4. Maximum 4 sub-queries — combine if more are needed
5. Extract comparison dimensions (e.g., 重量, 算力, 功耗, 尺寸)

Output JSON:
{
    "sub_queries": ["sub_query_1", "sub_query_2", ...],
    "comparison_dimensions": ["dimension1", "dimension2", ...],
    "extracted_params": {"product_type": "...", "models": [...], "dimensions": [...]}
}"""

    GENERALIZE_SYSTEM_PROMPT = """You are an expert query analyzer for a space satellite product knowledge base.

Your task is to GENERALIZE (step back) a recommendation query to find the best option.

Rules:
1. Step back from specific models to product categories
2. Identify the user's core requirement (what do they want to achieve?)
3. Broaden the query to cover all relevant options, not just named models
4. Preserve key constraints (e.g., weight < 3kg, specific interface requirements)
5. Make the generalized query a good retrieval query for finding recommendations

Output JSON:
{
    "generalized_query": "broader query for finding best option",
    "original_specifics": {"constraints": [...], "implied_needs": [...]},
    "extracted_params": {"product_type": "...", "constraints": {...}}
}"""

    def __init__(self):
        self.llm = get_llm_client()

    def _transform_query(
        self,
        query: str,
        intent: str,
    ) -> RewrittenQueryV2:
        """
        统一的 query transform（V2 版本）

        使用一个统一的 prompt，根据 intent 对 query 进行改写：
        - lookup: clarify（澄清查询）
        - compare/recommend/aggregate: decompose/generalize（拆分/泛化）

        Args:
            query: 用户原始查询
            intent: 已识别出的意图类型

        Returns:
            RewrittenQueryV2: 包含重写后的查询及结构化指令
        """
        from prompt import UNIFIED_TRANSFORM_PROMPT

        logger.info(f"[Query Rewrite V2] 统一 transform, intent={intent}")

        prompt = UNIFIED_TRANSFORM_PROMPT.replace("__INTENT__", intent).replace("__QUERY__", query)
        response = self.llm.call(
            [
                {"role": "system", "content": "你是一个专业的查询改写助手。"},
                {"role": "user", "content": prompt},
            ]
        )

        result = parse_json_response(response)
        result["original_query"] = query
        result["intent"] = intent

        # 解析 constraints（格式：{"重量": "<=3.0"} 或 {"重量": {"operator": "<=", "value": 3.0}}）
        constraints_raw = result.get("constraints", {})
        numerical_constraints = {}
        if isinstance(constraints_raw, dict):
            for field, op_val in constraints_raw.items():
                if isinstance(op_val, dict):
                    op = op_val.get("operator", "<=")
                    val = op_val.get("value", "")
                else:
                    # 直接是字符串格式如 "<=3.0"
                    val = str(op_val)
                    op = ""
                if op:
                    numerical_constraints[field] = f"{op}{val}"
                else:
                    numerical_constraints[field] = val
        elif isinstance(constraints_raw, list):
            # LLM 有时返回 list 而非 dict，忽略
            pass

        rewritten = RewrittenQueryV2(
            original_query=query,
            intent=intent,
            rewritten_queries=result.get("search_queries") if isinstance(result.get("search_queries"), list) else [],
            entities=result.get("entities") if isinstance(result.get("entities"), list) else [],
            required_fields=result.get("required_fields") if isinstance(result.get("required_fields"), list) else [],
            numerical_constraints=numerical_constraints,
            extracted_params=result.get("extracted_params") if isinstance(result.get("extracted_params"), dict) else {},
        )

        logger.info(
            f"[Query Rewrite V2] 完成: intent={rewritten.intent}, "
            f"rewritten_queries={rewritten.rewritten_queries}, "
            f"entities={rewritten.entities}, "
            f"constraints={rewritten.numerical_constraints}"
        )
        return rewritten

    def rewrite(
        self,
        query: str,
        intent: Optional[str] = None,
    ) -> RewrittenQueryV2:
        """
        基于意图的查询转换（V2 版本）

        流程：
        1. 使用 embedding 语义路由判断意图
        2. 统一 transform（根据 intent 改写 query）

        Args:
            query: 用户原始查询
            intent: 意图类型（可选，如果为 None 则由 SemanticRouter 判断）

        Returns:
            RewrittenQueryV2: 包含意图判断结果及提取的参数
        """
        logger.info(f"[Query Rewrite V2] 原始查询: {query}, 传入intent: {intent}")

        try:
            # 第一步：意图识别（使用 embedding 语义路由）
            if intent:
                detected_intent = intent
            else:
                from core.router import get_semantic_router

                router = get_semantic_router()
                detected_intent, confidence = router.classify_intent(query)
                logger.info(
                    f"[Query Rewrite V2] SemanticRouter: intent={detected_intent}, conf={confidence:.3f}"
                )

            # 第二步：统一 transform
            rewritten = self._transform_query(query, detected_intent)

            return rewritten

        except Exception as e:
            logger.error(f"[Query Rewrite V2] 失败: {e}")
            import traceback
            traceback.print_exc()
            return RewrittenQueryV2(
                original_query=query,
                intent=intent or "simple_lookup",
                rewritten_queries=[query],
            )

    def _extract_params(self, query: str) -> Dict[str, Any]:
        """
        使用 LLM 从查询中提取结构化参数

        Returns:
            {
                "target_models": [...],
                "required_fields": [...],
                "numerical_constraints": {...}
            }
        """
        from prompt import INTENT_ROUTING_PROMPT

        try:
            response = self.llm.call(
                [
                    {"role": "system", "content": INTENT_ROUTING_PROMPT},
                    {"role": "user", "content": f"用户问题: {query}"},
                ]
            )

            result = parse_json_response(response)

            return {
                "target_models": result.get("target_models", []),
                "required_fields": result.get("required_fields", []),
                "numerical_constraints": result.get("numerical_constraints", {}),
            }
        except Exception as e:
            logger.warning(f"[Query Rewrite V2] 参数提取失败: {e}")
            return {
                "target_models": [],
                "required_fields": [],
                "numerical_constraints": {},
            }

    # ── 转换策略实现（保留兼容）───────────────────────────────────────

    def _clarify_query(
        self,
        query: str,
    ) -> RewrittenQueryV2:
        """clarify 策略：适用于 simple_lookup 和 aggregate 意图"""
        logger.info("[Query Rewrite V2] 策略: clarify (clarify query)")

        prompt = self._build_clarify_prompt(query)
        response = self.llm.call(
            [
                {"role": "system", "content": self.CLARIFY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        result = parse_json_response(response)
        result["original_query"] = query
        result["intent"] = "simple_lookup"
        result["transform_strategy"] = "clarify"

        # 如果没有返回 rewritten_query 或为空，说明 query 本身已清晰
        if not result.get("rewritten_query"):
            result["rewritten_query"] = query

        # 确保 rewritten_queries 是列表
        if "rewritten_queries" not in result:
            result["rewritten_queries"] = [result["rewritten_query"]]

        # 提取顶层参数字段
        extracted = result.get("extracted_params", {})
        # models可能是字符串或列表
        models = extracted.get("models", []) or extracted.get("model", [])
        if isinstance(models, str):
            models = [models] if models else []
        result["target_models"] = models or []
        result["required_fields"] = extracted.get("param_names", [])
        result["numerical_constraints"] = extracted.get("constraints", {})

        rewritten = RewrittenQueryV2(**result)

        logger.info(f"[Query Rewrite V2] clarify 完成: {rewritten.rewritten_queries}")
        return rewritten

    def _decompose_query(
        self,
        query: str,
    ) -> RewrittenQueryV2:
        """decompose 策略：适用于 compare 意图"""
        logger.info("[Query Rewrite V2] 策略: decompose (break into sub-queries)")

        prompt = self._build_decompose_prompt(query)
        response = self.llm.call(
            [
                {"role": "system", "content": self.DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        result = parse_json_response(response)
        result["original_query"] = query
        result["intent"] = "compare"
        result["transform_strategy"] = "decompose"

        if result.get("clarification_needed") is None:
            result["clarification_needed"] = ""

        # sub_queries 字段转为 rewritten_queries
        if "sub_queries" in result:
            result["rewritten_queries"] = result.pop("sub_queries")

        # 提取顶层参数字段
        extracted = result.get("extracted_params", {})
        result["target_models"] = extracted.get("models") or extracted.get("model") or []
        result["required_fields"] = extracted.get("dimensions", [])
        result["numerical_constraints"] = {}

        rewritten = RewrittenQueryV2(**result)

        logger.info(
            f"[Query Rewrite V2] decompose 完成: {len(rewritten.rewritten_queries)} sub-queries"
        )
        return rewritten

    def _generalize_query(
        self,
        query: str,
    ) -> RewrittenQueryV2:
        """generalize 策略：适用于 recommend 意图"""
        logger.info("[Query Rewrite V2] 策略: generalize (step back)")

        prompt = self._build_generalize_prompt(query)
        response = self.llm.call(
            [
                {"role": "system", "content": self.GENERALIZE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        result = parse_json_response(response)
        result["original_query"] = query
        result["intent"] = "recommend"
        result["transform_strategy"] = "generalize"

        if result.get("clarification_needed") is None:
            result["clarification_needed"] = ""

        # generalized_query 字段转为 rewritten_queries (作为唯一项)
        if "generalized_query" in result:
            result["rewritten_queries"] = [result.pop("generalized_query")]

        # 提取顶层参数字段
        extracted = result.get("extracted_params", {})
        original_specifics = result.get("original_specifics", {})
        result["target_models"] = []
        result["required_fields"] = original_specifics.get("implied_needs", [])

        # numerical_constraints需要是Dict[str, str]
        raw_constraints = extracted.get("constraints", {})
        constraints = {}
        for k, v in raw_constraints.items():
            if isinstance(v, list):
                constraints[k] = ", ".join(str(x) for x in v)
            else:
                constraints[k] = str(v)
        result["numerical_constraints"] = constraints

        rewritten = RewrittenQueryV2(**result)

        logger.info(
            f"[Query Rewrite V2] generalize 完成: {rewritten.rewritten_queries}"
        )
        return rewritten

    # ── Prompt 构建辅助 ────────────────────────────────────

    def _build_clarify_prompt(
        self,
        query: str,
    ) -> str:
        parts = [f"原始查询: {query}"]
        parts.append("请澄清此查询，使其清晰且可检索。")
        return "\n".join(parts)

    def _build_decompose_prompt(
        self,
        query: str,
    ) -> str:
        parts = [f"比较/推理查询: {query}"]
        parts.append("请将此查询分解为多个可并行检索的子查询。")
        return "\n".join(parts)

    def _build_generalize_prompt(
        self,
        query: str,
    ) -> str:
        parts = [f"推荐查询: {query}"]
        parts.append("请泛化此查询，从产品类别角度找到最佳选项。")
        return "\n".join(parts)
