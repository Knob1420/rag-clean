"""
查询理解与重写服务

基于意图的查询转换（Intent-Driven Query Transform）：

转换策略：
- lookup → clarify（澄清查询，使检索更精准）
- compare / reasoning → decompose（分解为多个子查询，并行检索后合并）
- recommend → generalize（泛化查询，step back 获取更宽视角）
- aggregate → decompose（分解为统计子查询）

所有路径均会提取结构化参数，置于检索上下文最前方。
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from prompt import INTENT_ROUTING_PROMPT


class RewrittenQuery(BaseModel):
    """重写后的查询"""

    original_query: str = ""
    intent: str = ""  # lookup | compare | recommend | aggregate

    # 重写后的查询列表（compare 时可能有多个）
    rewritten_queries: List[str] = Field(default_factory=list)

    # 结构化参数 → 传入 SpecMatcher
    entities: List[str] = Field(
        default_factory=list
    )  # 主要实体（产品型号/文档名/合作单位等）
    required_fields: List[str] = Field(default_factory=list)  # 关心的参数字段
    numerical_constraints: Dict[str, str] = Field(
        default_factory=dict
    )  # 数值约束，格式 {"重量": "<=3.0"}


class QueryRewriteService:
    """
    基于意图的查询理解与重写服务

    提供两套 rewrite 路径：
    - rewrite(query, intent)：统一 prompt 路径（当前实际使用）
    - _clarify_query / _decompose_query / _generalize_query：策略专属 prompt（保留，待启用）
    """

    # ── 策略专属 Prompt 模板（保留，待启用）──────────────────────

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
        from core.generation.llm import get_llm_client, parse_json_response
        self.llm = get_llm_client()
        self._parse_json = parse_json_response

    # ── 统一 transform（当前实际路径）────────────────────────────

    def _transform_query(
        self,
        query: str,
        intent: str,
    ) -> RewrittenQuery:
        """
        统一的 query transform

        使用一个统一的 prompt，根据 intent 对 query 进行改写：
        - lookup: clarify（澄清查询）
        - compare/recommend/aggregate: decompose/generalize（拆分/泛化）

        Args:
            query: 用户原始查询
            intent: 已识别出的意图类型

        Returns:
            RewrittenQuery: 包含重写后的查询及结构化指令
        """
        from prompt import UNIFIED_TRANSFORM_PROMPT

        logger.info(f"[Query Rewrite] 统一 transform, intent={intent}")

        prompt = UNIFIED_TRANSFORM_PROMPT.replace("__INTENT__", intent).replace(
            "__QUERY__", query
        )
        response = self.llm.call(
            [
                {"role": "system", "content": "你是一个专业的查询改写助手。"},
                {"role": "user", "content": prompt},
            ]
        )

        result = self._parse_json(response)
        result["original_query"] = query
        result["intent"] = intent

        # 解析 constraints
        # LLM 返回格式: [{"field": "重量", "operator": "<=", "value": 2.0}]
        constraints_raw = result.get("constraints", {})
        numerical_constraints = {}
        if isinstance(constraints_raw, list):
            for item in constraints_raw:
                if not isinstance(item, dict):
                    continue
                field = item.get("field", "")
                operator = item.get("operator", "")
                value = item.get("value", "")
                if field and operator:
                    numerical_constraints[field] = f"{operator}{value}"
        elif isinstance(constraints_raw, dict):
            # 兼容 dict 格式: {"重量": "<=3.0"} 或 {"重量": {"operator": "<=", "value": 3.0}}
            for field, op_val in constraints_raw.items():
                if isinstance(op_val, dict):
                    op = op_val.get("operator", "<=")
                    val = op_val.get("value", "")
                else:
                    val = str(op_val)
                    op = ""
                if op:
                    numerical_constraints[field] = f"{op}{val}"
                else:
                    numerical_constraints[field] = val

        rewritten = RewrittenQuery(
            original_query=query,
            intent=intent,
            rewritten_queries=(
                result.get("search_queries")
                if isinstance(result.get("search_queries"), list)
                else []
            ),
            entities=(
                result.get("entities")
                if isinstance(result.get("entities"), list)
                else []
            ),
            required_fields=(
                result.get("required_fields")
                if isinstance(result.get("required_fields"), list)
                else []
            ),
            numerical_constraints=numerical_constraints,
        )

        logger.info(
            f"[Query Rewrite] 完成: intent={rewritten.intent}, "
            f"rewritten_queries={rewritten.rewritten_queries}, "
            f"entities={rewritten.entities}, "
            f"constraints={rewritten.numerical_constraints}"
        )
        return rewritten

    def rewrite(
        self,
        query: str,
        intent: Optional[str] = None,
    ) -> RewrittenQuery:
        """
        基于意图的查询转换

        流程：
        1. 始终调用 SemanticRouter 获取意图，与传入 intent 交叉验证
        2. 两者不一致时以 SemanticRouter 为准并记录警告
        3. 统一 transform（根据确认后的 intent 改写 query）

        Args:
            query: 用户原始查询
            intent: 意图类型（来自 QueryUnderstanding，可选）

        Returns:
            RewrittenQuery: 包含意图判断结果及提取的参数
        """
        logger.info(f"[Query Rewrite] 原始查询: {query}, 传入intent: {intent}")

        try:
            # 第一步：意图识别（始终调用 SemanticRouter 做交叉验证）
            from core.router import get_semantic_router

            router = get_semantic_router()
            router_intent, confidence = router.classify_intent(query)

            if intent and intent != router_intent:
                # 意图不一致，以 SemanticRouter 为准
                logger.warning(
                    f"[Query Rewrite] intent 不一致: 传入={intent}, SemanticRouter={router_intent} "
                    f"(conf={confidence:.3f})，以 SemanticRouter 为准"
                )
                detected_intent = router_intent
            elif intent:
                # 意图一致，使用传入值
                detected_intent = intent
                logger.info(
                    f"[Query Rewrite] SemanticRouter: intent={router_intent}, conf={confidence:.3f}（与传入 intent 一致）"
                )
            else:
                # 无传入 intent，使用 SemanticRouter 结果
                detected_intent = router_intent
                logger.info(
                    f"[Query Rewrite] SemanticRouter: intent={detected_intent}, conf={confidence:.3f}"
                )

            # 第二步：统一 transform
            rewritten = self._transform_query(query, detected_intent)

            return rewritten

        except Exception as e:
            logger.error(f"[Query Rewrite] 失败: {e}")
            import traceback

            traceback.print_exc()
            return RewrittenQuery(
                original_query=query,
                intent=intent or "simple_lookup",
                rewritten_queries=[query],
            )

    # ── 策略专属方法（保留，待启用）─────────────────────────────

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
        try:
            response = self.llm.call(
                [
                    {"role": "system", "content": INTENT_ROUTING_PROMPT},
                    {"role": "user", "content": f"用户问题: {query}"},
                ]
            )

            result = self._parse_json(response)

            return {
                "target_models": result.get("target_models", []),
                "required_fields": result.get("required_fields", []),
                "numerical_constraints": result.get("numerical_constraints", {}),
            }
        except Exception as e:
            logger.warning(f"[Query Rewrite] 参数提取失败: {e}")
            return {
                "target_models": [],
                "required_fields": [],
                "numerical_constraints": {},
            }

    def _clarify_query(self, query: str) -> RewrittenQuery:
        """clarify 策略：适用于 simple_lookup 和 aggregate 意图"""
        logger.info("[Query Rewrite] 策略: clarify (clarify query)")

        prompt = self._build_clarify_prompt(query)
        response = self.llm.call(
            [
                {"role": "system", "content": self.CLARIFY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        result = self._parse_json(response)
        result["original_query"] = query
        result["intent"] = "simple_lookup"
        result["transform_strategy"] = "clarify"

        if not result.get("rewritten_query"):
            result["rewritten_query"] = query

        if "rewritten_queries" not in result:
            result["rewritten_queries"] = [result["rewritten_query"]]

        extracted = result.get("extracted_params", {})
        models = extracted.get("models", []) or extracted.get("model", [])
        if isinstance(models, str):
            models = [models] if models else []
        result["target_models"] = models or []
        result["required_fields"] = extracted.get("param_names", [])
        result["numerical_constraints"] = extracted.get("constraints", {})

        rewritten = RewrittenQuery(**result)

        logger.info(f"[Query Rewrite] clarify 完成: {rewritten.rewritten_queries}")
        return rewritten

    def _decompose_query(self, query: str) -> RewrittenQuery:
        """decompose 策略：适用于 compare 意图"""
        logger.info("[Query Rewrite] 策略: decompose (break into sub-queries)")

        prompt = self._build_decompose_prompt(query)
        response = self.llm.call(
            [
                {"role": "system", "content": self.DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        result = self._parse_json(response)
        result["original_query"] = query
        result["intent"] = "compare"
        result["transform_strategy"] = "decompose"

        if result.get("clarification_needed") is None:
            result["clarification_needed"] = ""

        if "sub_queries" in result:
            result["rewritten_queries"] = result.pop("sub_queries")

        extracted = result.get("extracted_params", {})
        result["target_models"] = (
            extracted.get("models") or extracted.get("model") or []
        )
        result["required_fields"] = extracted.get("dimensions", [])
        result["numerical_constraints"] = {}

        rewritten = RewrittenQuery(**result)

        logger.info(
            f"[Query Rewrite] decompose 完成: {len(rewritten.rewritten_queries)} sub-queries"
        )
        return rewritten

    def _generalize_query(self, query: str) -> RewrittenQuery:
        """generalize 策略：适用于 recommend 意图"""
        logger.info("[Query Rewrite] 策略: generalize (step back)")

        prompt = self._build_generalize_prompt(query)
        response = self.llm.call(
            [
                {"role": "system", "content": self.GENERALIZE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        result = self._parse_json(response)
        result["original_query"] = query
        result["intent"] = "recommend"
        result["transform_strategy"] = "generalize"

        if result.get("clarification_needed") is None:
            result["clarification_needed"] = ""

        if "generalized_query" in result:
            result["rewritten_queries"] = [result.pop("generalized_query")]

        extracted = result.get("extracted_params", {})
        original_specifics = result.get("original_specifics", {})
        result["target_models"] = []
        result["required_fields"] = original_specifics.get("implied_needs", [])

        raw_constraints = extracted.get("constraints", {})
        constraints = {}
        for k, v in raw_constraints.items():
            if isinstance(v, list):
                constraints[k] = ", ".join(str(x) for x in v)
            else:
                constraints[k] = str(v)
        result["numerical_constraints"] = constraints

        rewritten = RewrittenQuery(**result)

        logger.info(f"[Query Rewrite] generalize 完成: {rewritten.rewritten_queries}")
        return rewritten

    # ── Prompt 构建辅助 ───────────────────────────────────

    def _build_clarify_prompt(self, query: str) -> str:
        parts = [f"原始查询: {query}"]
        parts.append("请澄清此查询，使其清晰且可检索。")
        return "\n".join(parts)

    def _build_decompose_prompt(self, query: str) -> str:
        parts = [f"比较/推理查询: {query}"]
        parts.append("请将此查询分解为多个可并行检索的子查询。")
        return "\n".join(parts)

    def _build_generalize_prompt(self, query: str) -> str:
        parts = [f"推荐查询: {query}"]
        parts.append("请泛化此查询，从产品类别角度找到最佳选项。")
        return "\n".join(parts)


# ── 全局实例 ──────────────────────────────────────────

_query_rewrite_service: Optional[QueryRewriteService] = None


def get_query_rewrite_service() -> QueryRewriteService:
    """获取查询重写服务单例"""
    global _query_rewrite_service
    if _query_rewrite_service is None:
        _query_rewrite_service = QueryRewriteService()
    return _query_rewrite_service
