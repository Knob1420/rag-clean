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

    使用统一的 _transform_query 进行改写。
    """

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




# ── 全局实例 ──────────────────────────────────────────

_query_rewrite_service: Optional[QueryRewriteService] = None


def get_query_rewrite_service() -> QueryRewriteService:
    """获取查询重写服务单例"""
    global _query_rewrite_service
    if _query_rewrite_service is None:
        _query_rewrite_service = QueryRewriteService()
    return _query_rewrite_service
