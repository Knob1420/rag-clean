"""
Query Understanding — 卫星三体计算星座项目专用

前置 LLM 解析层：
1. 多问句自动拆分（query_list）
2. 意图分类（lookup/compare/recommend/aggregate）

输出结构化 JSON，代码直接解析，无需后处理。
"""

from dataclasses import dataclass, field
from typing import List

from loguru import logger

from core.generation.llm import get_llm_client, parse_json_response
from prompt import QUERY_UNDERSTANDING_SYSTEM_PROMPT, QUERY_UNDERSTANDING_USER_PROMPT


@dataclass
class SubQuery:
    """单个子问句解析结果"""
    query: str
    intent: str  # lookup | compare | recommend | aggregate


@dataclass
class QueryUnderstandingResult:
    """Query Understanding 解析结果"""
    original_query: str
    sub_queries: List[SubQuery] = field(default_factory=list)
    generation_constraints: List[str] = field(default_factory=list)  # 顶层汇总的生成约束

    @classmethod
    def from_llm_response(cls, original_query: str, data: dict) -> "QueryUnderstandingResult":
        """从 LLM 返回的 JSON dict 构建结果"""
        query_list = data.get("query_list", [original_query])
        intent_list = data.get("intent_list", ["lookup"])
        constraints_list = data.get("generation_constraints", [])

        sub_queries = []
        for i, q in enumerate(query_list):
            intent = intent_list[i] if i < len(intent_list) else "lookup"
            sub_queries.append(SubQuery(query=q, intent=intent))

        # generation_constraints 取顶层
        if not constraints_list:
            # fallback：从 prompt 指令中提取（如"翻译成英文"在 query 末尾）
            constraints_list = cls._extract_constraints_from_query(original_query)

        return cls(
            original_query=original_query,
            sub_queries=sub_queries,
            generation_constraints=constraints_list,
        )

    @staticmethod
    def _extract_constraints_from_query(query: str) -> List[str]:
        """从原始 query 末尾提取生成约束（翻译、不超过X字等）"""
        import re

        constraints = []

        # 翻译类
        if re.search(r"翻译成英文", query):
            constraints.append("翻译成英文")
        elif re.search(r"翻译成中文", query):
            constraints.append("翻译成中文")

        # 字数限制
        m = re.search(r"不超过\s*(\d+)\s*字", query)
        if m:
            constraints.append(f"不超过{m.group(1)}字")

        return constraints

    def __repr__(self):
        return (
            f"QueryUnderstandingResult(original={self.original_query[:30]!r}..., "
            f"sub_queries={len(self.sub_queries)})"
        )


class QueryUnderstandingService:
    """Query Understanding 服务（卫星三体计算星座项目专用）"""

    VALID_INTENTS = {"simple_lookup", "compare", "recommend", "aggregate"}

    def parse(self, query: str) -> QueryUnderstandingResult:
        """
        解析用户输入，拆分问句、分类意图。

        Args:
            query: 用户原始问句

        Returns:
            QueryUnderstandingResult，包含所有子问句及其意图
        """
        if not query or not query.strip():
            return QueryUnderstandingResult(
                original_query=query,
                sub_queries=[SubQuery(query=query, intent="simple_lookup")]
            )

        user_prompt = QUERY_UNDERSTANDING_USER_PROMPT.format(query=query)
        messages = [
            {"role": "system", "content": QUERY_UNDERSTANDING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            llm = get_llm_client()
            response = llm.call(messages, temperature=0.3)
            data = parse_json_response(response)

            result = QueryUnderstandingResult.from_llm_response(query, data)

            # 校验 intent 合法性
            for sq in result.sub_queries:
                if sq.intent not in self.VALID_INTENTS:
                    logger.warning(f"无效 intent [{sq.intent}]，修正为 simple_lookup")
                    sq.intent = "simple_lookup"

            logger.info(
                f"[QueryUnderstanding] 拆分={len(result.sub_queries)} 条，"
                f"意图={[sq.intent for sq in result.sub_queries]}"
            )
            return result

        except Exception as e:
            logger.warning(f"Query Understanding 失败，降级为单句 simple_lookup: {e}")
            return QueryUnderstandingResult(
                original_query=query,
                sub_queries=[SubQuery(query=query.strip(), intent="simple_lookup")]
            )


# ── 全局实例 ──────────────────────────────────────────

_query_understanding_service = None


def get_query_understanding_service() -> QueryUnderstandingService:
    global _query_understanding_service
    if _query_understanding_service is None:
        _query_understanding_service = QueryUnderstandingService()
    return _query_understanding_service
