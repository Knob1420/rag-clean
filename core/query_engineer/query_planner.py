"""
QueryPlanner — 将自然语言 Query 转换为 QueryIR

设计原则：
1. 单次 LLM 调用获取 intent / target / entities / required_fields
2. 后置规则推导 constraints / operations / expand_map / dimensions
3. 不依赖现有 Understand / Rewrite 结果，独立工作
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger

from core.generation.llm import get_llm_client, parse_json_response
from core.query_engineer.query_ir import QueryIR, Constraint

# ── Prompt Templates ────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """你是一个智能查询规划器。

你的任务是将用户问题转换为结构化的查询执行计划。

输出格式（严格 JSON）：
{
    "intent": "lookup | analysis | recommendation",
    "target": "查询主体（产品名/合作单位/概念）",
    "entities": ["识别出的实体列表"],
    "required_fields": ["需要返回的字段"],
    "reasoning": "你的规划推理过程"
}

意图说明：
- lookup：查定义、查参数、查属性
- analysis：分析原因、分析结构、按维度归纳
- recommendation：推荐、选择、对比

约束识别规则：
- "不超过Xkg"、"小于X" → 数值上界
- "大于X"、"超过X" → 数值下界
- "上海"、"江苏"、"长三角" → 地域限制
- "包括"、"包含" → 需展开的父类实体
- "按X分类"、"分析X" → dimensions

直接输出 JSON，不要有其他内容。"""

PLANNER_USER_PROMPT = """用户问题：{query}

请输出 JSON 格式的查询规划。"""

# ── 规则配置（用于后置推导）───────────────────────────────────────────────

REGION_KEYWORDS = [
    "上海", "江苏", "浙江", "安徽", "合肥",
    "长三角", "环渤海", "珠三角", "西部", "东部",
    "北京", "广州", "深圳", "成都", "西安",
]

NUMERIC_PATTERNS = [
    (r"不超过\s*(\d+\.?\d*)\s*(kg|KG|千克)", "weight", "<=", "numeric"),
    (r"小于\s*(\d+\.?\d*)\s*(kg|KG|千克)", "weight", "<", "numeric"),
    (r"(\d+\.?\d*)\s*(kg|KG|千克)\s*以内", "weight", "<=", "numeric"),
    (r"大于\s*(\d+\.?\d*)\s*(TFlops|TF|TOPS)", "compute", ">", "numeric"),
    (r"超过\s*(\d+\.?\d*)\s*(TFlops|TF|TOPS)", "compute", ">", "numeric"),
    (r"重量\s*[<<=]\s*(\d+\.?\d*)", "weight", "<=", "numeric"),
    (r"算力\s*[>>=]\s*(\d+\.?\d*)", "compute", ">=", "numeric"),
    (r"功耗\s*[<<=]\s*(\d+\.?\d*)\s*W", "power", "<=", "numeric"),
]

UNIT_MAP = {
    "kg": "kg", "KG": "kg", "千克": "kg",
    "TFlops": "TFlops", "TF": "TFlops", "TOPS": "TFlops",
    "W": "W", "瓦": "W",
}

FIELD_ALIAS = {
    "weight": ["重量", "质量", "mass", "Weight"],
    "compute": ["算力", "计算力", "计算性能", "Compute"],
    "power": ["功耗", "功率", "Power", "power"],
    "size": ["尺寸", "大小", "Size"],
    "interface": ["接口", "接口类型", "Interface"],
    "region": ["地区", "地域", "区域", "地点"],
}

# ── Category Expansion Map ───────────────────────────────────────────────────

_CATEGORY_KEYWORDS = {
    "太空计算组件": ["智算机", "激光通信机", "星载路由器"],
    "太空计算产品": ["智算机", "激光通信机", "星载路由器"],
    "产品": ["智算机", "激光通信机", "星载路由器"],
    "智算机系列": ["NX1", "G1", "G2", "G3", "智加G3"],
}

def normalize_field(text: str) -> Optional[str]:
    text_lower = text.lower()
    for standard, aliases in FIELD_ALIAS.items():
        if text_lower in aliases or text_lower == standard:
            return standard
    return None

# ── Temperature ─────────────────────────────────────────────────────────────

PLANNER_TEMPERATURE = 0.1

# ── QueryPlanner ────────────────────────────────────────────────────────────

class QueryPlanner:
    """
    查询规划器

    输入：自然语言 query
    输出：QueryIR（结构化查询规划）
    """

    def __init__(self):
        self.llm = get_llm_client()

    def plan(self, query: str) -> QueryIR:
        if not query or not query.strip():
            return QueryIR(
                original_query=query,
                intent="lookup",
                target="",
            )

        logger.info(f"[QueryPlanner] planning: {query}")

        # Step 1: LLM 调用
        try:
            llm_result = self._call_llm(query)
        except Exception as e:
            logger.warning(f"[QueryPlanner] _call_llm raised: {e}，降级为默认规划")
            llm_result = {"intent": "lookup", "target": query, "entities": [], "required_fields": []}

        # Step 2: 后置规则推导
        constraints = self._extract_constraints(query)
        operations = self._infer_operations(llm_result.get("intent", "lookup"), constraints)
        expand_map = self._infer_expand_map(query, llm_result.get("entities", []))
        dimensions = self._infer_dimensions(query, llm_result.get("intent", ""))

        ir = QueryIR(
            original_query=query,
            intent=llm_result.get("intent", "lookup"),
            target=llm_result.get("target", query),
            constraints=constraints,
            operations=operations,
            expand_map=expand_map,
            need_split=len(llm_result.get("entities", [])) > 1,
            need_aggregate="analysis" in llm_result.get("intent", ""),
            need_foreach=self._needs_foreach(query, llm_result),
            dimensions=dimensions,
            entities=llm_result.get("entities", []),
            required_fields=llm_result.get("required_fields", []),
        )

        logger.info(
            f"[QueryPlanner] done: intent={ir.intent}, target={ir.target}, "
            f"constraints={len(ir.constraints)}, operations={ir.operations}"
        )
        return ir

    def _call_llm(self, query: str) -> Dict[str, Any]:
        try:
            messages = [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": PLANNER_USER_PROMPT.format(query=query)},
            ]
            response = self.llm.call(messages, temperature=PLANNER_TEMPERATURE)
            result = parse_json_response(response)
            return result
        except Exception as e:
            logger.warning(f"[QueryPlanner] LLM call failed: {e}，降级为默认规划")
            return {"intent": "lookup", "target": query, "entities": [], "required_fields": []}

    def _extract_constraints(self, query: str) -> List[Constraint]:
        constraints = []

        # 地域约束
        found_regions = []
        for region in REGION_KEYWORDS:
            if region in query:
                found_regions.append(region)
        if found_regions:
            if "长三角" in found_regions:
                found_regions.remove("长三角")
                found_regions.extend(["上海", "江苏", "浙江", "安徽", "合肥"])
            constraints.append(Constraint(
                type="region",
                field="region",
                op="contains",
                value=list(set(found_regions)),
            ))

        # 数值约束
        for pattern, field, op, ctype in NUMERIC_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                value_str = match.group(1)
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str

                unit_match = re.search(r"(kg|KG|TFlops|TF|TOPS|W|千克|TF)", query[match.end():match.end()+10], re.IGNORECASE)
                unit = UNIT_MAP.get(unit_match.group(1).upper(), unit_match.group(1)) if unit_match else None

                standard_field = normalize_field(field) or field
                constraints.append(Constraint(
                    type=ctype,
                    field=standard_field,
                    op=op,
                    value=value,
                    unit=unit,
                ))

        return constraints

    def _infer_operations(self, intent: str, constraints: List[Constraint]) -> List[str]:
        operations = []
        if constraints:
            operations.append("filter")
        if intent == "analysis":
            operations.append("aggregate")
        if intent == "recommendation":
            operations.append("filter")
        return operations

    def _infer_expand_map(self, query: str, entities: List[str]) -> Dict[str, List[str]]:
        expand_map: Dict[str, List[str]] = {}
        for entity in entities:
            if entity in _CATEGORY_KEYWORDS:
                expand_map[entity] = _CATEGORY_KEYWORDS[entity]
        return expand_map

    def _infer_dimensions(self, query: str, intent: str) -> List[str]:
        dimensions = []
        if intent != "analysis":
            return dimensions
        dimension_keywords = [
            "合作形式", "合作单位", "成果", "效果",
            "型号", "产品类型", "重量", "算力", "功耗",
            "地区", "区域", "时间", "年份",
        ]
        for dim in dimension_keywords:
            if dim in query:
                dimensions.append(dim)
        return dimensions

    def _needs_foreach(self, query: str, llm_result: Dict[str, Any]) -> bool:
        entities = llm_result.get("entities", [])
        if len(entities) > 1:
            if any(kw in query for kw in ["分别", "各自", "每个", "逐个"]):
                return True
        return False