# core/query_engineer/query_ir.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Constraint:
    """查询约束条件"""
    type: str        # "numeric" | "region" | "category"
    field: str       # "weight" | "compute" | "region" | ...
    op: str          # ">" | "<" | ">=" | "<=" | "=" | "contains"
    value: Any
    unit: Optional[str] = None

    def matches(self, item_value: Any) -> bool:
        """判断给定的值是否满足此约束（用于 filter 操作）"""
        if self.type == "numeric":
            try:
                v = float(item_value)
                threshold = float(self.value)
                if self.op == ">":
                    return v > threshold
                elif self.op == "<":
                    return v < threshold
                elif self.op == ">=":
                    return v >= threshold
                elif self.op == "<=":
                    return v <= threshold
                elif self.op == "=":
                    return v == threshold
            except (ValueError, TypeError):
                return False
        elif self.type == "region":
            return str(item_value) in self.value if isinstance(self.value, list) else str(item_value) == str(self.value)
        return True

@dataclass
class QueryIR:
    """查询中间表示 — QueryPlanner 的输出格式"""
    original_query: str
    intent: str                  # "lookup" | "analysis" | "recommendation"
    target: str                  # 查询主体：产品名 / 合作单位 / ...

    # 约束和操作
    constraints: List[Constraint] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)  # "filter" | "expand" | "aggregate" | "foreach"

    # 实体扩展（need_expand=True 时使用）
    expand_map: Dict[str, List[str]] = field(default_factory=dict)

    # 语义标志
    need_split: bool = False      # 是否需要拆分子查询
    need_aggregate: bool = False  # 是否需要聚合
    need_foreach: bool = False    # 是否需要遍历实体

    # 维度（analysis 类查询使用）
    dimensions: List[str] = field(default_factory=list)

    # 已识别的实体
    entities: List[str] = field(default_factory=list)

    # 需要返回的字段
    required_fields: List[str] = field(default_factory=list)

    def has_constraints(self) -> bool:
        return len(self.constraints) > 0

    def needs_filter(self) -> bool:
        return "filter" in self.operations

    def needs_expand(self) -> bool:
        return "expand" in self.operations

    def needs_aggregate(self) -> bool:
        return "aggregate" in self.operations

    def needs_foreach(self) -> bool:
        return "foreach" in self.operations