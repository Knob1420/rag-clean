"""
意图路由数据模型
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RoutingResult:
    """路由结果"""

    intent: str  # 意图类型: simple_lookup | compare | recommend | aggregate
    confidence: float  # 置信度 [0, 1]
    original_query: str  # 原始查询
    category: str = ""  # 类别（可选）
    entities: List[str] = field(default_factory=list)  # 提取的实体列表
    constraints: Dict[str, any] = field(default_factory=dict)  # 约束条件

    @property
    def is_high_confidence(self) -> bool:
        """是否是高置信度路由"""
        return self.confidence >= 0.7


# ── 意图类型常量 ──────────────────────────────────

INTENT_SIMPLE_LOOKUP = "simple_lookup"
INTENT_COMPARE = "compare"
INTENT_RECOMMEND = "recommend"
INTENT_AGGREGATE = "aggregate"

ALL_INTENTS = [INTENT_SIMPLE_LOOKUP, INTENT_COMPARE, INTENT_RECOMMEND, INTENT_AGGREGATE]
