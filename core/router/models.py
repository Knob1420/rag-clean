"""
意图路由数据模型

仅保留 RoutingResult 供 legacy eval 脚本使用。
Intent 常量已迁移到 prompt.py。
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RoutingResult:
    """路由结果"""

    intent: str
    confidence: float
    original_query: str
    category: str = ""
    entities: List[str] = field(default_factory=list)
    constraints: Dict[str, any] = field(default_factory=dict)

    @property
    def is_high_confidence(self) -> bool:
        """是否是高置信度路由"""
        return self.confidence >= 0.7
