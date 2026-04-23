"""
Intent Router 模块

提供基于 embedding 相似度的意图分类和路由。
"""

from core.router.models import (
    RoutingResult,
    INTENT_SIMPLE_LOOKUP,
    INTENT_COMPARE,
    INTENT_RECOMMEND,
    INTENT_AGGREGATE,
    ALL_INTENTS,
)
from core.router.semantic_router import SemanticRouter, get_semantic_router
from core.router.intent_prototypes import INTENT_PROTOTYPES

__all__ = [
    "RoutingResult",
    "SemanticRouter",
    "get_semantic_router",
    "INTENT_PROTOTYPES",
    "INTENT_SIMPLE_LOOKUP",
    "INTENT_COMPARE",
    "INTENT_RECOMMEND",
    "INTENT_AGGREGATE",
    "ALL_INTENTS",
]
