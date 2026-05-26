"""
Intent Router 模块（已精简）

Intent 常量已迁移到 prompt.py。
仅保留 RoutingResult 数据模型供 legacy eval 脚本使用。
"""

from core.router.models import RoutingResult

__all__ = [
    "RoutingResult",
]
