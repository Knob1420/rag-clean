"""
Pipeline 模块

集成 retrieval、generation 等服务
"""

from core.pipeline.simple_pipeline import (
    SimplePipeline,
    SimplePipelineResult,
)

__all__ = [
    "SimplePipeline",
    "SimplePipelineResult",
]
