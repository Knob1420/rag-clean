"""
Pipeline 模块

集成 query_rewrite、retrieval、generation 等服务
"""

from core.pipeline.rag_pipeline import (
    RAGPipeline,
    PipelineResult,
)
from core.pipeline.simple_pipeline import (
    SimplePipeline,
    SimplePipelineResult,
)

__all__ = [
    "RAGPipeline",
    "PipelineResult",
    "SimplePipeline",
    "SimplePipelineResult",
]
