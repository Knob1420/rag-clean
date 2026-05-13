"""
检索相关 Pydantic 模型

基于 rag-knowledge-base schemas/document.py，适配 rag-clean 扁平字段结构：
- 移除 user_roles / access_roles
- 移除 page_number / page_end / chunk_index / chunk_role / parent_content / metadata
- 新增 doc_type / domain / filter_terms / spec_table / spec_rows
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class RetrievedChunk(BaseModel):
    """检索到的分块"""

    chunk_id: str
    doc_id: str
    content: str
    score: float = 0.0

    # 文档级字段
    doc_title: Optional[str] = None
    dataset_id: Optional[str] = None

    # chunk 级字段
    chunk_type: Optional[str] = None  # "parent" | "child" | "summary"
    doc_hash: Optional[str] = None

    # 父子导航
    parent_id: Optional[str] = None  # child 有，parent 无


class RetrievalResult(BaseModel):
    """检索结果"""

    query: str
    total: int
    chunks: List[RetrievedChunk]
    timing: Dict[str, float] = Field(default_factory=dict)


class RetrievalOptions(BaseModel):
    """检索选项"""

    top_k: int = Field(20, ge=1, le=100, description="返回结果数量")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="最小相关性评分")
    doc_ids: Optional[List[str]] = Field(None, description="按文档ID列表筛选")
    dataset_ids: Optional[List[str]] = Field(None, description="按数据集ID列表筛选")

    # RRF 融合权重（仅 hybrid 模式生效）
    vector_weight: Optional[float] = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="向量检索权重（RRF 融合），默认 0.7；BM25 权重 = 1 - vector_weight",
    )

    # HyDE 选项
    use_hyde: bool = Field(
        False, description="是否使用 HyDE（假设性文档嵌入）提升向量检索"
    )
    hyde_num_hypotheses: int = Field(
        1, ge=1, le=3, description="HyDE 生成假设性文档数量"
    )

    # Rerank 选项
    use_rerank: bool = Field(True, description="是否使用 Rerank")
    rerank_top_k: Optional[int] = Field(
        None, ge=1, le=100, description="Rerank 后保留数量"
    )


# ========== RAG 问答模型 ==========


class SourceInfo(BaseModel):
    """来源信息"""

    chunk_id: str
    doc_id: str
    doc_name: Optional[str] = None
    score: float
    snippet: str


class TokenUsage(BaseModel):
    """Token使用统计"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatRequest(BaseModel):
    """RAG问答请求"""

    query: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(25, ge=1, le=50, description="召回数量")
    use_rewrite: bool = Field(True, description="是否使用Query Rewrite")
    use_rerank: bool = Field(True, description="是否使用Rerank")
    rerank_top_k: Optional[int] = Field(
        None, ge=1, le=50, description="Rerank后保留数量"
    )
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="最低相关性评分")
    mode: str = Field(
        "quick",
        pattern="^(quick|agent)$",
        description="问答模式：quick=快速问答，agent=智能推理",
    )
    use_hyde: bool = Field(False, description="是否使用 HyDE")
    max_iterations: int = Field(
        10, ge=1, le=20, description="Agent 最大迭代数（仅 agent 模式）"
    )


class ChatResponse(BaseModel):
    """RAG问答响应"""

    answer: str
    sources: List[SourceInfo]
    time: dict
    usage: TokenUsage
    chunks_count: int


class SearchRequest(BaseModel):
    """检索请求"""

    query: str = Field(..., min_length=1, description="检索查询")
    top_k: int = Field(25, ge=1, le=50, description="召回数量")
    use_hyde: bool = Field(False, description="是否使用 HyDE")
    use_understand: bool = Field(False, description="(已弃用) 是否使用 Query Understand")
    use_rewrite: bool = Field(False, description="(已弃用) 是否使用 Query Rewrite")
    use_rerank: bool = Field(True, description="是否使用 Rerank")
    rerank_top_k: Optional[int] = Field(
        None, ge=1, le=50, description="Rerank后保留数量"
    )
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="最低相关性评分")


class SearchResponse(BaseModel):
    """检索响应"""

    query: str
    total: int
    chunks: List[Dict[str, Any]]
    timing: Dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str
    services: Dict[str, str]
