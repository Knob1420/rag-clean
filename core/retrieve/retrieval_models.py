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
    section_title: Optional[str] = None
    score: float = 0.0

    # 文档级字段（扁平存储）
    doc_type: Optional[str] = None
    domain: Optional[str] = None
    filter_terms: List[str] = Field(default_factory=list)

    # chunk 级字段
    chunk_type: Optional[str] = None
    spec_table: Optional[Dict[str, Any]] = None
    spec_rows: Optional[List[Any]] = None  # ES 中存为 list of dict

    # Enrichment 字段
    entities_text: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    context_summary: Optional[str] = None

    # 父子导航
    parent_id: Optional[str] = None

    # 高亮
    highlight: Dict[str, List[str]] = Field(default_factory=dict)


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

    # Query Rewrite 结果字段
    target_models: Optional[List[str]] = Field(
        None, description="目标实体列表（用于检索加权）"
    )
    keywords: Optional[List[str]] = Field(None, description="关键词列表")
    chunk_types: Optional[List[str]] = Field(
        None, description="chunk类型列表，用于BM25加权覆盖多个类型"
    )

    # Enrichment 过滤 (暂保留，未使用)
    # chunk_types: Optional[List[str]] = Field(None, description="按chunk类型筛选")

    # Rerank 选项
    use_rerank: bool = Field(True, description="是否使用 Rerank")
    rerank_top_k: Optional[int] = Field(
        None, ge=1, le=100, description="Rerank 后保留数量"
    )


class HighlightOptions(BaseModel):
    """高亮选项"""

    pre_tags: List[str] = Field(["<em>"], description="高亮前缀标签")
    post_tags: List[str] = Field(["</em>"], description="高亮后缀标签")
    fragment_size: int = Field(150, ge=50, le=500, description="摘要片段大小")
    number_of_fragments: int = Field(3, ge=1, le=10, description="返回片段数量")


# ========== RAG 问答模型 ==========


class SourceInfo(BaseModel):
    """来源信息"""

    chunk_id: str
    doc_id: str
    doc_name: Optional[str] = None
    section_title: Optional[str] = None
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
    top_k: int = Field(5, ge=1, le=50, description="召回数量")
    use_rewrite: bool = Field(True, description="是否使用Query Rewrite")
    use_rerank: bool = Field(True, description="是否使用Rerank")
    rerank_top_k: Optional[int] = Field(
        None, ge=1, le=50, description="Rerank后保留数量"
    )
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="最低相关性评分")


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
    top_k: int = Field(5, ge=1, le=50, description="召回数量")
    use_rewrite: bool = Field(True, description="是否使用Query Rewrite")
    use_rerank: bool = Field(True, description="是否使用Rerank")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="最低相关性评分")


class SearchResponse(BaseModel):
    """检索响应"""

    query: str
    total: int
    chunks: List[Dict[str, Any]]
    timing: Dict[str, float] = Field(default_factory=dict)
    rewrite: Optional[Dict[str, Any]] = Field(None, description="Query Rewrite 结果")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str
    services: Dict[str, str]
