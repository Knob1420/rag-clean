"""
数据模型 — rag-clean 数据层核心数据结构

基于 Dify RAG 设计，统一文档对象模型体系：

分层结构：
- Document: 统一文档对象（主块 + 子块 + 附件）
- ChildDocument: 子块，用于精准检索
- AttachmentDocument: 附件（图片等）

其他模型：
- TableSpec: 表格结构化参数
- DocumentAnalysis: 文档分析结果
- ProcessedDocument: 处理后的完整文档
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# 统一 Document 对象体系
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ChildDocument:
    """
    子块

    用于精准检索的最小单元。
    父子模式下：Child 索引到向量库，Parent 提供完整上下文。
    """

    content: str
    primary_entity: str = ""  # 核心实体（从 parent summary 继承）
    metadata: Dict[str, str] = field(
        default_factory=dict
    )  # doc_id, doc_hash, parent_id 等
    vector: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChildDocument":
        return cls(
            content=d["content"],
            primary_entity=d.get("primary_entity", ""),
            metadata=d.get("metadata", {}),
            vector=d.get("vector"),
        )


@dataclass
class SummaryDocument:
    """
    Summary 块

    存储 parent chunk 的摘要和核心实体。
    单独作为 child chunk 保存，关联到对应的 parent_id。
    """

    content: str  # summary 文本
    primary_entity: str = ""  # 核心实体
    metadata: Dict[str, str] = field(
        default_factory=dict
    )  # doc_id, doc_title, parent_id, chunk_id 等
    vector: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SummaryDocument":
        return cls(
            content=d["content"],
            primary_entity=d.get("primary_entity", ""),
            metadata=d.get("metadata", {}),
            vector=d.get("vector"),
        )


@dataclass
class AttachmentDocument:
    """
    附件 — 图片等信息

    存储文档中的图片等附件。
    """

    content: str  # 通常是文件名
    metadata: Dict[str, str] = field(default_factory=dict)  # doc_id, doc_type 等
    vector: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AttachmentDocument":
        return cls(
            content=d["content"],
            metadata=d.get("metadata", {}),
            vector=d.get("vector"),
        )


@dataclass
class Document:
    """
    统一文档对象

    支持父子双层结构：
    - content: 主块内容
    - children: 子块列表（可选）
    - summaries: summary 块列表（可选，每个 parent 对应一个 summary）
    - attachments: 附件列表（可选）

    三层 ID 体系：
    - dataset_id: 知识库/数据集 ID
    - document_id: 原始文档 ID（一次上传 = 一个 document_id）
    - doc_id: 分块 ID（每个 chunk 独立）
    - parent_id: 父块 ID（仅父子模式）

    Parent chunk 额外字段：
    - summary: 摘要文本
    - primary_entity: 核心实体
    """

    content: str
    summary: str = ""  # 摘要文本
    primary_entity: str = ""  # 核心实体
    metadata: Dict[str, str] = field(
        default_factory=dict
    )  # dataset_id, document_id, doc_id, parent_id, doc_hash
    vector: Optional[List[float]] = None
    children: Optional[List[ChildDocument]] = None  # 父子块
    summaries: Optional[List[SummaryDocument]] = None  # summary 块
    attachments: Optional[List[AttachmentDocument]] = None  # 附件

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(
            content=d["content"],
            summary=d.get("summary", ""),
            primary_entity=d.get("primary_entity", ""),
            metadata=d.get("metadata", {}),
            vector=d.get("vector"),
            children=(
                [ChildDocument.from_dict(c) for c in d.get("children", [])]
                if d.get("children")
                else None
            ),
            summaries=(
                [SummaryDocument.from_dict(s) for s in d.get("summaries", [])]
                if d.get("summaries")
                else None
            ),
            attachments=(
                [AttachmentDocument.from_dict(a) for a in d.get("attachments", [])]
                if d.get("attachments")
                else None
            ),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 现有模型（保留，兼容已有代码）
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TableSpec:
    """从表格提取的结构化参数"""

    source_table: str  # 原始表格标题（如"技术参数"）
    fields: Dict[str, str]  # 两列 KV 表: {"重量": "1.29kg", "功耗": "64W"}
    rows: List[Dict[str, str]] = field(
        default_factory=list
    )  # 多列表: [{"参数": "算力", "G1": "768TOPS"}, ...]
    position: int = 0  # 表格在文档中的字符偏移位置
