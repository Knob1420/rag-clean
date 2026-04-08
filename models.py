"""
数据模型 — rag-clean 数据层核心数据结构

- TableSpec: 从表格提取的结构化参数
- DocumentAnalysis: 文档级结构化分析结果（analyzer 输出，chunker 输入）
- Chunk: 一个文档分块
- ProcessedDocument: 处理后的完整文档
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class TableSpec:
    """从表格提取的结构化参数"""

    source_table: str  # 原始表格标题（如"技术参数"）
    fields: Dict[str, str]  # 两列 KV 表: {"重量": "1.29kg", "功耗": "64W"}
    rows: List[Dict[str, str]] = field(
        default_factory=list
    )  # 多列表: [{"参数": "算力", "G1": "768TOPS"}, ...]
    position: int = 0  # 表格在文档中的字符偏移位置


@dataclass
class DocumentAnalysis:
    """文档级结构化分析结果（analyzer 输出，chunker 输入）"""

    doc_type: str  # "产品手册" / "技术文档" / ... (ingestion 用：entity 模板 + chunk 策略)
    domain: str = ""  # 4 选 1: Product_Tech / Biz_Market / Ops_Mgmt / Corp_Support
    entities: Dict[str, str] = field(default_factory=dict)  # {"品牌": "智加", "型号": "G1"} (ingestion 上下文传递)
    filter_terms: List[str] = field(default_factory=list)  # 硬过滤专有名词: ["G1", "智加", "三体计算星座"]
    topics: List[str] = field(default_factory=list)  # 文档涉及的主题列表
    doc_intent: Optional[str] = None  # 文档意图
    summary: str = ""  # 文档摘要
    tables: List[TableSpec] = field(default_factory=list)  # 提取的所有表格（chunker 消费后生命周期结束）
    section_tree: List[Dict] = field(default_factory=list)  # 章节结构（chunker 消费后生命周期结束）

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DocumentAnalysis":
        tables = [TableSpec(**t) for t in d.get("tables", [])]
        return cls(
            doc_type=d["doc_type"],
            domain=d.get("domain", ""),
            entities=d.get("entities", {}),
            filter_terms=d.get("filter_terms", []),
            topics=d.get("topics", []),
            doc_intent=d.get("doc_intent"),
            summary=d.get("summary", ""),
            tables=tables,
            section_tree=d.get("section_tree", []),
        )


@dataclass
class Chunk:
    """一个文档分块"""

    chunk_id: str
    doc_id: str
    content: str
    chunk_type: str = "other"  # 7 种功能性标签: intro / spec_data / feature / procedure / faq / profile / other

    # 结构信息
    section_title: Optional[str] = None
    spec_table: Optional[Dict[str, str]] = None  # 两列 KV 表格结构化数据
    spec_rows: Optional[List[Dict[str, str]]] = None  # 多列表格行数据

    # 导航关系（父子层级，无 prev/next 链表）
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)  # parent 专用

    # 检索辅助
    keywords: List[str] = field(default_factory=list)
    context_summary: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            content=d["content"],
            chunk_type=d.get("chunk_type", "other"),
            section_title=d.get("section_title"),
            spec_table=d.get("spec_table"),
            spec_rows=d.get("spec_rows"),
            parent_id=d.get("parent_id"),
            children_ids=d.get("children_ids", []),
            keywords=d.get("keywords", []),
            context_summary=d.get("context_summary"),
        )


@dataclass
class ProcessedDocument:
    """处理后的完整文档"""

    doc_id: str
    title: str
    analysis: DocumentAnalysis
    chunks: List[Chunk]
    content: str  # 原始 markdown
