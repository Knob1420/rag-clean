"""
步骤1增强 — 智能分块（表格优先提取版）

扩展 core/ingestion/chunker.py：
1. 替换 clean_text → clean_and_normalize（术语归一）
2. 每个分片携带 table_type / doc_type / source_file / source_weight 元数据
3. 表格优先提取：清洗后先提取所有 HTML 表格，表格单独成 Document（HTML 格式），
   其余内容正常按标题切分
"""

import hashlib
import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from core.ingestion.chunker import (
    HEADERS_TO_SPLIT_ON,
    MIN_PARENT_SIZE,
    MAX_PARENT_SIZE,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    _html_table_to_text,
    _generate_doc_hash,
    SmartChunker as BaseSmartChunker,
)
from core.model.models import (
    ChildDocument,
    Document,
)

from tools.preprocessing.cleaner_ext import clean_and_normalize, TextCleaner

# ── 配置 ──────────────────────────────────────────────────────

# HTML 表格匹配正则（从父类 chunker.py 复用）
_TABLE_PATTERN = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)

# 文档类型识别（从文件名/路径推断）
_DOC_TYPE_KEYWORDS = {
    "技术文档": ["技术要求", "技术规范", "技术指标", "试验规范", "试验大纲"],
    "汇报材料": ["汇报", "报告", "工作进展", "情况报告", "请示", "建议"],
    "方案": ["整体方案", "实施方案", "设计方案", "建设方案"],
    "规范": ["规范", "标准", "要求", "准则"],
    "PPT资料": ["ppt", "pptx", "幻灯片", "演示"],
}


def _infer_doc_type(file_path: str) -> str:
    """从文件路径推断文档类型"""
    path_lower = file_path.lower()
    for doc_type, keywords in _DOC_TYPE_KEYWORDS.items():
        if any(kw.lower() in path_lower for kw in keywords):
            return doc_type
    return "其他"


def _detect_table_type_from_html(html: str) -> Optional[str]:
    """
    从 HTML 表格内容中检测表格类型。

    先把 HTML 表格转成 MD 表格，再用 TextCleaner.detect_table_type 判断。
    """
    # HTML → MD
    md_table = _html_table_to_text(html)
    if not md_table.strip():
        return None
    return TextCleaner.detect_table_type(md_table)


# ── 分块器 ────────────────────────────────────────────────────


class SmartChunker(BaseSmartChunker):
    """
    步骤1增强版智能分块器。

    继承 core/ingestion/chunker.py 的 SmartChunker，新增：
    - clean_and_normalize（术语归一）替代 clean_text
    - table_type / doc_type / source_file / source_weight 元数据
    - 表格优先提取：清洗后先提取 HTML 表格，表格单独成 Document（HTML 格式）
    """

    def __init__(
        self,
        remove_images: bool = True,
        dataset_id: str = "",
        source_file: str = "",
        doc_type: Optional[str] = None,
        source_weight: float = 1.0,
    ):
        """
        初始化增强版分块器

        Args:
            remove_images: 是否移除图片（继承父类行为）
            dataset_id: 数据集 ID（继承父类行为）
            source_file: 来源文件名（如"三体计算星座整体方案20251031.md"）
            doc_type: 文档类型（如"技术文档"/"汇报材料"），None 则自动推断
            source_weight: 信源权重（0.0-1.0），默认 1.0
        """
        super().__init__(remove_images=remove_images, dataset_id=dataset_id)
        self._source_file = source_file
        self._doc_type = doc_type or (
            "" if not source_file else _infer_doc_type(source_file)
        )
        self._source_weight = source_weight

    def chunk(
        self,
        markdown: str,
        title: str,
        doc_id: str,
        mode: str = "recursive",
    ) -> List[Document]:
        """
        将 Markdown 内容切分为 Document 列表（父子结构）。

        流程（相比父类SmartChunker变化）：
        1. clean_and_normalize 清洗全文（术语归一）
        2. 【新增】提取所有 HTML 表格 → 单独成 Document（HTML 格式）
        3. 其余内容 → MarkdownHeaderTextSplitter 按标题切分 parent sections
        4. parent 合并/拆分/清理
        5. 生成 Document 列表

        Args:
            markdown: 原始 Markdown 文本
            title: 文档标题
            doc_id: 文档 ID
            mode: 分块模式，"recursive"（默认）或 "semantic"

        Returns:
            List[Document]: 每个 parent 对应一个 Document，包含切分后的 children
        """
        # ── 1. 清洗全文（术语归一）───────────────────────────────────────────
        md = clean_and_normalize(
            markdown, remove_images=self._remove_images, dataset_id=self._dataset_id
        )

        # ── 2. 【核心变化】提取 HTML 表格，单独成 Document ─────────────────
        table_documents, md_without_tables = self._extract_tables_as_documents(
            md, doc_id, title
        )

        # if mode == "semantic":
        #     return self._chunk_semantic(md, title, doc_id)

        # ── 3. 其余内容按 Markdown 标题切分 parent sections ───────────────
        raw_sections = self._parent_splitter.split_text(md_without_tables)

        # ── 3b. 丢弃目录、索引、版本修订记录等无用 section ─────────────────
        raw_sections = self._drop_meta_sections(raw_sections)

        # ── 4. parent 合并/拆分/清理 ───────────────────────────────────────
        merged = self._merge_small_parents(raw_sections)
        split = self._split_large_parents(merged)
        cleaned = self._clean_small_chunks(split)

        # ── 5. 生成 Document 列表 ─────────────────────────────────────────
        documents: List[Document] = list(table_documents)  # 表格 Document 排在前面
        child_global_idx = 0

        for p_idx, p_doc in enumerate(cleaned):
            parent_id = f"{doc_id}_p{p_idx}"

            # 提取表格类型（如果有表格的话）
            table_type = self._detect_table_type_from_content(p_doc.page_content)

            # ── 5a. 切 child chunks ───────────────────────────────────────
            child_document_list = self._split_children(
                p_doc.page_content,
                p_doc.metadata,
                doc_id,
                parent_id,
                child_global_idx,
                table_type,
            )
            if child_document_list:
                child_global_idx += len(child_document_list)

            # 构建 parent 元数据
            parent_metadata = {
                "doc_id": doc_id,
                "doc_title": title,
                "chunk_id": parent_id,
                "doc_hash": _generate_doc_hash(p_doc.page_content),
                "doc_type": self._doc_type,
                "source_file": self._source_file,
                "source_weight": self._source_weight,
                "table_type": table_type,
            }

            document = Document(
                content=p_doc.page_content,
                metadata=parent_metadata,
                children=child_document_list if child_document_list else None,
            )
            documents.append(document)

        from loguru import logger

        logger.info(
            f"[ChunkerExt] 分块完成: {len(documents)} documents "
            f"(tables={len(table_documents)}, children={child_global_idx})"
        )

        return documents

    def _extract_tables_as_documents(
        self,
        md: str,
        doc_id: str,
        title: str,
    ) -> tuple[List[Document], str]:
        """
        从清洗后的 MD 中提取所有 HTML 表格，每个表格单独成 Document（HTML 格式）。
        同时返回去掉表格后的 MD 内容。

        Args:
            md: 清洗后的 Markdown 文本（可能含 HTML 表格）
            doc_id: 文档 ID
            title: 文档标题

        Returns:
            (table_documents, md_without_tables)
            - table_documents: 每个 HTML 表格对应一个 Document（standalone parent，HTML 格式 content）
            - md_without_tables: 去掉 HTML 表格后的 Markdown 文本
        """
        table_documents: List[Document] = []
        table_idx = 0
        last_end = 0
        parts: List[str] = []

        # 遍历所有 HTML 表格
        for match in _TABLE_PATTERN.finditer(md):
            # 这个表格之前的内容 → 保留到 parts
            parts.append(md[last_end : match.start()])

            html_table = match.group(0)
            table_type = _detect_table_type_from_html(html_table)

            # 为这个表格创建独立的 Document（MD 格式）
            table_parent_id = f"{doc_id}_t{table_idx}"
            table_child_id = f"{doc_id}_tc{table_idx}"  # tc = table child

            # HTML → MD 转换
            md_table = _html_table_to_text(html_table)

            # 构建 child（standalone 表格 chunk，MD 格式）
            child_doc = ChildDocument(
                content=md_table,
                metadata={
                    "doc_id": doc_id,
                    "doc_title": title,
                    "chunk_id": table_child_id,
                    "doc_hash": _generate_doc_hash(md_table),
                    "parent_id": table_parent_id,
                    "doc_type": self._doc_type,
                    "source_file": self._source_file,
                    "source_weight": self._source_weight,
                    "table_type": table_type,
                },
            )

            # 构建 parent Document（整个 Document 就是这个表格）
            parent_metadata = {
                "doc_id": doc_id,
                "doc_title": title,
                "chunk_id": table_parent_id,
                "doc_hash": _generate_doc_hash(md_table),
                "doc_type": self._doc_type,
                "source_file": self._source_file,
                "source_weight": self._source_weight,
                "table_type": table_type,
            }

            document = Document(
                content=md_table,  # parent content = MD 表格
                metadata=parent_metadata,
                children=[child_doc],
            )
            table_documents.append(document)
            table_idx += 1
            last_end = match.end()

        # 最后一个表格之后的内容
        parts.append(md[last_end:])
        md_without_tables = "".join(parts)

        return table_documents, md_without_tables

    def _detect_table_type_from_content(self, content: str) -> Optional[str]:
        """
        从 regular parent content 中检测 MD 表格类型。
        这里的 parent 已经不包含 HTML 表格（已在上游提取为独立 Document）。
        """
        md_table = self._extract_md_table(content)
        if md_table:
            return TextCleaner.detect_table_type(md_table)
        return None

    def _extract_md_table(self, content: str) -> Optional[str]:
        """从 content 中提取第一个 MD 表格"""
        lines = content.split("\n")
        in_table = False
        table_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|"):
                in_table = True
                table_lines.append(stripped)
            elif in_table:
                # 遇到非表格行，停止
                break

        if len(table_lines) >= 2:
            return "\n".join(table_lines)
        return None

    def _split_children(
        self,
        content: str,
        parent_metadata: dict,
        doc_id: str,
        parent_id: str,
        start_child_idx: int,
        table_type: Optional[str] = None,
    ) -> List[ChildDocument]:
        """
        将 parent content 切分为 child chunks。

        表格已在上游提取为独立 Document，这里的 parent 不含 HTML 表格，
        直接用 RecursiveCharacterTextSplitter 切分。
        """
        child_document_list: List[ChildDocument] = []
        child_idx = start_child_idx

        parent_doc = LCDocument(page_content=content, metadata=parent_metadata)
        child_docs = self._child_splitter.split_documents([parent_doc])
        for c_doc in child_docs:
            child_content = c_doc.page_content.strip()
            if not child_content:
                continue
            child_id = f"{doc_id}_c{child_idx}"
            child_document = ChildDocument(
                content=child_content,
                metadata={
                    "doc_id": doc_id,
                    "doc_title": parent_metadata.get("doc_title", ""),
                    "chunk_id": child_id,
                    "doc_hash": _generate_doc_hash(child_content),
                    "parent_id": parent_id,
                    "doc_type": self._doc_type,
                    "source_file": self._source_file,
                    "source_weight": self._source_weight,
                    "table_type": None,
                },
            )
            child_document_list.append(child_document)
            child_idx += 1

        # 合并过短 chunks
        child_document_list = self._merge_short_children(child_document_list)

        return child_document_list
