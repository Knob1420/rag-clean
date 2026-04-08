"""
智能分块 — 父子双层分块 + 表格特殊处理

核心逻辑：
- 父子双层结构：parent 提供完整上下文，child 提供精准检索
- 小 chunk 合并（_merge_small_parents）、大 chunk 拆分（_split_large_parents）、
  残余小块清理（_clean_small_chunks）
- 所有文档统一流程

存储策略：
- Parent chunk → 本地 JSON 文件（data/parents/{parent_id}.json），不入 ES
- Child chunk  → ES 索引，用于检索
- Spec chunk   → ES 索引，用于结构化数据检索
- 检索命中 child 时，通过 parent_id 定位本地文件加载完整上下文

ID 格式：{doc_id}_{类型缩写}_{序号}
  - parent: {doc_id}_p0, {doc_id}_p1, ...
  - child:  {doc_id}_c0, {doc_id}_c1, ...
  - spec:   {doc_id}_s0, {doc_id}_s1, ...
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from models import Chunk, DocumentAnalysis, TableSpec
from config import settings

# ── helpers ────────────────────────────────────────────


def _table_to_content(table: "TableSpec") -> str:
    """将 TableSpec 转为易读文本，供 BM25 检索和 embedding 使用"""
    parts = []
    if table.fields:
        parts.append(", ".join(f"{k}={v}" for k, v in table.fields.items()))
    for row in table.rows:
        parts.append(", ".join(f"{k}={v}" for k, v in row.items()))
    return "\n".join(parts)


# ── 配置 ──────────────────────────────────────────────

HEADERS_TO_SPLIT_ON = [("#", "H1"), ("##", "H2"), ("###", "H3")]

CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 500
MAX_PARENT_SIZE = 1000

# 匹配 Markdown 图片和 HTML <table>
_IMG_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img\s[^>]*/?>", re.IGNORECASE)
_TABLE_PATTERN = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
# Markdown 管道表格
_MD_TABLE_PATTERN = re.compile(
    r"(?:^\|.*\|$)\n(?:^\|[\s\-:|]+\|$)\n(?:(?:^\|.*\|$)\n?)+",
    re.MULTILINE,
)


# ── 文本清洗 ──────────────────────────────────────────


def _remove_images(text: str) -> str:
    return _IMG_PATTERN.sub("", text)


def _clean_text(text: str) -> str:
    """去除图片 + 清理多余空行"""
    text = _remove_images(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── 分块器 ────────────────────────────────────────────


class SmartChunker:
    """
    父子双层分块器 + 表格特殊处理

    Parent chunk (500-1000字): 完整章节上下文
    Child chunk  (~300字): 精准检索单元
    """

    def __init__(self):
        self._parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
        )

    def chunk(
        self, markdown: str, doc_id: str, analysis: DocumentAnalysis
    ) -> List[Chunk]:
        """
        将 Markdown 内容切分为带父子关系的 chunks。

        流程：按标题切分 → 合并小块 → 拆分大块 → 清理残余小块
              → 生成 parent/child → parent 存本地 → child+spec 入 ES

        返回值只包含 child + spec chunks（用于 ES 索引）。
        Parent chunks 保存到本地文件（data/parents/{parent_id}.json）。
        """
        # 1. 清洗全文
        md = _clean_text(markdown)

        # 2. 按标题切分
        raw_sections = self._parent_splitter.split_text(md)
        for sec in raw_sections:
            sec.page_content = _clean_text(sec.page_content)

        # 3. 分离 spec table sections（不走 merge/split/clean）
        spec_chunks, table_free_sections = self._extract_spec_chunks(
            raw_sections, doc_id, analysis
        )

        # 4. 合并小块 → 拆分大块 → 清理残余小块
        merged = self._merge_small_parents(table_free_sections)
        split = self._split_large_parents(merged)
        cleaned = self._clean_small_chunks(split)

        # 5. 生成 parent + child chunks
        retrievable_chunks: List[Chunk] = []  # 只有 child + spec，入 ES
        child_global_idx = 0
        parent_dir = Path(settings.parent_store_dir)
        parent_dir.mkdir(parents=True, exist_ok=True)

        for i, p_doc in enumerate(cleaned):
            parent_id = f"{doc_id}_p{i}"
            section_title = self._extract_section_title(p_doc.metadata)

            # 6. 生成 child chunks（用 child splitter 作用在 Document 上，保留 metadata）
            child_docs = self._child_splitter.split_documents([p_doc])
            children_ids: List[str] = []
            for c_doc in child_docs:
                child_content = c_doc.page_content.strip()
                if not child_content:
                    continue
                child_id = f"{doc_id}_c{child_global_idx}"
                child_section = self._infer_child_title(child_content, section_title)
                child_chunk = Chunk(
                    chunk_id=child_id,
                    doc_id=doc_id,
                    content=child_content,
                    section_title=child_section,
                    parent_id=parent_id,
                )
                retrievable_chunks.append(child_chunk)
                children_ids.append(child_id)
                child_global_idx += 1

            # 7. Parent chunk 存本地 JSON，不入 ES
            parent_data = {
                "chunk_id": parent_id,
                "doc_id": doc_id,
                "content": p_doc.page_content,
                "section_title": section_title,
                "children_ids": children_ids,
            }
            parent_file = parent_dir / f"{parent_id}.json"
            parent_file.write_text(json.dumps(parent_data, ensure_ascii=False, indent=2))

        # 8. 加入 spec chunks
        for s_idx, spec in enumerate(spec_chunks):
            spec.chunk_id = f"{doc_id}_s{s_idx}"
            retrievable_chunks.append(spec)

        parent_count = len(cleaned)
        logger.info(
            f"[Chunker] 分块完成: {len(retrievable_chunks)} retrievable chunks "
            f"(parents_saved={parent_count}, children={child_global_idx}, specs={len(spec_chunks)})"
        )
        return retrievable_chunks

    # ── 核心：merge / split / clean ───────────────────────

    def _merge_small_parents(self, chunks):
        """
        将连续的小块（< MIN_PARENT_SIZE）合并，
        直到合并后达到 MIN_PARENT_SIZE。
        """
        if not chunks:
            return []

        merged, current = [], None

        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                self._merge_metadata(current, chunk)

            if len(current.page_content) >= MIN_PARENT_SIZE:
                merged.append(current)
                current = None

        # 处理剩余
        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                self._merge_metadata(merged[-1], current)
            else:
                merged.append(current)

        return merged

    def _split_large_parents(self, chunks):
        """将超过 MAX_PARENT_SIZE 的 Document 拆分（使用 split_documents 保留 metadata）"""
        result = []
        for chunk in chunks:
            if len(chunk.page_content) <= MAX_PARENT_SIZE:
                result.append(chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=MAX_PARENT_SIZE,
                    chunk_overlap=CHILD_CHUNK_OVERLAP,
                )
                result.extend(splitter.split_documents([chunk]))
        return result

    def _clean_small_chunks(self, chunks):
        """清理合并后仍然小于 MIN_PARENT_SIZE 的残余小块，合并到前一个 chunk"""
        cleaned = []

        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < MIN_PARENT_SIZE:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    self._merge_metadata(cleaned[-1], chunk)
                elif i < len(chunks) - 1:
                    # 合并到下一个
                    chunks[i + 1].page_content = (
                        chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    )
                    self._merge_metadata(chunks[i + 1], chunk)
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)

        return cleaned

    # ── 表格 chunk 处理 ──────────────────────────────────

    def _extract_spec_chunks(
        self, sections, doc_id: str, analysis: DocumentAnalysis
    ) -> Tuple[List[Chunk], list]:
        """
        识别包含结构化表格的 section，将表格提取为独立 chunk。
        chunk_type 默认为 other，enrichment 阶段由 LLM 按实际内容决定。
        返回 (table_chunks, table_free_sections)
        """
        table_chunks: List[Chunk] = []
        table_free_sections = []

        for sec_doc in sections:
            sec_text = sec_doc.page_content
            if not sec_text:
                continue

            matched_table = self._find_table_for_section(sec_text, analysis.tables)

            if matched_table:
                table_text, other_text = self._separate_table_content(sec_text)
                if table_text:
                    # chunk_type 由 enrichment LLM 按实际内容决定，这里用默认值
                    table_chunks.append(
                        Chunk(
                            chunk_id="",  # ID later assigned in chunk()
                            doc_id=doc_id,
                            content=_table_to_content(matched_table),
                            chunk_type="other",  # enrichment 阶段由 LLM 覆盖
                            section_title=self._extract_section_title(sec_doc.metadata),
                            spec_table=(
                                matched_table.fields if matched_table.fields else None
                            ),
                            spec_rows=(
                                matched_table.rows if matched_table.rows else None
                            ),
                        )
                    )
                if other_text and len(other_text.strip()) > 50:
                    sec_doc.page_content = other_text
                    table_free_sections.append(sec_doc)
            else:
                table_free_sections.append(sec_doc)

        return table_chunks, table_free_sections

    # ── 辅助方法 ──────────────────────────────────────────

    def _find_table_for_section(
        self, section_text: str, tables: List[TableSpec]
    ) -> Optional[TableSpec]:
        """查找 section 对应的表格（同时匹配 fields 和 rows）"""
        for table in tables:
            all_values = list(table.fields.values())
            for row in table.rows:
                all_values.extend(row.values())

            if not all_values:
                continue

            matched = sum(1 for v in all_values if v in section_text)
            if matched >= len(all_values) * 0.5:
                return table
        return None

    def _separate_table_content(self, text: str) -> Tuple[str, str]:
        """将文本中的表格部分和非表格部分分离"""
        html_match = _TABLE_PATTERN.search(text)
        if html_match:
            return html_match.group(0), _TABLE_PATTERN.sub("", text).strip()

        md_match = _MD_TABLE_PATTERN.search(text)
        if md_match:
            return md_match.group(0).strip(), _MD_TABLE_PATTERN.sub("", text).strip()

        return "", text

    @staticmethod
    def _extract_section_title(metadata: dict) -> Optional[str]:
        for key in ("H3", "H2", "H1"):
            if key in metadata:
                return metadata[key]
        return None

    @staticmethod
    def _infer_child_title(content: str, parent_title: Optional[str]) -> Optional[str]:
        if not content:
            return parent_title
        first_line = content.split("\n", 1)[0].strip()
        m = re.match(r"^(#{1,6})\s+(.+)$", first_line)
        if m:
            return m.group(2).strip()
        return parent_title

    @staticmethod
    def _merge_metadata(target, source):
        """将 source.metadata 合并到 target.metadata"""
        if not hasattr(source, "metadata"):
            return
        for k, v in source.metadata.items():
            if k in target.metadata:
                target.metadata[k] = f"{target.metadata[k]} -> {v}"
            else:
                target.metadata[k] = v
