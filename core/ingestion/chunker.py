"""
智能分块

基于 models.py 定义的数据结构：
- Document: 统一文档对象（主块 content = parent 内容 + children）
- ChildDocument: 子块，用于精准检索

分块流程：
1. clean_text() 清洗全文（调用 cleaner.py）
2. MarkdownHeaderTextSplitter 按标题切分 parent sections
3. RecursiveCharacterTextSplitter 将每个 parent 切分为 child chunks
4. 每个 parent 生成一个 Document，其 content = parent 内容，children 为切分出的 ChildDocument
5. 返回 List[Document]（供存储层消费）

ID 格式：{doc_id}_p{parent_idx} / {doc_id}_c{child_idx}

"""

import hashlib
import re
from typing import List

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from core.ingestion.cleaner import clean_text
from core.model.models import (
    ChildDocument,
    Document,
)
from config import settings

# ── 配置 ──────────────────────────────────────────────────────────────────────

HEADERS_TO_SPLIT_ON = [("#", "H1"), ("##", "H2"), ("###", "H3")]

# parent chunk 大小（按字符数）
MIN_PARENT_SIZE = 1000
MAX_PARENT_SIZE = 2000

# child chunk 大小（按字符数）
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100

# 表格匹配正则
_TABLE_PATTERN = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
_MD_TABLE_PATTERN = re.compile(
    r"(?:^\|.*\|$)\n(?:^\|[\s\-:|\s]+\|$)\n(?:(?:^\|.*\|$)\n?)+",
    re.MULTILINE,
)


def _html_table_to_text(html: str) -> str:
    """将 HTML 表格转为 Markdown 格式"""
    import re as _re

    # 提取表头
    headers = _re.findall(r"<th[^>]*>(.*?)</th>", html, _re.DOTALL | _re.IGNORECASE)
    headers = [_re.sub(r"<[^>]+>", "", h).strip() for h in headers]

    # 提取所有行（包括表头行）
    rows = _re.findall(r"<tr[^>]*>(.*?)</tr>", html, _re.DOTALL | _re.IGNORECASE)

    md_lines = []

    # 第一行作为表头
    if headers:
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # 处理后续数据行
    for row in rows[1:] if len(rows) > 1 else []:
        cells = _re.findall(r"<td[^>]*>(.*?)</td>", row, _re.DOTALL | _re.IGNORECASE)
        cells = [_re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if cells:
            md_lines.append("| " + " | ".join(cells) + " |")

    # 如果没有提取到有效内容，返回原始 HTML
    if len(md_lines) <= 2:
        return html

    return "\n".join(md_lines)


# ── 工具函数 ──────────────────────────────────────────────────────────────────


def _generate_doc_hash(text: str) -> str:
    """生成文档内容的 MD5 哈希"""
    return hashlib.md5(text.encode()).hexdigest()[:16]


# ── 分块器 ─────────────────────────────────────────────────────────────────────


class SmartChunker:
    """
    父子双层分块器

    支持两种分块模式：
    - recursive（默认）: MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
    - semantic: 使用 embedding 相似度找语义边界切分

    - parent: 按 Markdown 标题切分（1000-2000 字）
    - child:  每个 parent 用 RecursiveCharacterTextSplitter 切成 ~500 字，索引向量库
    - 每个 parent 对应一个 Document，Document.content = parent 内容，Document.children 为切分出的 ChildDocument
    - Document.metadata["parent_id"] = parent_id（同 doc_id，但标注这是 parent）
    """

    def __init__(self, remove_images: bool = True, dataset_id: str = ""):
        """
        初始化分块器

        Args:
            remove_images: 清洗时是否移除图片，默认 True（与 cleaner.py 行为一致）
            dataset_id: 数据集 ID，用于判断是否触发产品参数表格删除（仅产品类知识库生效）
        """
        self._remove_images = remove_images
        self._dataset_id = dataset_id

        self._parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
        )

    def chunk(
        self,
        markdown: str,
        title: str,
        doc_id: str,
        mode: str = "recursive",
    ) -> List[Document]:
        """
        将 Markdown 内容切分为 Document 列表（父子结构）。

        Args:
            markdown: 原始 Markdown 文本
            doc_id: 文档 ID
            mode: 分块模式，"recursive"（默认）或 "semantic"
                  - recursive: 使用 MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
                  - semantic: 使用 embedding 相似度找语义边界

        Returns:
            List[Document]: 每个 parent 对应一个 Document，包含切分后的 children
        """
        # ── 1. 清洗全文 ────────────────────────────────────────────────────────
        md = clean_text(markdown, remove_images=self._remove_images, dataset_id=self._dataset_id)

        if mode == "semantic":
            return self._chunk_semantic(md, title, doc_id)

        # ── 2. 按 Markdown 标题切分 parent sections ──────────────────────────
        raw_sections = self._parent_splitter.split_text(md)
        for sec in raw_sections:
            sec.page_content = clean_text(
                sec.page_content, remove_images=self._remove_images, dataset_id=self._dataset_id
            )

        # ── 2b. 丢弃目录、索引、版本修订记录、免责声明等无用 section ──────────
        raw_sections = self._drop_meta_sections(raw_sections)

        # ── 3. parent 合并/拆分/清理 ──────────────────────────────────────────
        merged = self._merge_small_parents(raw_sections)
        split = self._split_large_parents(merged)
        cleaned = self._clean_small_chunks(split)

        # ── 4. 生成 Document 列表（每个 parent 一个 Document）──────────────────
        documents: List[Document] = []
        child_global_idx = 0

        for p_idx, p_doc in enumerate(cleaned):
            parent_id = f"{doc_id}_p{p_idx}"

            # 给 p_doc.metadata 添加 doc_title（child 需要用到）
            p_doc.metadata["doc_title"] = title

            # ── 4a. 切 child chunks（表格不被拆分）────────────────────────────
            child_document_list, content_for_parent = self._split_children_with_tables(
                p_doc.page_content, p_doc.metadata, doc_id, parent_id, child_global_idx
            )
            if child_document_list:
                child_global_idx += len(child_document_list)

            # ── 4b. 创建 Document（主块 content = parent 内容 + children）───────
            document = Document(
                content=content_for_parent,
                metadata={
                    "doc_id": doc_id,
                    "doc_title": title,
                    "chunk_id": parent_id,
                    "doc_hash": _generate_doc_hash(content_for_parent),
                },
                children=child_document_list if child_document_list else None,
            )
            documents.append(document)

        from loguru import logger

        logger.info(
            f"[Chunker] 分块完成: {len(documents)} documents "
            f"(children={child_global_idx})"
        )

        return documents

    # ── child chunk 切分（表格不被拆分）────────────────────────────────────

    def _split_children_with_tables(
        self,
        content: str,
        parent_metadata: dict,
        doc_id: str,
        parent_id: str,
        start_child_idx: int,
    ) -> tuple[List[ChildDocument], str]:
        """
        切分 child chunks，确保表格内容不被拆分。

        Returns:
            (child_document_list, content_with_converted_tables):
            - child_document_list: ChildDocument 列表
            - content_with_converted_tables: HTML 表格已转为 Markdown 的 content

        策略：
        1. 先将 HTML 表格转为 Markdown 格式
        2. 找到所有 Markdown 表格（支持单元格内换行）
        3. 表格作为整体，作为一个 child chunk
        4. 非表格内容用 RecursiveCharacterTextSplitter 切分
        """
        # 先将 HTML 表格转为 Markdown 格式
        content = self._convert_tables_to_markdown(content)

        # 收集所有表格的位置和内容（支持单元格内换行）
        tables = self._find_markdown_tables(content)

        child_document_list: List[ChildDocument] = []
        child_idx = start_child_idx

        if not tables:
            # 没有表格，直接用 RecursiveCharacterTextSplitter 切分
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
                    },
                )
                child_document_list.append(child_document)
                child_idx += 1
            # 合并过短 chunks
            child_document_list = self._merge_short_children(child_document_list)
            return child_document_list, content

        # 按位置排序表格
        tables.sort(key=lambda x: x[0])

        # 切分：表格作为整体，非表格内容单独切分
        # 追溯最近的 section header 合并到表格
        prev_end = 0

        for start, end, table_text in tables:
            # 在当前表格前的范围内追溯 section header
            section_header = (
                self._find_last_section_header(
                    content[prev_end:start], len(content[prev_end:start])
                )
                if start > prev_end
                else ""
            )
            if section_header:
                table_text = section_header + "\n\n" + table_text

            # 处理表格前的非表格内容（不含 section header）
            before_text = content[prev_end:start].strip()
            if before_text and section_header:
                # 去掉 section header 部分
                header_pos = before_text.rfind(section_header)
                if header_pos >= 0:
                    before_text = before_text[:header_pos].rstrip()

            if before_text:
                # 短 caption（如 "表7随机振动试验条件"）直接拼入 table_text，不单独生成 chunk
                if len(before_text) < self.MIN_CHILD_CHUNK_SIZE and not section_header:
                    table_text = before_text + "\n\n" + table_text
                else:
                    before_doc = LCDocument(
                        page_content=before_text, metadata=parent_metadata
                    )
                    child_docs = self._child_splitter.split_documents([before_doc])
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
                            },
                        )
                        child_document_list.append(child_document)
                        child_idx += 1

            # 表格作为整体（不被切分）
            if table_text:
                child_id = f"{doc_id}_c{child_idx}"
                child_document = ChildDocument(
                    content=table_text,
                    metadata={
                        "doc_id": doc_id,
                        "doc_title": parent_metadata.get("doc_title", ""),
                        "chunk_id": child_id,
                        "doc_hash": _generate_doc_hash(table_text),
                        "parent_id": parent_id,
                    },
                )
                child_document_list.append(child_document)
                child_idx += 1

            prev_end = end

        # 处理最后一个表格之后的内容
        after_text = content[prev_end:].strip()
        if after_text:
            after_doc = LCDocument(page_content=after_text, metadata=parent_metadata)
            child_docs = self._child_splitter.split_documents([after_doc])
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
                    },
                )
                child_document_list.append(child_document)
                child_idx += 1

        # 合并过短 chunks
        child_document_list = self._merge_short_children(child_document_list)
        return child_document_list, content

    # ── parent 合并/拆分/清理 ───────────────────────────────────────────────

    def _merge_small_parents(self, sections: list) -> list:
        """
        将连续的小 section（< MIN_PARENT_SIZE）合并，
        直到合并后达到 MIN_PARENT_SIZE。
        """
        if not sections:
            return []

        merged, current = [], None

        for sec in sections:
            if current is None:
                current = sec
            else:
                current.page_content += "\n\n" + sec.page_content
                self._merge_metadata(current, sec)

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

    def _split_large_parents(self, sections: list) -> list:
        """将超过 MAX_PARENT_SIZE 的 section 拆分"""
        result = []
        for sec in sections:
            if len(sec.page_content) <= MAX_PARENT_SIZE:
                result.append(sec)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=MAX_PARENT_SIZE,
                    chunk_overlap=CHILD_CHUNK_OVERLAP,
                )
                result.extend(splitter.split_documents([sec]))
        return result

    def _clean_small_chunks(self, sections: list) -> list:
        """清理合并后仍 < MIN_PARENT_SIZE 的残余小块，合并到前一个"""
        cleaned = []

        for i, sec in enumerate(sections):
            if len(sec.page_content) < MIN_PARENT_SIZE:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + sec.page_content
                    self._merge_metadata(cleaned[-1], sec)
                elif i < len(sections) - 1:
                    sections[i + 1].page_content = (
                        sec.page_content + "\n\n" + sections[i + 1].page_content
                    )
                    self._merge_metadata(sections[i + 1], sec)
                else:
                    cleaned.append(sec)
            else:
                cleaned.append(sec)

        return cleaned

    # ── 表格提取 ─────────────────────────────────────────────────────────

    def _extract_spec_sections(
        self,
        sections: list,
    ) -> tuple[list, list]:
        """
        识别包含结构化表格的 section，将表格提取为独立 section。

        Returns:
            (spec_sections, table_free_sections)
            - spec_sections: 表格 LCDocument 列表
            - table_free_sections: 去除表格后的 section 列表
        """
        spec_sections: list = []
        table_free_sections: list = []

        for sec_doc in sections:
            sec_text = sec_doc.page_content
            if not sec_text:
                continue

            table_text, other_text = self._separate_table_content(sec_text)

            if table_text:
                spec_sections.append(
                    LCDocument(page_content=table_text, metadata=sec_doc.metadata)
                )

                if other_text and len(other_text.strip()) > 50:
                    sec_doc.page_content = other_text
                    table_free_sections.append(sec_doc)
            else:
                table_free_sections.append(sec_doc)

        return spec_sections, table_free_sections

    def _separate_table_content(self, text: str) -> tuple[str, str]:
        """将文本中的表格部分和非表格部分分离"""
        html_match = _TABLE_PATTERN.search(text)
        if html_match:
            html_table = html_match.group(0)
            # 将 HTML 表格转为 Markdown 格式
            md_table = _html_table_to_text(html_table)
            return md_table, _TABLE_PATTERN.sub("", text).strip()

        md_match = _MD_TABLE_PATTERN.search(text)
        if md_match:
            return md_match.group(0).strip(), _MD_TABLE_PATTERN.sub("", text).strip()

        return "", text

    # ── 辅助方法 ─────────────────────────────────────────────────────────

    def _find_markdown_tables(self, text: str) -> List[tuple[int, int, str]]:
        """
        查找所有 Markdown 表格，返回 (start, end, table_text) 列表。
        支持单元格内换行的表格。
        """
        tables: List[tuple[int, int, str]] = []
        lines = text.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # 检查是否可能是表格的第一行
            if line.startswith("|") and line.endswith("|") and line.count("|") >= 2:
                table_lines = [line]
                table_start_line = i

                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()

                    # 如果不是以 | 开头，说明表格结束
                    if not next_line.startswith("|"):
                        # 检查是否是延续行（不以 | 开头但可能是多行单元格内容）
                        if next_line:
                            # 非空行且不以 | 开头，可能是上一行的延续
                            if table_lines and not table_lines[-1].rstrip().endswith(
                                "|"
                            ):
                                table_lines[-1] += "\n" + next_line
                                j += 1
                                continue
                        break

                    # 检查是否是分隔行（|---|格式）
                    if re.match(r"^\|[\s\-:|]+\|$", next_line):
                        table_lines.append(next_line)
                        j += 1
                        # 继续读取数据行
                        while j < len(lines):
                            data_line = lines[j].strip()
                            if data_line.startswith("|"):
                                table_lines.append(data_line)
                                j += 1
                            else:
                                break
                        break
                    else:
                        table_lines.append(next_line)
                        j += 1

                table_text = "\n".join(table_lines)
                # 计算在原始文本中的字符位置
                char_start = sum(len(lines[k]) + 1 for k in range(table_start_line))
                char_end = char_start + len(table_text)
                tables.append((char_start, char_end, table_text))
                i = j
            else:
                i += 1

        return tables

    def _find_last_section_header(self, text: str, before_pos: int) -> str:
        """找到 text 中 before_pos 之前的最后一个 Markdown section header"""
        lines = text[:before_pos].split("\n")
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if re.match(r"^#{1,6}\s+", line):
                return line
        return ""

    def _convert_tables_to_markdown(self, text: str) -> str:
        """将文本中所有 HTML 表格转为 Markdown 格式"""
        # 递归替换所有 HTML 表格，防止死循环（如果转换后仍是原始 HTML 则不再重复匹配）
        max_iterations = 100
        for _ in range(max_iterations):
            match = _TABLE_PATTERN.search(text)
            if not match:
                break
            html_table = match.group(0)
            md_table = _html_table_to_text(html_table)
            # 如果转换后内容没变，说明无法转换，不再重复处理
            if md_table == html_table:
                text = _TABLE_PATTERN.sub("", text, count=1)
            else:
                text = text.replace(html_table, md_table, 1)
        return text

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

    # ── 丢弃无用 section ───────────────────────────────────────────

    # 需丢弃的 section header 关键词（大小写不敏感）
    _META_HEADER_KEYWORDS: list[str] = [
        "目录", "索引", "table of contents", "index",
        "版本修订记录", "修订记录", "变更记录", "changelog", "版本历史",
        "免责声明", "版权声明", "声明", "注意事项", "重要声明",
        " preface", "preface", "foreword",
        "acknowledgement", "acknowledgments", "致谢",
        "abbreviation", "缩写", "glossary", "词汇表",
        "参考文献", "reference", "references",
    ]

    def _drop_meta_sections(
        self, sections: list
    ) -> list:
        """
        丢弃目录、索引、版本修订记录、免责声明等文档元信息 section。

        判断逻辑：section 的 header 包含 _META_HEADER_KEYWORDS 中的任意关键词。
        """
        import re

        def is_meta_section(sec) -> bool:
            # MarkdownHeaderTextSplitter 的 metadata 用 H1/H2/H3 作为 key
            header = (
                sec.metadata.get("H1")
                or sec.metadata.get("H2")
                or sec.metadata.get("H3")
                or ""
            )
            if not header:
                return False
            header_lower = header.lower()
            for kw in self._META_HEADER_KEYWORDS:
                if kw.lower() in header_lower:
                    return True
            return False

        before = len(sections)
        filtered = [s for s in sections if not is_meta_section(s)]
        dropped = before - len(filtered)

        if dropped > 0:
            from loguru import logger
            logger.info(f"[Chunker] 丢弃 {dropped} 个元信息 section: 目录/索引/修订记录等")

        return filtered

    # ── 短块合并 ──────────────────────────────────────────────────

    MIN_CHILD_CHUNK_SIZE = 50  # 子块最小长度阈值

    def _merge_short_children(
        self, children: List[ChildDocument]
    ) -> List[ChildDocument]:
        """
        合并过短的 child chunks（< MIN_CHILD_CHUNK_SIZE）。

        策略：
        - 如果有前一个 chunk，合并到前一个
        - 否则合并到后一个
        - 如果是中间块且前后都有，优先合并到前一个
        - 表格块（以 | 开头）不参与合并
        """
        if not children:
            return children

        result: List[ChildDocument] = []
        skip_next = False

        for i, child in enumerate(children):
            if skip_next:
                skip_next = False
                continue

            # 表格块不参与合并
            if child.content.strip().startswith("|"):
                result.append(child)
                continue

            # 检查是否过短
            if len(child.content) < self.MIN_CHILD_CHUNK_SIZE:
                if result:
                    # 合并到前一个
                    prev = result[-1]
                    prev.content = prev.content.rstrip() + "\n\n" + child.content
                    prev.metadata["doc_hash"] = _generate_doc_hash(prev.content)
                    # 如果不是最后一个，且后面有非表格块，尝试继续合并
                    if i < len(children) - 1:
                        next_child = children[i + 1]
                        if not next_child.content.strip().startswith("|"):
                            # 把下一个也合并进来
                            next_content = next_child.content.lstrip()
                            prev.content = prev.content.rstrip() + "\n\n" + next_content
                            prev.metadata["doc_hash"] = _generate_doc_hash(prev.content)
                            skip_next = True
                elif i < len(children) - 1:
                    # 没有前一个，合并到后一个
                    next_child = children[i + 1]
                    if not next_child.content.strip().startswith("|"):
                        next_child.content = child.content.rstrip() + "\n\n" + next_child.content.lstrip()
                        next_child.metadata["doc_hash"] = _generate_doc_hash(next_child.content)
                        skip_next = True
                        result.append(next_child)
                    else:
                        result.append(child)
                else:
                    # 最后一个且没有前一个，保留
                    result.append(child)
            else:
                result.append(child)

        return result

    # ── Semantic Chunking ──────────────────────────────────────────────────

    def _chunk_semantic(
        self,
        md: str,
        title: str,
        doc_id: str,
    ) -> List[Document]:
        """
        基于 embedding 相似度的语义分块（支持中文）。

        流程：
        1. 按中英文句子结束符预分割
        2. 将句子组合成 ~3 句一组的片段，计算 embedding
        3. 计算相邻片段的 cosine similarity
        4. 相似度低于阈值的断点切分
        5. 每个语义 chunk 创建一个 Document
        6. 每个 Document 内部用 _split_children_with_tables 切 child chunks

        Returns:
            List[Document]: 每个语义 chunk 对应一个 Document
        """
        from core.client.embedder import encode_batch

        # ── 1. 按中英文句子结束符预分割 ───────────────────────────────────
        sentences = self._split_sentences_by_punctuation(md)

        # ── 2. 将句子组合成 ~3 句一组的片段 ───────────────────────────────
        chunk_size = 3
        segments = []
        for i in range(0, len(sentences), chunk_size):
            seg = " ".join(sentences[i : i + chunk_size])
            if seg.strip():
                segments.append(seg)

        if not segments:
            return []

        # ── 3. 计算每个 segment 的 embedding ──────────────────────────────
        embeddings = encode_batch(segments)
        valid_embeddings = [(i, emb) for i, emb in enumerate(embeddings) if emb is not None]

        if len(valid_embeddings) < 2:
            # 无法分块，直接返回一个 Document
            return self._create_single_document(md, title, doc_id, 0, len(segments))

        # ── 4. 计算相邻 segment 的相似度，找语义断点 ─────────────────────
        SIMILARITY_THRESHOLD = 0.7

        breakpoints = [0]
        for idx in range(len(valid_embeddings) - 1):
            _, emb1 = valid_embeddings[idx]
            _, emb2 = valid_embeddings[idx + 1]

            # cosine similarity
            dot = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = sum(a * a for a in emb1) ** 0.5
            norm2 = sum(b * b for b in emb2) ** 0.5
            sim = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

            if sim < SIMILARITY_THRESHOLD:
                breakpoints.append(valid_embeddings[idx + 1][0])

        breakpoints.append(len(segments))

        # ── 5. 构建 Document 列表 ─────────────────────────────────────────
        documents: List[Document] = []
        child_global_idx = 0

        for chunk_idx in range(len(breakpoints) - 1):
            start_seg_idx = breakpoints[chunk_idx]
            end_seg_idx = breakpoints[chunk_idx + 1]

            # 收集属于这个 semantic chunk 的所有 sentences
            start_sent_idx = start_seg_idx * chunk_size
            end_sent_idx = min(end_seg_idx * chunk_size, len(sentences))
            chunk_sentences = sentences[start_sent_idx:end_sent_idx]
            chunk_text = " ".join(chunk_sentences).strip()

            if not chunk_text:
                continue

            parent_id = f"{doc_id}_p{chunk_idx}"

            child_document_list, content_for_parent = self._split_children_with_tables(
                chunk_text,
                {"doc_title": title},
                doc_id,
                parent_id,
                child_global_idx,
            )
            if child_document_list:
                child_global_idx += len(child_document_list)

            document = Document(
                content=content_for_parent,
                metadata={
                    "doc_id": doc_id,
                    "doc_title": title,
                    "chunk_id": parent_id,
                    "doc_hash": _generate_doc_hash(content_for_parent),
                },
                children=child_document_list if child_document_list else None,
            )
            documents.append(document)

        from loguru import logger

        logger.info(
            f"[Chunker] 语义分块完成: {len(documents)} documents "
            f"(children={child_global_idx})"
        )

        return documents

    def _split_sentences_by_punctuation(self, text: str) -> List[str]:
        """按中英文句子结束符分割句子"""
        import re

        # 按中英文句号、问号、感叹号、英文句点分割
        pattern = r"(?<=[。！？.!?])\s+"
        sentences = re.split(pattern, text)

        # 过滤空句子
        return [s.strip() for s in sentences if s.strip()]

    def _create_single_document(
        self,
        md: str,
        title: str,
        doc_id: str,
        parent_idx: int,
        child_global_idx: int,
    ) -> List[Document]:
        """当无法分块时，创建单个 Document"""
        parent_id = f"{doc_id}_p{parent_idx}"

        child_document_list, content_for_parent = self._split_children_with_tables(
            md,
            {"doc_title": title},
            doc_id,
            parent_id,
            child_global_idx,
        )

        document = Document(
            content=content_for_parent,
            metadata={
                "doc_id": doc_id,
                "doc_title": title,
                "chunk_id": parent_id,
                "doc_hash": _generate_doc_hash(content_for_parent),
            },
            children=child_document_list if child_document_list else None,
        )

        from loguru import logger

        logger.info(
            f"[Chunker] 语义分块完成: 1 document "
            f"(children={len(child_document_list) if child_document_list else 0})"
        )

        return [document]
