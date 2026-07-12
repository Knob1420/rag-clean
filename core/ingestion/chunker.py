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

from core.ingestion.cleaner import clean_text, is_low_quality_content
from core.model.models import (
    ChildDocument,
    Document,
)

# ── 配置 ──────────────────────────────────────────────────────────────────────

HEADERS_TO_SPLIT_ON = [("#", "H1"), ("##", "H2"), ("###", "H3")]

# parent chunk 大小（按字符数）
MIN_PARENT_SIZE = 1000
MAX_PARENT_SIZE = 2000

# child chunk 大小（按字符数）
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100

# 表格匹配正则（完整闭合表格）
_TABLE_PATTERN = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)


def _html_table_to_text(html: str) -> str:
    """
    将 HTML 表格转为 Markdown 格式（用 pandas，支持 rowspan/colspan）。

    pandas.read_html 自动展开合并单元格：
    - rowspan="N": 内容重复填充到下方 N 行
    - colspan="M": 内容重复填充到右侧 M 列

    解析失败时回退到简单的 <td>/<th> 正则提取（不展开合并单元格，但至少保留文本）。
    最终失败返回空字符串（让调用方过滤）。
    """
    import re as _re
    from io import StringIO

    # 优先用 pandas 解析（处理 rowspan/colspan）
    try:
        import pandas as pd
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_th = bool(_re.search(r"<th[^>]*>", html, _re.IGNORECASE))
            dfs = pd.read_html(StringIO(html), header=0 if has_th else None)
        if dfs:
            df = dfs[0].fillna("")
            # 删除全空列（colspan 展开后多出的列、纯空数据列）
            df = df.loc[:, (df.astype(str) != "").any()]
            if not df.empty:
                # 无 <th> 时，pandas 给的列名是 0,1,2,...，重命名为 col1/col2/...
                if not has_th:
                    df.columns = [f"col{i+1}" for i in range(len(df.columns))]
                md = df.to_markdown(index=False)
                # 内容检查：去 markdown 表格语法后是否有实质文本
                content_check = _re.sub(r"[|\- \n]", "", md)
                if len(content_check) >= 3:
                    return md
    except Exception as e:
        from loguru import logger
        logger.debug(f"[chunker] pandas 表格解析失败，回退到简单提取: {e}")

    # 回退方案：简单正则提取（不展开 rowspan/colspan，至少保留文本）
    return _html_table_to_text_simple(html)


def _html_table_to_text_simple(html: str) -> str:
    """简单 HTML→Markdown 表格转换（不支持 rowspan/colspan）。

    作为 pandas 解析失败的兜底。提取 <th>/<td> 文本，过滤全空行，
    第一行作为表头。
    """
    import re as _re

    def _strip_tags(s: str) -> str:
        return _re.sub(r"<[^>]+>", "", s).strip()

    rows = _re.findall(r"<tr[^>]*>(.*?)</tr>", html, _re.DOTALL | _re.IGNORECASE)
    if not rows:
        return ""

    th_cells_all = _re.findall(r"<th[^>]*>(.*?)</th>", html, _re.DOTALL | _re.IGNORECASE)
    headers = [_strip_tags(h) for h in th_cells_all]

    parsed_rows: list[list[str]] = []
    for row in rows:
        cells = _re.findall(r"<td[^>]*>(.*?)</td>", row, _re.DOTALL | _re.IGNORECASE)
        stripped = [_strip_tags(c) for c in cells]
        if stripped:
            parsed_rows.append(stripped)

    parsed_rows = [r for r in parsed_rows if any(c for c in r)]

    if not parsed_rows:
        return ""

    md_lines = []

    if headers:
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        data_rows = parsed_rows
    elif parsed_rows:
        first_row = parsed_rows[0]
        md_lines.append("| " + " | ".join(first_row) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(first_row)) + " |")
        data_rows = parsed_rows[1:]

    for row in data_rows:
        md_lines.append("| " + " | ".join(row) + " |")

    content_check = _re.sub(r"[|\- \n]", "", "\n".join(md_lines))
    if len(content_check) < 3:
        return ""

    return "\n".join(md_lines)


# ── 工具函数 ──────────────────────────────────────────────────────────────────


def _generate_doc_hash(text: str) -> str:
    """生成文档内容的 SHA-256 哈希"""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _split_large_md_table(md_table: str, max_size: int = 1000) -> list[str]:
    """
    把大 markdown 表格按行切成多个小表，每个切片自带表头。

    输入：完整 markdown 表格（| 表头 | + | --- | + 数据行）
    输出：list[str]，每个元素是一个完整 markdown 表格（表头 + 若干数据行）

    切分逻辑：
    - 第 1 行（表头）+ 第 2 行（| --- |）固定作为每个切片的开头
    - 数据行按字符数累加，超过 max_size 时输出当前切片
    - 单行很长（> max_size）时，单行单独成片
    - 极小表格（< 4 行）或总长 ≤ max_size 时原样返回
    """
    lines = md_table.split("\n")
    # 去掉末尾空行
    while lines and not lines[-1].strip():
        lines.pop()

    if len(lines) < 4 or len(md_table) <= max_size:
        return [md_table]

    header = lines[0]
    separator = lines[1]
    data_rows = lines[2:]

    # 验证 separator 是 markdown 表格分隔符（避免误切非表格文本）
    if not re.match(r"^\s*\|[\s\-:|]+\|\s*$", separator):
        return [md_table]

    # 表头开销（每个切片都带）
    header_len = len(header) + 1 + len(separator) + 1  # +2 个 \n

    pieces: list[str] = []
    current_rows: list[str] = []
    current_len = header_len

    for row in data_rows:
        # 累加会超 max_size 且当前已有累加行 → 输出当前切片
        if current_rows and current_len + len(row) + 1 > max_size:
            pieces.append("\n".join([header, separator] + current_rows))
            current_rows = [row]
            current_len = header_len + len(row) + 1
        else:
            current_rows.append(row)
            current_len += len(row) + 1

    # 输出剩余
    if current_rows:
        pieces.append("\n".join([header, separator] + current_rows))

    return pieces if pieces else [md_table]


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

    def __init__(self, remove_images: bool = True):
        """
        初始化分块器

        Args:
            remove_images: 清洗时是否移除图片，默认 True（与 cleaner.py 行为一致）
        """
        self._remove_images = remove_images

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
        source_key: str = "",
    ) -> List[Document]:
        """
        将 Markdown 内容切分为 Document 列表（父子结构）。

        Args:
            markdown: 原始 Markdown 文本
            doc_id: 文档 ID
            mode: 分块模式，"recursive"（默认）或 "semantic"
                  - recursive: 使用 MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
                  - semantic: 使用 embedding 相似度找语义边界
            source_key: 版本键（dataset_id::file_stem），透传到每个 chunk 的 metadata

        Returns:
            List[Document]: 每个 parent 对应一个 Document，包含切分后的 children
        """
        # ── 1. 清洗全文 ────────────────────────────────────────────────────────
        md = clean_text(markdown, remove_images=self._remove_images)

        if mode == "semantic":
            return self._chunk_semantic(md, title, doc_id, source_key=source_key)

        # ── 2. 按 Markdown 标题切分 parent sections ──────────────────────────
        raw_sections = self._parent_splitter.split_text(md)

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

            # ── 4a. 切 child chunks ─────────────────────────────────────────
            child_document_list = self._split_children(
                p_doc.page_content, p_doc.metadata, doc_id, parent_id, child_global_idx,
                source_key=source_key,
            )
            if child_document_list:
                child_global_idx += len(child_document_list)

            # parent 内容也做 HTML→MD 转换（保持与 child 一致），加 H1/H2/H3 路径前缀
            path_prefix = self._build_path_prefix(p_doc.metadata)
            parent_content = path_prefix + self._convert_tables_to_markdown(p_doc.page_content)
            document = Document(
                content=parent_content,
                metadata={
                    "doc_id": doc_id,
                    "doc_title": title,
                    "chunk_id": parent_id,
                    "doc_hash": _generate_doc_hash(parent_content),
                    "source_key": source_key,
                    # 透传 H1/H2/H3 路径，让下游 generation 可用
                    "H1": p_doc.metadata.get("H1", ""),
                    "H2": p_doc.metadata.get("H2", ""),
                    "H3": p_doc.metadata.get("H3", ""),
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

    # ── child chunk 切分 ─────────────────────────────────────────────────

    def _split_children(
        self,
        content: str,
        parent_metadata: dict,
        doc_id: str,
        parent_id: str,
        start_child_idx: int,
        source_key: str = "",
    ) -> List[ChildDocument]:
        """
        将 parent content 切分为 child chunks。

        策略（简化版）：
        1. 按 <table>...</table> 边界把 parent 切成单元序列
           [文本段1, 表格1, 文本段2, 表格2, ...]
        2. 文本段累加到 ~CHILD_CHUNK_SIZE 后输出；
           单段超长用 RecursiveCharacterTextSplitter 切（保留语义边界）
        3. 每个表格用 _html_table_to_text 转 markdown，作为独立 child
        4. 统一构造 ChildDocument（1 处，不重复）
        5. _merge_short_children 合并过短 chunks（< 50 字符）
        """
        # ── 1. 按表格边界切成单元 ─────────────────────────────────
        units: list[tuple[str, str]] = []  # [("text", ...), ("table", ...)]
        last_end = 0
        for m in _TABLE_PATTERN.finditer(content):
            if m.start() > last_end:
                units.append(("text", content[last_end:m.start()]))
            units.append(("table", m.group(0)))
            last_end = m.end()
        if last_end < len(content):
            units.append(("text", content[last_end:]))

        # ── 2. 处理每个单元，收集到 pieces: list[str] ──────────────
        pieces: list[str] = []
        text_buffer = ""

        def _flush_text(buf: str) -> list[str]:
            """切分或保留文本 buffer，返回 pieces 列表。"""
            buf = buf.strip()
            if not buf or is_low_quality_content(buf):
                return []
            if len(buf) <= CHILD_CHUNK_SIZE:
                return [buf]
            # 超长：用 RecursiveCharacterTextSplitter 切（保留段落/句子边界）
            return [
                s.strip()
                for s in self._child_splitter.split_text(buf)
                if s.strip() and not is_low_quality_content(s)
            ]

        for typ, unit in units:
            if typ == "table":
                # 表格前先 flush 文本 buffer
                if text_buffer:
                    pieces.extend(_flush_text(text_buffer))
                    text_buffer = ""
                # 表格转 markdown
                md = _html_table_to_text(unit)
                if md:
                    # 大表格按行切分（每片自带表头，~CHILD_CHUNK_SIZE*2 字符）
                    if len(md) > CHILD_CHUNK_SIZE * 2:
                        pieces.extend(_split_large_md_table(md, max_size=CHILD_CHUNK_SIZE * 2))
                    else:
                        pieces.append(md)
            else:  # text
                # 累加到 buffer；超过 2× CHILD_CHUNK_SIZE 时提前 flush（避免 buffer 过大）
                if text_buffer and len(text_buffer) + len(unit) > CHILD_CHUNK_SIZE * 2:
                    pieces.extend(_flush_text(text_buffer))
                    text_buffer = unit
                else:
                    text_buffer += unit

        # 末尾 flush
        if text_buffer:
            pieces.extend(_flush_text(text_buffer))

        # ── 3. 统一构造 ChildDocument ──────────────────────────────
        children: List[ChildDocument] = []
        for i, text in enumerate(pieces):
            child_id = f"{doc_id}_c{start_child_idx + i}"
            children.append(
                ChildDocument(
                    content=text,
                    metadata={
                        "doc_id": doc_id,
                        "doc_title": parent_metadata.get("doc_title", ""),
                        "chunk_id": child_id,
                        "doc_hash": _generate_doc_hash(text),
                        "parent_id": parent_id,
                    },
                )
            )

        # ── 4. 合并过短 chunks（< 50 字符）────────────────────────
        children = self._merge_short_children(children)

        # ── 5. 透传 source_key + H1/H2/H3 到 child metadata ──────
        if source_key:
            for child in children:
                child.metadata["source_key"] = source_key
        for hk in ("H1", "H2", "H3"):
            hv = parent_metadata.get(hk, "")
            if hv:
                for child in children:
                    child.metadata[hk] = hv

        return children

    # ── parent 合并/拆分/清理 ───────────────────────────────────────────────

    def _merge_small_parents(self, sections: list) -> list:
        """
        将连续的小 section（< MIN_PARENT_SIZE）合并，
        直到合并后达到 MIN_PARENT_SIZE。
        同时有 MAX_PARENT_SIZE 上限保护，避免合并后过大。
        """
        if not sections:
            return []

        merged, current = [], None

        for sec in sections:
            if current is None:
                current = sec
            else:
                # Size guard: 合并后不超过 MAX_PARENT_SIZE
                if len(current.page_content) + len(sec.page_content) + 2 > MAX_PARENT_SIZE:
                    merged.append(current)
                    current = sec
                    continue

                current.page_content += "\n\n" + sec.page_content
                self._merge_metadata(current, sec)

            if len(current.page_content) >= MIN_PARENT_SIZE:
                merged.append(current)
                current = None

        # 处理剩余
        if current:
            if merged:
                # 尝试合并到最后一个，但不超过 MAX_PARENT_SIZE
                if len(merged[-1].page_content) + len(current.page_content) + 2 <= MAX_PARENT_SIZE:
                    merged[-1].page_content += "\n\n" + current.page_content
                    self._merge_metadata(merged[-1], current)
                else:
                    merged.append(current)
            else:
                merged.append(current)

        return merged

    def _split_large_parents(self, sections: list) -> list:
        """
        将超过 MAX_PARENT_SIZE 的 section 拆分。

        表格保护：<table>...</table> 作为不可分割单元，整体保留。
        切分逻辑：按 _TABLE_PATTERN 把 section 切成 [文本段, 表格, 文本段, 表格, ...] 单元序列，
        然后按字符数累加，累加超过 MAX_PARENT_SIZE 时输出当前 chunk，开启新 chunk。
        表格作为整体进入某个 chunk（即使单独超过 MAX_PARENT_SIZE，也保留完整）。
        """
        result = []
        for sec in sections:
            if len(sec.page_content) <= MAX_PARENT_SIZE:
                result.append(sec)
                continue

            # 1. 切成"单元"序列（表格整体 + 文本段）
            units: list[str] = []
            last_end = 0
            for m in _TABLE_PATTERN.finditer(sec.page_content):
                if m.start() > last_end:
                    units.append(sec.page_content[last_end:m.start()])
                units.append(m.group(0))  # 整个 <table>...</table>
                last_end = m.end()
            if last_end < len(sec.page_content):
                units.append(sec.page_content[last_end:])

            # 2. 按字符数累加，表格作为不可分单元
            chunks: list[str] = []
            current = ""
            for unit in units:
                if current and len(current) + len(unit) > MAX_PARENT_SIZE:
                    chunks.append(current)
                    current = unit
                else:
                    current += unit
            if current:
                chunks.append(current)

            # 3. 转 LCDocument（保留原 section 的 metadata：H1/H2/H3 路径）
            for ch in chunks:
                new_doc = LCDocument(page_content=ch, metadata=dict(sec.metadata))
                result.append(new_doc)

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

    # ── 辅助方法 ─────────────────────────────────────────────────────────

    def _convert_tables_to_markdown(self, text: str) -> str:
        """将文本中所有 HTML 表格转为 Markdown 格式"""
        return _TABLE_PATTERN.sub(lambda m: _html_table_to_text(m.group(0)), text)

    @staticmethod
    def _build_path_prefix(metadata: dict) -> str:
        """
        从 metadata 构建路径前缀，用于让每个 parent 自带 H1/H2/H3 上下文。

        例：metadata = {H1: "1. 总体设计", H2: "1.1 系统", H3: "1.1.1 硬件"}
            → "[路径: 1. 总体设计 / 1.1 系统 / 1.1.1 硬件]\\n\\n"

        无标题路径时返回空字符串。
        """
        parts = [metadata.get(k, "") for k in ("H1", "H2", "H3") if metadata.get(k)]
        if not parts:
            return ""
        return f"[路径: {' / '.join(parts)}]\n\n"

    @staticmethod
    def _merge_metadata(target, source):
        """将 source.metadata 合并到 target.metadata。

        相同 key 相同值时不拼接（避免 "A -> A -> A" 重复），
        仅当值不同时拼接为 "原值 -> 新值"（记录多个不同 section 的合并路径）。
        """
        if not hasattr(source, "metadata"):
            return
        for k, v in source.metadata.items():
            if k in target.metadata:
                if target.metadata[k] != v:
                    target.metadata[k] = f"{target.metadata[k]} -> {v}"
                # 相同值不动
            else:
                target.metadata[k] = v

    # ── 丢弃无用 section ───────────────────────────────────────────

    # 需丢弃的 section header 关键词（大小写不敏感，子串匹配）
    _META_HEADER_KEYWORDS: list[str] = [
        "目录", "table of contents", "index",
        "版本修订记录", "修订记录", "变更记录", "changelog", "版本历史",
        "免责声明", "版权声明", "声明", "注意事项", "重要声明",
        "preface", "foreword",
        "acknowledgement", "acknowledgments", "致谢",
        "abbreviation", "缩写", "glossary", "词汇表",
        "参考文献", "reference", "references",
    ]

    # section header 正则模式（处理 MinerU 把目录条目识别成 H1 的情况）
    # 例: H1='5 单机试验要求.. 25'  H1='6.2.2 太空计算标准体系 ..... 56'
    _META_HEADER_PATTERNS: list[re.Pattern] = [
        # 目录条目格式：章节号 + 文字 + 连续点/空格 + 末尾页码
        re.compile(r"^\s*\d+(?:\.\d+)*\s+\S.+[\.\s]{2,}\d{1,4}\s*$"),
        # "目 录"（中间任意空格数）
        re.compile(r"^目\s*录\s*$"),
    ]

    def _drop_meta_sections(
        self, sections: list
    ) -> list:
        """
        丢弃目录、索引、版本修订记录、免责声明等文档元信息 section。

        判断逻辑（任一命中即丢弃）：
        1. section header 含 _META_HEADER_KEYWORDS 关键词（子串匹配）
        2. section header 匹配 _META_HEADER_PATTERNS 正则（目录条目格式 / 目 录）
        """
        import re

        def is_meta_section(sec) -> bool:
            # MarkdownHeaderTextSplitter 的 metadata 用 H1/H2/H3 作为 key
            # 三个都检查（最深的优先），任意一个命中即丢
            for key in ("H1", "H2", "H3"):
                header = sec.metadata.get(key, "")
                if not header:
                    continue
                header_lower = header.lower()
                # 关键词子串匹配
                for kw in self._META_HEADER_KEYWORDS:
                    if kw.lower() in header_lower:
                        return True
                # 正则模式匹配
                for pat in self._META_HEADER_PATTERNS:
                    if pat.search(header):
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
        - 如果有前一个 chunk，合并到前一个（不超过 CHILD_CHUNK_SIZE）
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

            # 表格块不参与合并（当前 child 是表格 → 直接 append）
            if child.content.strip().startswith("|"):
                result.append(child)
                continue

            # 检查是否过短
            if len(child.content) < self.MIN_CHILD_CHUNK_SIZE:
                if result:
                    prev = result[-1]
                    # 前一个是表格：短 child 作为表格的 caption/note 始终合并（跳过 size guard）
                    if prev.content.strip().startswith("|"):
                        prev.content = prev.content.rstrip() + "\n\n" + child.content
                        prev.metadata["doc_hash"] = _generate_doc_hash(prev.content)
                        continue
                    # 前一个是文本：size guard（不超过 CHILD_CHUNK_SIZE * 1.5）
                    merged_len = len(prev.content) + len(child.content) + 2
                    if merged_len <= CHILD_CHUNK_SIZE * 1.5:
                        prev.content = prev.content.rstrip() + "\n\n" + child.content
                        prev.metadata["doc_hash"] = _generate_doc_hash(prev.content)
                    else:
                        # 合并会超长，保留原 child
                        result.append(child)
                    continue
                elif i < len(children) - 1:
                    # 没有前一个，合并到后一个
                    next_child = children[i + 1]
                    # 后一个是表格：短 child 作为 caption 合并到表格（跳过 size guard）
                    if next_child.content.strip().startswith("|"):
                        next_child.content = child.content.rstrip() + "\n\n" + next_child.content.lstrip()
                        next_child.metadata["doc_hash"] = _generate_doc_hash(next_child.content)
                        continue
                    # 后一个是文本：size guard 检查
                    merged_len = len(child.content) + len(next_child.content) + 2
                    if merged_len <= CHILD_CHUNK_SIZE * 1.5:
                        next_child.content = child.content.rstrip() + "\n\n" + next_child.content.lstrip()
                        next_child.metadata["doc_hash"] = _generate_doc_hash(next_child.content)
                    else:
                        result.append(child)
                    continue
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
        source_key: str = "",
    ) -> List[Document]:
        """
        基于 embedding 相似度的语义分块（支持中文）。

        流程：
        1. 按中英文句子结束符预分割
        2. 将句子组合成 ~3 句一组的片段，计算 embedding
        3. 计算相邻片段的 cosine similarity
        4. 相似度低于阈值的断点切分
        5. 每个语义 chunk 创建一个 Document
        6. 每个 Document 内部用 _split_children 切 child chunks

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
            return self._create_single_document(md, title, doc_id, 0, len(segments), source_key=source_key)

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

            child_document_list = self._split_children(
                chunk_text,
                {"doc_title": title},
                doc_id,
                parent_id,
                child_global_idx,
                source_key=source_key,
            )
            if child_document_list:
                child_global_idx += len(child_document_list)

            document = Document(
                content=chunk_text,
                metadata={
                    "doc_id": doc_id,
                    "doc_title": title,
                    "chunk_id": parent_id,
                    "doc_hash": _generate_doc_hash(chunk_text),
                    "source_key": source_key,
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
        source_key: str = "",
    ) -> List[Document]:
        """当无法分块时，创建单个 Document"""
        parent_id = f"{doc_id}_p{parent_idx}"

        child_document_list = self._split_children(
            md,
            {"doc_title": title},
            doc_id,
            parent_id,
            child_global_idx,
            source_key=source_key,
        )

        document = Document(
            content=md,
            metadata={
                "doc_id": doc_id,
                "doc_title": title,
                "chunk_id": parent_id,
                "doc_hash": _generate_doc_hash(md),
                "source_key": source_key,
            },
            children=child_document_list if child_document_list else None,
        )

        from loguru import logger

        logger.info(
            f"[Chunker] 语义分块完成: 1 document "
            f"(children={len(child_document_list) if child_document_list else 0})"
        )

        return [document]
