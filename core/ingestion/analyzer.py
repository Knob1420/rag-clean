"""
文档结构化分析 — 分块之前先理解文档

核心创新：把 enrichment 从"事后打标签"前移为"分块前的文档理解"，
根据理解结果选择分块策略。
"""

import re
from typing import List, Dict, Optional, Tuple

from loguru import logger

from models import TableSpec, DocumentAnalysis


# ── domain 合法枚举 ──────────────────────────────────────

VALID_DOMAINS = frozenset({
    "Product_Tech",   # 产品与技术：手册、规格、研发文档、技术方案
    "Biz_Market",     # 业务与市场：宣传册、售前方案、竞品分析、报价
    "Ops_Mgmt",       # 运营与管理：项目管理、流程规范、会议纪要
    "Corp_Support",   # 企业与职能：HR 制度、财务报销、行政通知
})


def _validate_domain(raw: str) -> str:
    """后置校验 domain，不在枚举内则回退默认值"""
    if raw in VALID_DOMAINS:
        return raw
    return "Product_Tech"


# ── 内部数据结构 ──────────────────────────────────────────


class _LLMAnalysisResult:
    """LLM 分析的原始结果"""

    def __init__(
        self,
        doc_type: str,
        domain: str,
        entities: Dict[str, str],
        filter_terms: List[str],
        topics: List[str],
        doc_intent: Optional[str],
        summary: str,
        confidence: int,
    ):
        self.doc_type = doc_type
        self.domain = domain
        self.entities = entities
        self.filter_terms = filter_terms
        self.topics = topics
        self.doc_intent = doc_intent
        self.summary = summary
        self.confidence = confidence


# ── 分析器 ──────────────────────────────────────────────


class DocumentAnalyzer:
    """文档级结构化分析 — 分块之前先理解文档"""

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM 客户端，需提供 extract_doc_info() 方法。
                        延迟导入以避免循环依赖。
        """
        self._llm_client = llm_client

    def _get_llm(self):
        """延迟获取 LLM 客户端"""
        if self._llm_client is None:
            from core.generation.llm import get_llm_client

            self._llm_client = get_llm_client()
        return self._llm_client

    def analyze(self, title: str, markdown: str) -> DocumentAnalysis:
        """
        分析文档结构

        Args:
            title: 文档标题
            markdown: 文档 Markdown 内容

        Returns:
            DocumentAnalysis 分析结果
        """
        logger.info(f"[Analyzer] 开始分析文档: {title}")

        # 1. 提取所有表格（正则，不依赖 LLM）
        tables = self._extract_all_tables(markdown)
        logger.info(f"  提取到 {len(tables)} 个表格")

        # 2. 提取章节结构树
        sections = self._extract_section_tree(markdown)
        logger.info(f"  提取到 {len(sections)} 个章节")

        # 3. LLM 分析：doc_type + domain + entities + filter_terms + ...（一次调用）
        llm_result = self._llm_analyze(title, markdown[:3000], tables)

        # 4. domain 后置校验
        domain = _validate_domain(llm_result.domain)

        # 5. confidence 低分警告（只记日志，不持久化）
        if llm_result.confidence < 70:
            logger.warning(
                f"  [低置信度] confidence={llm_result.confidence}, "
                f"doc_type={llm_result.doc_type}, domain={domain} — "
                f"建议人工复核"
            )

        analysis = DocumentAnalysis(
            doc_type=llm_result.doc_type,
            domain=domain,
            entities=llm_result.entities,
            filter_terms=llm_result.filter_terms,
            topics=llm_result.topics,
            doc_intent=llm_result.doc_intent,
            summary=llm_result.summary,
            tables=tables,
            section_tree=sections,
        )

        logger.info(
            f"  doc_type={analysis.doc_type}, domain={analysis.domain}, "
            f"confidence={llm_result.confidence}"
        )
        logger.info(f"  filter_terms={analysis.filter_terms}")
        logger.info(f"  entities={analysis.entities}")
        return analysis

    # ── 表格提取 ──────────────────────────────────────

    def _extract_all_tables(self, markdown: str) -> List[TableSpec]:
        """
        在文档级别提取所有表格，分块之前。
        支持 Markdown 管道表格和 HTML <table>。
        """
        tables: List[TableSpec] = []

        # 提取 HTML 表格
        tables.extend(self._extract_html_tables(markdown))

        # 提取 Markdown 管道表格
        tables.extend(self._extract_md_tables(markdown))

        # 按 position 排序
        tables.sort(key=lambda t: t.position)
        return tables

    def _extract_html_tables(self, markdown: str) -> List[TableSpec]:
        """提取 HTML <table> 并转为 key-value"""
        tables: List[TableSpec] = []
        pattern = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)

        for match in pattern.finditer(markdown):
            position = match.start()
            table_html = match.group(1)

            rows = re.findall(
                r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE
            )
            if not rows:
                continue

            parsed_rows: List[List[str]] = []
            for row in rows:
                cells = re.findall(
                    r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL | re.IGNORECASE
                )
                cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
                if cells:
                    parsed_rows.append(cells)

            if not parsed_rows:
                continue

            # 尝试结构化解析
            fields, rows = self._parse_table_kv(parsed_rows)
            if fields or rows:
                pre_text = markdown[:position]
                source = self._guess_table_title(pre_text)
                tables.append(
                    TableSpec(source_table=source, fields=fields, rows=rows, position=position)
                )

        return tables

    def _extract_md_tables(self, markdown: str) -> List[TableSpec]:
        """提取 Markdown 管道表格"""
        tables: List[TableSpec] = []

        lines = markdown.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line.startswith("|"):
                i += 1
                continue

            # 收集连续的表格行
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1

            if len(table_lines) < 3:  # 至少需要 header + separator + 1 data row
                continue

            # 验证分隔行
            if not re.match(r"^\|[\s\-:|]+\|$", table_lines[1]):
                continue

            # 解析表格
            parsed_rows: List[List[str]] = []
            for tl in table_lines:
                if re.match(r"^\|[\s\-:|]+\|$", tl):
                    continue  # 跳过分隔行
                cells = [c.strip() for c in tl.strip("|").split("|")]
                parsed_rows.append(cells)

            if not parsed_rows:
                continue

            fields, rows = self._parse_table_kv(parsed_rows)
            if fields or rows:
                pos = sum(len(lines[j]) + 1 for j in range(i - len(table_lines)))
                pre_text = markdown[:pos]
                source = self._guess_table_title(pre_text)
                tables.append(
                    TableSpec(source_table=source, fields=fields, rows=rows, position=pos)
                )

        return tables

    @staticmethod
    def _parse_table_kv(rows: List[List[str]]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        """
        将表格行解析为结构化数据。
        - 两列表格 → fields KV dict
        - 多列表格 → rows 行列表（保留完整行结构）

        Returns:
            (fields, rows) 二者互斥：两列表 fields 有值 rows 为空，多列表反之
        """
        if not rows:
            return {}, []

        col_count = len(rows[0])

        # 两列表格：直接 K-V
        if col_count == 2 and len(rows) >= 2:
            fields = {}
            for row in rows:
                if len(row) >= 2:
                    key = row[0].strip()
                    val = row[1].strip()
                    if key and val:
                        fields[key] = val
            return fields, []

        # 多列表格（3列+）：保留行结构，避免同名 header 覆盖
        if col_count >= 3 and len(rows) >= 2:
            headers = [h.strip() for h in rows[0]]
            if all(headers):
                structured_rows = []
                for row in rows[1:]:
                    row_dict = {}
                    for j, header in enumerate(headers):
                        if j < len(row) and row[j].strip():
                            row_dict[header] = row[j].strip()
                    if row_dict:
                        structured_rows.append(row_dict)
                return {}, structured_rows

        return {}, []

    @staticmethod
    def _guess_table_title(pre_text: str) -> str:
        """从表格前方的文本猜测表格标题"""
        lines = pre_text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^#+\s*", "", line)
            if cleaned.startswith("!["):
                continue
            if cleaned:
                return cleaned
        return "未知表格"

    # ── 章节提取 ──────────────────────────────────────

    def _extract_section_tree(self, markdown: str) -> List[Dict]:
        """
        提取标题层级结构

        Returns:
            [{"title": "技术参数", "level": 2, "position": 123, "has_table": True}]
        """
        sections: List[Dict] = []
        lines = markdown.split("\n")
        pos = 0

        for line in lines:
            m = re.match(r"^(#{1,6})\s+(.+)$", line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                sections.append(
                    {
                        "title": title,
                        "level": level,
                        "position": pos,
                    }
                )
            pos += len(line) + 1

        # 标记哪些 section 包含表格
        for sec in sections:
            sec_start = sec["position"]
            sec_end = len(markdown)
            for other in sections:
                if (
                    other["position"] > sec_start
                    and other["level"] <= sec["level"]
                    and other["position"] < sec_end
                ):
                    sec_end = other["position"]

            section_text = markdown[sec_start:sec_end]
            sec["has_table"] = bool(
                re.search(r"<table|^\|", section_text, re.MULTILINE | re.IGNORECASE)
            )

        return sections

    # ── LLM 分析 ──────────────────────────────────────

    def _llm_analyze(
        self, title: str, content_preview: str, tables: List[TableSpec]
    ) -> _LLMAnalysisResult:
        """
        一次 LLM 调用，提取文档级元数据。
        传入表格信息帮助 LLM 更好理解文档。
        """
        # 构建表格摘要
        tables_summary = ""
        if tables:
            tables_summary = "\n\n【文档包含的表格】"
            for i, t in enumerate(tables[:5]):
                fields_preview = list(t.fields.items())[:5]
                fields_str = ", ".join(f"{k}={v}" for k, v in fields_preview)
                tables_summary += f"\n表格{i + 1}({t.source_table}): {fields_str}"

        try:
            llm = self._get_llm()
            result = llm.extract_doc_info(
                doc_id="",
                title=title,
                content=content_preview + tables_summary,
            )

            return _LLMAnalysisResult(
                doc_type=result.get("doc_type", "其他"),
                domain=result.get("domain", ""),
                entities=result.get("entities", {}),
                filter_terms=result.get("filter_terms", []),
                topics=result.get("topics", []),
                doc_intent=result.get("doc_intent"),
                summary=result.get("summary", ""),
                confidence=result.get("confidence", 100),
            )

        except Exception as e:
            logger.warning(f"LLM 分析失败，使用默认值: {e}")
            return _LLMAnalysisResult(
                doc_type="其他",
                domain="Product_Tech",
                entities={},
                filter_terms=[],
                topics=[],
                doc_intent=None,
                summary="",
                confidence=0,
            )
