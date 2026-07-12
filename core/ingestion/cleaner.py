"""
文本清洗器 — 对标 Dify CleanProcessor

将原始文本清洗为干净的可分块文本。

清洗规则：
1. XML/模板符号清洗（默认）
2. 控制字符清洗（默认）
3. 多余空格/换行清洗（可选）
4. URL/邮箱移除（可选）
5. Markdown 图片移除（可选）
"""

import re


def _is_meta_table(table_content: str) -> bool:
    """
    判断表格是否为无意义/元信息表格（应删除）。

    判断逻辑（满足任一即删除）：
    1. 签字区/审批区表格（含"编制/审核/批准/会签"）
    2. 文档封面信息表（含"文件编号/研制单位/阶段标记/版本"）
    3. 全空或几乎全空的表格
    """
    content = table_content.lower()

    # ── 1. 签字区/审批区/封面信息表（所有数据集通用） ──────────
    _META_TABLE_KEYWORDS = [
        "编制",
        "审核",
        "批准",
        "会签",
        "签字",
        "文件编号",
        "研制单位",
        "阶段标记",
        "文档编号",
        "版次",
        "共.*页",
    ]
    for kw in _META_TABLE_KEYWORDS:
        if re.search(kw, content):
            return True

    # ── 2. 全空表格判断 ──────────────────────────────────────────
    # 去掉所有 HTML 标签和空白后，剩余有效文本过少
    plain = re.sub(r"<[^>]+>", "", table_content)
    plain = re.sub(r"\s+", "", plain)
    # 去掉 markdown 表格语法
    plain = re.sub(r"[|\-:]", "", plain)
    if len(plain) < 5:
        return True

    return False


# ── 低质量内容判定（共享） ──────────────────────────────────────────

# 预编译正则
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TABLE_FRAG_RE = re.compile(r"</td>\s*</tr>\s*<tr>\s*<td", re.IGNORECASE)
_TD_CLOSE_RE = re.compile(r"</td>", re.IGNORECASE)
_EFFECTIVE_WORD_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z]{2,}|\d+")


def is_low_quality_content(text: str) -> bool:
    """
    判断文本是否为低质量内容（HTML 碎片等）。

    五条信号（任一命中即判定为垃圾）：
    1. HTML 标签占比 >50% 且去标签纯文本 <20 字符
    2. 表格碎片签名：含 </td></tr><tr><td 模式（MinerU 表格切割残留）
    3. 重复 HTML 单元格：</td> 出现 ≥3 次，且有效词 <15 个
    4. 低内容密度：HTML 占比 >15%，且有效词 <15 个
    5. 纯文本（无 HTML 标签）且有效词 <10 个

    安全保障：信号 2-4 仅在有 HTML 标签时激活；
    信号 5 覆盖纯文本短 chunk（如 "共3"、"64W" 等）。
    """
    tags = _HTML_TAG_RE.findall(text)
    if not tags:
        # 信号 5：纯文本（无 HTML 标签）且有效词 <3 个
        plain = text.strip()
        effective_words = _EFFECTIVE_WORD_RE.findall(plain)
        if len(effective_words) < 3:
            return True
        return False

    tag_len = sum(len(t) for t in tags)
    html_ratio = tag_len / max(len(text), 1)
    plain = _HTML_TAG_RE.sub("", text).strip()

    # 信号 1：原信号保留 — HTML >50% 且纯文本 <20 字符
    if html_ratio > 0.5 and len(plain) < 20:
        return True

    effective_words = _EFFECTIVE_WORD_RE.findall(plain)
    word_count = len(effective_words)

    # 信号 2：表格碎片签名（MinerU 残留）
    if _TABLE_FRAG_RE.search(text):
        return True

    # 信号 3：重复 HTML 单元格 + 少量有效词
    td_count = len(_TD_CLOSE_RE.findall(text))
    if td_count >= 3 and word_count < 15:
        return True

    # 信号 4：低内容密度
    if html_ratio > 0.15 and word_count < 15:
        return True

    return False


class TextCleaner:
    """文本清洗器"""

    # 控制字符范围
    _CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\xEF\xBF\xBE]")
    _FFFE = re.compile("\ufffe")

    # 多余空白
    _MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
    _MULTIPLE_SPACES = re.compile(
        r"[\t\f\r\x20\u00a0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]{2,}"
    )

    # Markdown 链接/图片占位符
    _MD_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\((https?://[^)]+)\)")
    _MD_IMAGE_PATTERN = re.compile(r"!\[.*?\]\((https?://[^)]+)\)")

    # URL 和邮箱正则
    _EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    _URL_PATTERN = re.compile(r"https?://\S+")

    @classmethod
    def clean(cls, text: str, remove_images: bool = True) -> str:
        """
        清洗文本（默认执行所有清洗规则）。

        清洗规则：
        1. XML/模板符号清洗
        2. 控制字符清洗
        3. 多余空格/换行清洗
        4. URL/邮箱/Markdown图片移除（保护 Markdown 链接）
        5. Markdown 图片和 HTML img 标签移除（可选，同时移除紧跟的图例行）
        6. 签字区/审批区等元信息 HTML 表格移除（所有数据集）
        7. 页眉页脚清洗（页码、原理图标记、日期、文档编号等）

        Args:
            text: 原始文本
            remove_images: 是否移除 Markdown 图片和 HTML img 标签，默认 True

        Returns:
            清洗后的文本
        """
        if not text:
            return text

        # 1. XML/模板符号清洗
        text = cls._clean_xml_symbols(text)

        # 2. 控制字符清洗
        text = cls._clean_control_chars(text)

        # 3. 多余空格/换行清洗
        text = cls._clean_extra_spaces(text)

        # 4. URL/邮箱/Markdown图片移除（保护 Markdown 链接）
        text = cls._remove_urls_and_emails(text)

        # 5. 图片移除（可选）
        if remove_images:
            text = cls._remove_images(text)

        # 6. 签字区/审批区等元信息 HTML 表格移除（所有数据集）
        text = cls._remove_meta_html_tables(text)

        # 7. 页眉页脚清洗
        text = cls._clean_headers_footers(text)

        # 8. 目录页内容清洗（章节号+省略号+页码，如"4 力学环境试验条件.. 6"）
        text = cls._clean_toc_pages(text)

        return text.strip()

    @staticmethod
    def _clean_xml_symbols(text: str) -> str:
        """清洗 XML/模板特殊符号"""
        text = re.sub(r"<\|", "<", text)
        text = re.sub(r"\|>", ">", text)
        return text

    @classmethod
    def _clean_control_chars(cls, text: str) -> str:
        """移除控制字符"""
        text = cls._CONTROL_CHARS.sub("", text)
        text = cls._FFFE.sub("", text)
        return text

    @classmethod
    def _clean_extra_spaces(cls, text: str) -> str:
        """清理多余空格和换行"""
        # 3+ 换行 → 2 换行
        text = cls._MULTIPLE_NEWLINES.sub("\n\n", text)
        # 多余空格 → 单空格
        text = cls._MULTIPLE_SPACES.sub(" ", text)
        return text

    @classmethod
    def _remove_urls_and_emails(cls, text: str) -> str:
        """
        移除 URL、邮箱和 Markdown 图片，但保护 Markdown 链接。

        流程：
        1. 保护 Markdown 链接: [text](url) → __MD_LINK_0__
        2. 移除 Markdown 图片: ![alt](url) → 直接删除（不需要恢复）
        3. 移除邮箱
        4. 移除普通 URL
        5. 恢复 Markdown 链接
        """
        # 保护 Markdown 链接
        links: list[tuple[str, str, str]] = []

        def protect_link(match: re.Match) -> str:
            link_text = match.group(1)
            url = match.group(2)
            placeholder = f"__MD_LINK_{len(links)}__"
            links.append((link_text, url, placeholder))
            return placeholder

        text = cls._MD_LINK_PATTERN.sub(protect_link, text)

        # Markdown 图片直接删除（不恢复）
        text = cls._MD_IMAGE_PATTERN.sub("", text)

        # 移除邮箱
        text = cls._EMAIL_PATTERN.sub("", text)

        # 移除普通 URL
        text = cls._URL_PATTERN.sub("", text)

        # 恢复 Markdown 链接
        for link_text, url, placeholder in links:
            restored = f"[{link_text}]({url})"
            text = text.replace(placeholder, restored)

        return text

    @classmethod
    def _remove_images(cls, text: str) -> str:
        """移除 Markdown 图片、HTML img 标签及其紧跟的图例/解释文字"""
        # Markdown 图片: ![alt](url) 或 ![alt](url "title")
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)(?:\s*\"[^\"]*\")?", "", text)
        # HTML img 标签
        text = re.sub(r"<img\s[^>]*/?>", "", text, flags=re.IGNORECASE)

        # 按行处理，移除图片行及其紧跟的图例行
        lines = text.split("\n")
        result_lines: list[str] = []
        skip_next_caption = False

        i = 0
        while i < len(lines):
            line = lines[i]
            # 检查是否是图片行
            is_image = bool(
                re.search(r"!\[[^\]]*\]\([^)]*\)", line)
                or re.search(r"<img\s[^>]*/?>", line, re.IGNORECASE)
            )
            if is_image:
                skip_next_caption = True
                i += 1
                continue

            # 检查是否是图例行（图片后面紧跟的那一行）
            if skip_next_caption:
                caption_stripped = line.strip()
                # 判断是否是图例：图/Figure/Fig/注: 等开头
                caption_match = re.match(
                    r"^(图\s*\d+[\.、:：]?\s*|"
                    r"Figure\s*\d+[\.、:：]?\s*|"
                    r"Fig\.?\s*\d+[\.、:：]?\s*|"
                    r"注[：:]\s*|"
                    r"\[?图\s*\d+[^\s]*\s*)",
                    caption_stripped,
                    re.IGNORECASE,
                )
                if caption_match:
                    skip_next_caption = False  # 重置，下一行不是图例
                    i += 1
                    continue
                skip_next_caption = False

            result_lines.append(line)
            i += 1

        return "\n".join(result_lines)

    @classmethod
    def _remove_meta_html_tables(cls, text: str) -> str:
        """
        移除签字区/审批区/封面信息等元信息 HTML 表格（所有数据集通用）。

        这些表格通常包含：编制/审核/批准/会签、文件编号/研制单位/阶段标记等，
        对检索毫无价值，且 HTML 格式碎片化严重。

        支持 MinerU 把一个大表格拆成多个碎片 <table> 的情况：
        对间隔 <50 字符的连续 <table> 分组，聚合判断是否为元信息表格。
        """
        html_table_pattern = re.compile(
            r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE
        )

        # 第一遍：收集所有表格及其位置
        tables_info = []
        for match in html_table_pattern.finditer(text):
            tables_info.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "content": match.group(0),
                }
            )

        if not tables_info:
            return text

        # 第二遍：对连续的（间隔 <50 字符的）表格分组
        _META_KEYWORDS = [
            "编制",
            "审核",
            "批准",
            "会签",
            "签字",
            "文件编号",
            "研制单位",
            "阶段标记",
            "文档编号",
            "版次",
            "共",
            "页",
        ]

        groups: list[list[dict]] = []
        current_group = [tables_info[0]]
        for i in range(1, len(tables_info)):
            prev_end = tables_info[i - 1]["end"]
            curr_start = tables_info[i]["start"]
            gap_text = text[prev_end:curr_start].strip()
            # 只有间隔为纯空白/换行时才视为同一组碎片
            # 间隔含任何实质文本 → 不合并，属于不同组
            if not gap_text:
                current_group.append(tables_info[i])
            else:
                groups.append(current_group)
                current_group = [tables_info[i]]
        groups.append(current_group)

        # 第三遍：对每组聚合判断是否为元信息表格
        to_delete_ranges: list[tuple[int, int]] = []
        for group in groups:
            combined = " ".join(t["content"] for t in group)
            combined_lower = combined.lower()
            hit_count = sum(1 for kw in _META_KEYWORDS if kw in combined_lower)
            if hit_count >= 2:
                # 整组删除
                start = group[0]["start"]
                end = group[-1]["end"]
                to_delete_ranges.append((start, end))

        # 第四遍：从后往前删除，避免偏移
        result = text
        for start, end in reversed(to_delete_ranges):
            result = result[:start] + result[end:]

        return result

    # 页眉页脚正则（类方法，方便子类扩展）
    _HEADER_FOOTER_PATTERNS: list[re.Pattern] = [
        # 页码: 第X页, Page X, - 1 -, 1/3
        re.compile(r"第\s*\d+\s*页"),
        re.compile(r"Page\s*\d+", re.IGNORECASE),
        re.compile(r"-\s*\d+\s*-"),
        re.compile(r"\b\d+\s*/\s*\d+\b"),
        # 共N页 / 共 页 / 共N（独立行）— MinerU 经常输出"共3"、"共 页"残留
        re.compile(r"^[\s]*共\s*\d*\s*页?[\s]*$", re.MULTILINE),
        # 原理图/电路图标记: SHEET 1/3, DWG-001, 电路图, 原理图
        re.compile(r"SHEET\s*\d+\s*/\s*\d+", re.IGNORECASE),
        re.compile(r"DWG[_-]?\d+", re.IGNORECASE),
        re.compile(r"原理图|电路图|PCB图|布线图"),
        # 日期: 2024/01/01, 2024年01月01日, 2024-01-01
        re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?"),
        # 文档控制信息: 版本/版次, 文件编号, 文档编号
        re.compile(r"版本\s*[A-Z]?\d+[\d.]*", re.IGNORECASE),
        re.compile(r"版次\s*\d+", re.IGNORECASE),
        re.compile(r"文件编号[：:]?\s*[A-Z0-9_-]+", re.IGNORECASE),
        re.compile(r"文档编号[：:]?\s*[A-Z0-9_-]+", re.IGNORECASE),
        re.compile(r"DOC\s*NO[.:]?\s*[A-Z0-9_-]+", re.IGNORECASE),
        # 公司/厂商标记
        re.compile(r"©\s*\d{4}.*|版权所有|All\s*rights\s*reserved", re.IGNORECASE),
        # 保密标记
        re.compile(r"机密|保密|NDA|CONFIDENTIAL", re.IGNORECASE),
        # 签字区/审批区残留
        re.compile(r"编制\s*[:-]?\s*\S+|审核\s*[:-]?\s*\S+|批准\s*[:-]?\s*\S+"),
        # 续表 / 续表A1 / 续表 3 / 表X（续）（独立行）
        re.compile(r"^[\s]*续表\s*[A-Z]?[\d]*[\s]*$", re.MULTILINE),
        re.compile(r"^[\s]*表\s*\d+\s*[（(]续[)）][\s]*$", re.MULTILINE),
        # 元信息 section header 独立行（修订记录/参考文献/目录等）
        re.compile(
            r"^[\s]*(修订记录|修订履历|版本历史|变更记录|参考文献|目录|目  录|索引)[\s]*$",
            re.MULTILINE,
        ),
    ]

    # 目录页内容正则（章节号 + 标题 + 末尾页码）
    # 例: "4 力学环境试验条件.. 6"  "3.2 星上仪器设备.. .12"  "6.2.2 标准 ..... 56"
    # 放宽：要求末尾数字前有"连续点或空格"（避免误匹配'见表 3.5'这种正文引用）
    _TOC_LINE_PATTERN = re.compile(
        r"^\s*\d+(?:\.\d+)*\s+\S[^\n]*?[\.\s]{2,}\d{1,4}\s*$", re.MULTILINE
    )

    @classmethod
    def _clean_headers_footers(cls, text: str) -> str:
        """移除页眉页脚中的常见标识（页码、原理图标记、日期、文档编号等）"""
        for pattern in cls._HEADER_FOOTER_PATTERNS:
            text = pattern.sub("", text)
        # 清理替换后产生的多余空白
        text = cls._clean_extra_spaces(text)
        return text

    @classmethod
    def _clean_toc_pages(cls, text: str) -> str:
        """
        清洗目录页内容行（章节号 + 标题 + 连续点 + 页码）。

        匹配形如：
            4 力学环境试验条件.. 6
            3.2 星上仪器设备的试验顺序.. .12
            6.2.2 太空计算标准体系 ..... 56
            3.7 试验设备要求. .. 16

        特征：行首是章节号（1 / 1.1 / 1.1.1），行末是页码（数字），
        中间含 2 个或更多连续点（.... 或 . .. 等）作为页码引导。

        只删整行，不动普通段落（行内引用如"见表 3.5"不匹配）。
        """
        text = cls._TOC_LINE_PATTERN.sub("", text)
        # 清理删除后的多余空行
        text = cls._clean_extra_spaces(text)
        return text


# ── 便捷函数 ──────────────────────────────────────────────


def clean_text(
    text: str,
    remove_images: bool = True,
) -> str:
    """
    便捷函数：清洗文本。

    Args:
        text: 原始文本
        remove_images: 是否移除 Markdown 图片

    Returns:
        清洗后的文本
    """
    return TextCleaner.clean(
        text,
        remove_images=remove_images,
    )
