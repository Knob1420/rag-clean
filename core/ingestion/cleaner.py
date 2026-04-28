"""
文本清洗器 — 对标 Dify CleanProcessor

将原始文本清洗为干净的可分块文本。

清洗规则：
1. XML/模板符号清洗（默认）
2. 控制字符清洗（默认）
3. 多余空格/换行清洗（可选）
4. URL/邮箱移除（可选）
5. Markdown 图片移除（可选）
6. 产品参数表格移除（从 products_specs.json 匹配参数字段）
"""

import json
import re
from pathlib import Path

# ── 产品参数配置（从 products_specs.json 加载）───────────────


def _load_product_specs():
    """从 products_specs.json 加载产品参数字段名和产品名"""
    specs_path = Path(__file__).parent.parent.parent / "data" / "products_specs.json"
    if not specs_path.exists():
        return set(), set()

    try:
        with open(specs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 所有产品名
        product_names: set[str] = set()

        for products in data.values():
            if isinstance(products, dict):
                for product_name, params in products.items():
                    if isinstance(params, dict):
                        product_names.add(product_name)
        return product_names
    except Exception:
        return set()


_PRODUCT_NAMES = _load_product_specs()

# 核心参数字段（用于快速判断）
_CORE_PARAM_FIELDS = {
    "重量",
    "功耗",
    "算力",
    "尺寸",
    "内存",
    "存储",
    "对外接口",
    "工作温度",
    "储存温度",
    "输入电压",
    "设计寿命",
    "在轨情况",
}


def _is_product_spec_table(table_content: str) -> bool:
    """
    判断 markdown table 是否为产品参数表。

    判断逻辑（满足任一即删除）：
    1. 包含已知产品名
    2. 包含多个核心参数字段
    """
    content = table_content.lower()

    # 检查是否包含已知产品名
    for name in _PRODUCT_NAMES:
        if name.lower() in content:
            return True

    # 检查包含多少核心参数字段
    field_count = sum(1 for f in _CORE_PARAM_FIELDS if f.lower() in content)
    if field_count >= 2:
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
    def clean(cls, text: str, remove_images: bool = True, dataset_id: str = "") -> str:
        """
        清洗文本（默认执行所有清洗规则）。

        清洗规则：
        1. XML/模板符号清洗
        2. 控制字符清洗
        3. 多余空格/换行清洗
        4. URL/邮箱移除（保护 Markdown 链接/图片）
        5. Markdown 图片和 HTML img 标签移除（可选，同时移除紧跟的图例行）
        6. 产品参数表格移除（仅当 dataset_id 命中产品数据集时生效）
        7. 页眉页脚清洗（页码、原理图标记、日期、文档编号等）

        Args:
            text: 原始文本
            remove_images: 是否移除 Markdown 图片和 HTML img 标签，默认 True
            dataset_id: 数据集 ID，用于判断是否属于产品参数类知识库

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

        # 4. URL/邮箱移除
        text = cls._remove_urls_and_emails(text)

        # 5. 图片移除（可选）
        if remove_images:
            text = cls._remove_images(text)

        # 6. 产品参数表格移除（仅产品数据集）
        # text = cls._remove_product_spec_tables(text, dataset_id)

        # 7. 页眉页脚清洗
        text = cls._clean_headers_footers(text)

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
        移除 URL 和邮箱，但保护 Markdown 链接和图片。

        流程：
        1. 保护 Markdown 链接: [text](url) → __MD_LINK_0__
        2. 保护 Markdown 图片: ![alt](url) → __MD_IMAGE_0__
        3. 移除邮箱
        4. 移除普通 URL
        5. 恢复 Markdown 内容
        """
        # 保护 Markdown 链接和图片
        links: list[tuple[str, str]] = []

        def protect_link(match: re.Match) -> str:
            link_text = match.group(1)
            url = match.group(2)
            placeholder = f"__MD_LINK_{len(links)}__"
            links.append(("link", link_text, url, placeholder))
            return placeholder

        def protect_image(match: re.Match) -> str:
            alt_text = match.group(0)
            url_match = re.search(r"\((https?://[^)]+)\)", alt_text)
            if url_match:
                url = url_match.group(1)
                placeholder = f"__MD_IMAGE_{len(links)}__"
                links.append(("image", alt_text, url, placeholder))
                return placeholder
            return match.group(0)

        text = cls._MD_LINK_PATTERN.sub(protect_link, text)
        text = cls._MD_IMAGE_PATTERN.sub(protect_image, text)

        # 移除邮箱
        text = cls._EMAIL_PATTERN.sub("", text)

        # 移除普通 URL
        text = cls._URL_PATTERN.sub("", text)

        # 恢复 Markdown 链接和图片
        for link_type, content, url, placeholder in links:
            if link_type == "link":
                restored = f"[{content}]({url})"
            else:
                restored = f"![{content}]({url})"
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
    def _remove_product_spec_tables(cls, text: str, dataset_id: str = "") -> str:
        """
        移除产品参数表格（Markdown 和 HTML 格式）。

        仅当 dataset_id 命中以下条件时生效：
        - dataset_id 包含"产品"（products, 产品参数 等）
        - 或 dataset_id 为已知产品类别名
        否则直接返回原文本，不做任何处理。
        """
        # 产品数据集标识（可按需扩展）
        PRODUCT_DATASET_KEYWORDS = [
            "产品",
            "products",
            "specs",
            "星载智算机",
            "星载路由器",
            "星载激光通信机",
            "智能计算机",
            "激光通信",
        ]
        is_product_dataset = any(kw in dataset_id for kw in PRODUCT_DATASET_KEYWORDS)
        if not is_product_dataset:
            return text
        # 1. 移除 Markdown 格式 table
        md_table_pattern = re.compile(
            r"(?:\|[^\n]+\|\n){2,}(?:\|[^\n]+\|)", re.MULTILINE
        )

        def replace_md_table(match: re.Match) -> str:
            table_content = match.group(0)
            if _is_product_spec_table(table_content):
                return ""
            return table_content

        text = md_table_pattern.sub(replace_md_table, text)

        # 2. 移除 HTML 格式 table
        html_table_pattern = re.compile(
            r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE
        )

        def replace_html_table(match: re.Match) -> str:
            table_content = match.group(0)
            if _is_product_spec_table(table_content):
                return ""
            return table_content

        text = html_table_pattern.sub(replace_html_table, text)

        return text

    # 页眉页脚正则（类方法，方便子类扩展）
    _HEADER_FOOTER_PATTERNS: list[re.Pattern] = [
        # 页码: 第X页, Page X, - 1 -, 1/3
        re.compile(r"第\s*\d+\s*页"),
        re.compile(r"Page\s*\d+", re.IGNORECASE),
        re.compile(r"-\s*\d+\s*-"),
        re.compile(r"\b\d+\s*/\s*\d+\b"),
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
        # 表格分页残留（如 "续表"）
        re.compile(r"^续表\s*$", re.MULTILINE),
    ]

    @classmethod
    def _clean_headers_footers(cls, text: str) -> str:
        """移除页眉页脚中的常见标识（页码、原理图标记、日期、文档编号等）"""
        for pattern in cls._HEADER_FOOTER_PATTERNS:
            text = pattern.sub("", text)
        # 清理替换后产生的多余空白
        text = cls._clean_extra_spaces(text)
        return text


# ── 便捷函数 ──────────────────────────────────────────────


def clean_text(
    text: str,
    remove_images: bool = True,
    dataset_id: str = "",
) -> str:
    """
    便捷函数：清洗文本。

    Args:
        text: 原始文本
        remove_images: 是否移除 Markdown 图片
        dataset_id: 数据集 ID（仅产品类知识库触发产品参数表格删除）

    Returns:
        清洗后的文本
    """
    return TextCleaner.clean(
        text,
        remove_images=remove_images,
        dataset_id=dataset_id,
    )
