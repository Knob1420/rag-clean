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
from typing import Optional


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

    # 图片正则
    _IMG_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img\s[^>]*/?>", re.IGNORECASE)

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
        4. URL/邮箱移除（保护 Markdown 链接/图片）
        5. Markdown 图片移除（可选）

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

        # 4. URL/邮箱移除
        text = cls._remove_urls_and_emails(text)

        # 5. 图片移除（可选）
        if remove_images:
            text = cls._remove_images(text)

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
        """移除 Markdown 图片和 HTML img 标签"""
        return cls._IMG_PATTERN.sub("", text)


# ── 便捷函数 ──────────────────────────────────────────────


def clean_text(
    text: str,
    remove_images: bool = True,
) -> str:
    """
    便捷函数：清洗文本。

    Args:
        text: 原始文本
        remove_urls: 是否移除 URL 和邮箱（暂未实现，保留接口）
        remove_extra_spaces: 是否清理多余空格/换行（暂未实现，保留接口）
        remove_images: 是否移除 Markdown 图片

    Returns:
        清洗后的文本
    """
    return TextCleaner.clean(
        text,
        remove_images=remove_images,
    )
