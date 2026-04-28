"""
步骤1增强 — 文本清洗 + 术语归一

扩展 core/ingestion/cleaner.py：
1. 继承 TextCleaner 所有清洗规则
2. 新增 _normalize_terms() 术语归一
3. 新增 MD 表格类型识别（table_type）
"""

import json
import re
from pathlib import Path
from typing import Optional

from core.ingestion.cleaner import TextCleaner as BaseTextCleaner

# ── 术语白词配置 ─────────────────────────────────────────────


def _load_terms_seed() -> dict[str, str]:
    """
    加载 terms_seed.json，返回 {原词: 标准词} 映射。

    排序规则（确保单次遍历正确替换）：
    1. 先按长度降序（先匹配长词）
    2. 长度相同时，标准名优先于别名（避免别名中的子串被二次替换）
       例：NX系列(标准名) 和 NX1(别名) 长度相同时，NX系列 先匹配
    """
    terms_path = Path(__file__).parent / "scripts" / "terms_seed.json"
    if not terms_path.exists():
        return {}
    with open(terms_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def sort_key(item):
        original, standard = item
        # 长度降序；长度相同时，标准名（即 original == standard）优先
        is_standard = (original == standard)
        return (len(original), is_standard)

    return dict(sorted(data.items(), key=sort_key, reverse=True))


_TERMS_SEED = _load_terms_seed()

# ── 表格类型识别 ─────────────────────────────────────────────

# 表头关键词 → 表格类型映射
_TABLE_TYPE_KEYWORDS = {
    "参数表": ["型号", "重量", "算力", "功耗", "接口", "尺寸", "存储", "内存", "GPU", "工作温度"],
    "合作表": ["合作单位", "地区", "合作内容", "合作模式", "预期成果", "承担单位"],
    "对比表": ["对比", "比较", "规格对比", "型号对比"],
    "指标表": ["指标", "性能指标", "技术指标", "试验指标"],
}


def _detect_table_type(table_md: str) -> Optional[str]:
    """
    根据表头关键词判断 MD 表格类型。

    Args:
        table_md: Markdown 表格文本

    Returns:
        "参数表" / "合作表" / "对比表" / "指标表" / None
    """
    # 提取表头行（第一行 |xxx|xxx|...）
    lines = table_md.strip().split("\n")
    if not lines:
        return None

    # 合并前几行作为检测上下文
    header_context = "".join(lines[:3]).lower()

    for table_type, keywords in _TABLE_TYPE_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw.lower() in header_context)
        if matches >= 2:
            return table_type

    return None


# ── 扩展的 TextCleaner ────────────────────────────────────────


class TextCleaner(BaseTextCleaner):
    """
    步骤1增强版文本清洗器。

    新增能力：
    - _normalize_terms(): 术语归一
    - _detect_table_type(): 表格类型识别
    - 表格单独成片时元数据标记 table_type
    """

    @classmethod
    def clean(cls, text: str, remove_images: bool = True, dataset_id: str = "") -> str:
        """
        清洗文本，在父类清洗后执行术语归一。

        Args:
            text: 原始文本
            remove_images: 是否移除图片（继承父类行为）
            dataset_id: 数据集 ID（继承父类行为）

        Returns:
            清洗 + 术语归一后的文本
        """
        # 1. 父类清洗（XML/控制字符/多余空格/URL/图片/页眉页脚）
        text = super().clean(text, remove_images=remove_images, dataset_id=dataset_id)

        # 2. 术语归一
        text = cls._normalize_terms(text)

        return text

    @classmethod
    def _normalize_terms(cls, text: str) -> str:
        """
        全文档术语归一：将所有原词替换为标准写法。

        实现：手动从左到右扫描，每次取最长匹配。
        关键：把标准名（standard form）也加入匹配列表，优先级最低，
        确保"智加NX1"在"之江智加NX1"中被匹配到（但替换为自己，不变）。
        """
        if not _TERMS_SEED:
            return text

        # 构建匹配列表：标准名优先级最低（放最后），alias 优先级高
        # 每个元素: (original_text, replacement_text, priority)
        # priority: 0=standard(自身), 1=alias
        terms_with_priority: list[tuple[str, str, int]] = []
        for original, standard in _TERMS_SEED.items():
            is_alias = (original != standard)
            priority = 1 if is_alias else 0  # alias 优先
            terms_with_priority.append((original, standard, priority))

        # 把所有 VALUE（标准名）也加入，避免漏掉完整标准名的匹配
        # 例如 "智加NX1" 是 VALUE 但不是 KEY，需要能匹配到（替换为自身，不变）
        existing = {o for o, _, _ in terms_with_priority}
        for original, standard in _TERMS_SEED.items():
            if standard not in existing:
                terms_with_priority.append((standard, standard, 0))  # priority=0 最低
                existing.add(standard)

        # 按长度降序排列；长度相同时 alias 优先于 standard
        terms_with_priority.sort(key=lambda x: (len(x[0]), x[2]), reverse=True)

        # 构建 OR 正则
        escaped = [re.escape(orig) for orig, _, _ in terms_with_priority]
        pattern = "|".join(escaped)

        def find_longest_match(pos: int) -> tuple[int, str] | None:
            """
            从 text[pos] 开始，找最长匹配。
            返回 (matched_len, replacement) 或 None。
            """
            longest = None
            for orig, std, _ in terms_with_priority:
                if text[pos:pos + len(orig)].lower() == orig.lower():
                    if longest is None or len(orig) > longest[0]:
                        longest = (len(orig), std)
            return longest

        # 手动从左到右扫描
        result_parts: list[str] = []
        pos = 0
        while pos < len(text):
            match = find_longest_match(pos)
            if match is None:
                result_parts.append(text[pos])
                pos += 1
            else:
                matched_len, replacement = match
                result_parts.append(replacement)
                pos += matched_len

        return "".join(result_parts)

    @classmethod
    def detect_table_type(cls, table_md: str) -> Optional[str]:
        """判断表格类型（供 chunker_ext.py 调用）"""
        return _detect_table_type(table_md)


# ── 便捷函数 ─────────────────────────────────────────────────


def clean_and_normalize(text: str, remove_images: bool = True, dataset_id: str = "") -> str:
    """
    便捷函数：清洗文本 + 术语归一。

    Args:
        text: 原始文本
        remove_images: 是否移除图片
        dataset_id: 数据集 ID

    Returns:
        清洗后的文本
    """
    return TextCleaner.clean(text, remove_images=remove_images, dataset_id=dataset_id)
