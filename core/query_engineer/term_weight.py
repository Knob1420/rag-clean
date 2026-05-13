"""
词权重分类器 — 3 档权重：HIGH / MEDIUM / LOW

- HIGH (3.0): 产品名/型号名 — 从 terms_seed.json 的 key+value 自动推导
- MEDIUM (1.5): 名词性词汇 — jieba 词性标注自动判定（nr/ns/nt/nz/n/nw 等）
- LOW (0.5): 其他通用词（虚词/动词/形容词等）
"""

import json
import logging
from pathlib import Path

import jieba.posseg as pseg

# ── 权重常量 ──────────────────────────────────────────────────

WEIGHT_HIGH = 3.0
WEIGHT_MEDIUM = 1.5
WEIGHT_LOW = 0.5

# ── 名词性词性标签 ────────────────────────────────────────────

_NOUN_FLAGS = frozenset(
    {
        "n",  # 名词
        "nr",  # 人名
        "ns",  # 地名
        "nt",  # 机构名
        "nz",  # 其他专名
        "nw",  # 新词
        "ng",  # 名语素
        "j",  # 简称
        "l",  # 习用语
        "i",  # 成语
    }
)

# 非 MEDIUM 词性黑名单（keyword_extractor 停用词已过滤常见动词/虚词，
# 到 classify() 的 v 标签词基本都是"散热""接口"这类属性词，不应排除）
_NON_MEDIUM_FLAGS = frozenset(
    {
        "vn",  # 名动词（如"合作"）
        "vd",  # 副动词
        "ad",  # 副形词
        "an",  # 名形词
        "d",  # 副词
        "c",  # 连词
        "p",  # 介词
        "u",  # 助词
        "r",  # 代词
        "m",  # 数词
        "q",  # 量词
        "f",  # 方位词
        "b",  # 区别词
        "e",  # 叹词
        "o",  # 拟声词
        "h",  # 前缀
        "k",  # 后缀
        "x",  # 非语素字
        "y",  # 语气词
    }
)


class TermWeighter:
    """3 档词权重分类器

    MEDIUM 档基于 jieba 词性标注自动判定：名词 → MEDIUM，其余 → LOW。
    零手动维护。
    """

    def __init__(self):
        self._high_terms: set[str] = set()
        self._load_high_terms()

    def _load_high_terms(self):
        """从 terms_seed.json 加载产品名/型号名到 HIGH 档。

        terms_seed.json 结构: {"别名": "标准名"}
        所有 key 和 value 都是产品名/型号名，应归入 HIGH 档。
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / "data" / "terms_seed.json"
        try:
            with open(path, encoding="utf-8") as f:
                terms = json.load(f)
        except Exception:
            logging.warning("Failed to load terms_seed.json for TermWeighter")
            return

        for key, value in terms.items():
            self._high_terms.add(key.lower())
            if isinstance(value, str):
                self._high_terms.add(value.lower())

    @staticmethod
    def _is_noun(word: str) -> bool:
        """判断词是否为名词性词汇。

        1. jieba 词性标注：n/nr/ns/nt/nz/j/l/i → 名词
        2. 兜底：2 字及以上纯中文词，jieba 虽然标了动词等但实际可能是领域属性词
           （如"散热""接口"被标 v）。但 vn/vn/ad 等明确非名词的黑名单排除。
        """
        flags = [flag for _, flag in pseg.cut(word)]
        # 有任何名词标签 → 是名词
        if any(f in _NOUN_FLAGS for f in flags):
            return True
        # 兜底：2 字+纯中文，且没有被标为明确的非名词词性
        if len(word) >= 2 and all("\u4e00" <= ch <= "\u9fa5" for ch in word):
            if not any(f in _NON_MEDIUM_FLAGS for f in flags):
                return True
        return False

    def classify(self, word: str) -> float:
        """返回词的权重"""
        w = word.lower()
        if w in self._high_terms:
            return WEIGHT_HIGH
        if self._is_noun(word):
            return WEIGHT_MEDIUM
        return WEIGHT_LOW

    def classify_keywords(self, keywords: list[str]) -> list[tuple[str, float]]:
        """批量分类，返回 (keyword, weight) 元组列表"""
        return [(kw, self.classify(kw)) for kw in keywords]


# ── 全局实例 ──────────────────────────────────────────────────

_weighter: TermWeighter | None = None


def get_term_weighter() -> TermWeighter:
    """获取词权重分类器单例"""
    global _weighter
    if _weighter is None:
        _weighter = TermWeighter()
    return _weighter
