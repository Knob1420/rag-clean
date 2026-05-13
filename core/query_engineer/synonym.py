"""
同义词查找与扩展

从 data/synonym.json 加载手动维护的同义词映射，提供：
- get_synonyms(word): 查找单个词的同义词
- expand_with_synonyms(keywords, factor): 别名替换为 key（标准名），其余别名作为同义词降权扩展
"""

import json
import logging
from pathlib import Path


class SynonymLookup:
    """同义词查找器

    synonym.json 结构: {"标准名(key)": ["别名1", "别名2", ...]}
    语义：别名 → 应替换为标准名；其余别名作为同义词用于扩展匹配。
    """

    def __init__(self):
        self._mapping: dict[str, list[str]] = {}   # key → [aliases]
        self._reverse: dict[str, str] = {}          # alias → key
        self._load()

    def _load(self):
        """从 data/synonym.json 加载同义词映射，同时构建反向索引"""
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / "data" / "synonym.json"
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            logging.warning("Failed to load synonym.json for SynonymLookup")
            return

        for key, aliases in raw.items():
            key_lower = key.lower()
            if isinstance(aliases, str):
                aliases = [aliases]
            self._mapping[key_lower] = [a.lower() for a in aliases]

            # 反向索引：别名 → key（标准名）
            for alias in aliases:
                alias_lower = alias.lower()
                self._reverse[alias_lower] = key_lower

    def normalize(self, word: str) -> str:
        """将别名替换为标准名，如果本身就是标准名则原样返回。"""
        w = word.lower()
        return self._reverse.get(w, w)

    def get_synonyms(self, word: str) -> list[str]:
        """查找词的同义词。

        - 如果 word 是 key（标准名）→ 返回其所有别名
        - 如果 word 是别名 → 返回 key + 其余别名（不含自身）
        """
        w = word.lower()
        result = set()

        # 正向：word 是标准名 → 返回别名列表
        if w in self._mapping:
            result.update(self._mapping[w])

        # 反向：word 是别名 → 返回标准名
        if w in self._reverse:
            result.add(self._reverse[w])

        result.discard(w)  # 排除自身
        return list(result)

    def expand_with_synonyms(
        self,
        keywords: list[tuple[str, float]],
        factor: float = 0.5,
    ) -> list[tuple[str, float]]:
        """扩展关键词列表：别名替换为标准名 + 同义词降权扩展。

        1. 别名 → 替换为 key（标准名），保留原权重
        2. 标准名的其余别名作为同义词，权重 = 原权重 × factor
        3. 去重
        """
        normalized = []
        existing = set()

        # 第一步：别名替换为标准名
        for kw, weight in keywords:
            std = self.normalize(kw)
            if std not in existing:
                normalized.append((std, weight))
                existing.add(std)

        # 第二步：对每个标准名，追加其别名作为同义词
        expanded = list(normalized)
        for std, weight in normalized:
            if std in self._mapping:
                for alias in self._mapping[std]:
                    if alias not in existing:
                        expanded.append((alias, weight * factor))
                        existing.add(alias)

        return expanded


# ── 全局实例 ──────────────────────────────────────────────────

_lookup: SynonymLookup | None = None


def get_synonym_lookup() -> SynonymLookup:
    """获取同义词查找器单例"""
    global _lookup
    if _lookup is None:
        _lookup = SynonymLookup()
    return _lookup
