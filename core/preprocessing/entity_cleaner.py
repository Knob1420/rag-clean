"""
步骤2后处理 — entity_raw 清洗

5 级清洗链：
1. 小写归一 + 符号清洗
2. 长度过滤 + 停用词过滤（干掉垃圾）
3. 高频词保留 + 低频词删除（干掉无关）
4. 相似度聚类 + 别名合并（自动合并同义词）
5. 业务白名单/黑名单二次过滤（最终提纯）

输出：清洗后的 entity_raw.json
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# ── 停用词配置 ─────────────────────────────────────────────


# 中文停用词（常见无意义词/符号单独成词的情况）
_STOP_WORDS = {
    "的",
    "了",
    "在",
    "是",
    "我",
    "有",
    "和",
    "就",
    "不",
    "人",
    "都",
    "一",
    "一个",
    "上",
    "也",
    "很",
    "到",
    "说",
    "要",
    "去",
    "你",
    "会",
    "着",
    "没有",
    "看",
    "好",
    "自己",
    "这",
    "那",
    "但",
    "而",
    "与",
    "或",
    "以",
    "及",
    "等",
    "为",
    "对",
    "于",
    "之",
    "其",
    "所",
    "以",
    "因",
    "由",
    "从",
    "当",
    "中",
    "将",
    "把",
    "被",
    "让",
    "给",
    "向",
    "往",
    "朝",
    "在",
    "从",
    "自",
    "对",
    "为",
    "以",
    "因",
    "当",
    "把",
    # 英文停用词
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    # 符号/单字符
    "×",
    "×",
    "·",
    "•",
    "－",
    "–",
    "—",
    "~",
    "～",
    "＝",
    "=",
    "①",
    "②",
    "③",
    "④",
    "●",
    "○",
    "◎",
    "■",
    "□",
    "▲",
    "△",
    "▼",
    "▽",
    "◆",
    "◇",
    "★",
    "☆",
    "※",
    "§",
    "「",
    "」",
    "『",
    "』",
    "【",
    "】",
    "《",
    "》",
    "〈",
    "〉",
    "‖",
    "⌈",
    "⌋",
    "⌊",
    "⌋",
    "┌",
    "┐",
    "└",
    "┘",
    "├",
    "┤",
    "┬",
    "┴",
    "┼",
    # 常见无意义后缀
}


def _load_blacklist() -> set[str]:
    """加载业务黑名单。"""
    path = Path(__file__).parent / "scripts" / "entity_blacklist.json"
    if not path.exists():
        return set()
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("blacklist", []))


def _load_whitelist() -> set[str]:
    """加载业务白名单。"""
    path = Path(__file__).parent / "scripts" / "entity_whitelist.json"
    if not path.exists():
        return set()
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("whitelist", []))


_BLACKLIST = _load_blacklist()
_WHITELIST = _load_whitelist()


# ── 步骤1：小写归一 + 符号清洗 ─────────────────────────────


def _normalize_text(text: str) -> str:
    """
    文本归一化：
    1. 全角转半角
    2. 符号清洗（只保留中文/英文/数字）
    3. 转小写（用于相似度计算）
    """
    # 全角转半角
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        elif code == 0x3000:
            code = 0x0020
        result.append(chr(code))

    text = "".join(result)

    # 符号清洗：只保留中文、英文、数字
    text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf_a-zA-Z0-9]", "", text)

    return text


# ── 步骤2：长度过滤 + 停用词过滤 ──────────────────────────


_MIN_ENTITY_LEN = 2  # 小于等于 1 字符的过滤
_MAX_ENTITY_LEN = 30  # 超过 30 字符的过滤


def _is_stopword(name: str) -> bool:
    """判断是否是停用词。"""
    return name in _STOP_WORDS


def _passes_length_and_stopword(name: str) -> bool:
    """步骤2过滤器：长度 + 停用词。"""
    norm = _normalize_text(name)
    if len(norm) < _MIN_ENTITY_LEN or len(norm) > _MAX_ENTITY_LEN:
        return False
    if _is_stopword(name):
        return False
    return True


# ── 步骤3：频率过滤 ───────────────────────────────────────


_FREQ_LOW_THRESHOLD = 1  # 频率低于此值的删除


def _passes_frequency(entity: dict) -> bool:
    """
    步骤3过滤器：高频保保留，低频删除。
    白名单实体不受频率限制。
    """
    name = entity["entity_name"]
    freq = entity.get("frequency", 1)

    # 白名单不受频率限制
    if name in _WHITELIST:
        return True

    # 黑名单直接过滤
    if name in _BLACKLIST:
        return False

    return freq >= _FREQ_LOW_THRESHOLD


# ── 步骤4：相似度聚类 + 别名合并 ─────────────────────────


_SIM_THRESHOLD = 0.95  # Embedding 相似度阈值（提高防误合并）


def _get_embedder():
    """获取 embedding 工具。"""
    try:
        from core.client.embedder import encode_batch

        return encode_batch
    except ImportError:
        return None


def _levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串的编辑距离（Levenshtein distance）。"""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


def _should_merge_strings(name1: str, name2: str) -> bool:
    """
    确定性字符串合并规则。
    满足任一条件则视为同义词候选。
    """
    n1, n2 = _normalize_text(name1), _normalize_text(name2)

    # 规则1：归一化后相等
    if n1 == n2 and n1:
        return True

    # 规则2：子串关系（短名被长名包含）
    if len(n1) >= 2 and len(n2) >= 2:
        if n1 in n2 or n2 in n1:
            return True

    # 规则3：编辑距离 ≤ 1（适用于短名，长度 ≤ 5）
    if len(n1) <= 5 and len(n2) <= 5 and max(len(n1), len(n2)) >= 2:
        if _levenshtein_distance(n1, n2) <= 1:
            return True

    return False


def _compute_similarity_groups(
    entities: list[dict],
    embedder,
    sim_threshold: float = _SIM_THRESHOLD,
) -> list[list[dict]]:
    """
    基于确定性规则 + 可选 Embedding 相似度对实体进行聚类。

    优先使用字符串规则合并，再用 Embedding 对剩余实体做二次合并。

    Returns:
        [[entity, entity, ...], [entity, ...], ...]
    """
    n = len(entities)
    if n == 0:
        return []

    # ── 阶段1：确定性字符串规则合并 ─────────────────────────
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if _should_merge_strings(entities[i]["entity_name"], entities[j]["entity_name"]):
                union(i, j)

    # ── 阶段2：Embedding 相似度合并（对剩余未合并的）──────
    if embedder is not None:
        # 找出尚未合并的实体（即自成一组且组内 > 1 的跳过，只处理单独的）
        # 只对「频率 ≥ 3 且长度 ≥ 4」的实体计算 embedding
        names = [e["entity_name"] for e in entities]
        embeddings = embedder(names)
        if embeddings:
            vecs = np.array(embeddings)
            for i in range(n):
                for j in range(i + 1, n):
                    # 只有两者都在不同组时才合并
                    if find(i) != find(j):
                        # 只对长度 >= 4 的实体用 embedding
                        if len(entities[i]["entity_name"]) >= 4 and len(entities[j]["entity_name"]) >= 4:
                            v1, v2 = vecs[i], vecs[j]
                            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                            if norm1 > 0 and norm2 > 0:
                                sim = float(np.dot(v1, v2) / (norm1 * norm2))
                                if sim >= sim_threshold:
                                    union(i, j)

    # 按并查集根分组
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    return [[entities[i] for i in group] for group in groups.values()]


def _merge_entity_cluster(group: list[dict]) -> dict:
    """
    将同一聚类中的多个实体合并为一个。

    保留频次最高的实体名，其余作为 source_chunks 合并。
    """
    if len(group) == 1:
        return dict(group[0])

    # 找频次最高的实体（主名）
    primary = max(group, key=lambda e: e.get("frequency", 0))

    # 合并所有 source_chunks
    all_chunks: set[str] = set()
    all_docs: set[str] = set()
    for ent in group:
        all_chunks.update(ent.get("source_chunks", []))
        all_docs.update(ent.get("source_docs", []))

    return {
        "entity_name": primary["entity_name"],
        "entity_type": primary.get("entity_type", "UNKNOWN"),
        "frequency": primary.get("frequency", 1),
        "source_docs": list(all_docs),
        "source_chunks": list(all_chunks),
    }


# ── 步骤5：白名单/黑名单二次过滤 ──────────────────────────


def _apply_whitelist_filter(entities: list[dict]) -> list[dict]:
    """
    步骤5过滤器：白名单保保留，黑名单删除。
    """
    result = []
    for ent in entities:
        name = ent["entity_name"]
        if name in _BLACKLIST:
            continue
        result.append(ent)
    return result


# ── 主清洗流程 ───────────────────────────────────────────


def clean_entity_raw(
    entity_raw: list[dict],
    embedder=None,
    sim_threshold: float = _SIM_THRESHOLD,
    freq_low_threshold: int = _FREQ_LOW_THRESHOLD,
) -> tuple[list[dict], list[dict]]:
    """
    5 级清洗链：

    1. 小写归一 + 符号清洗（normalize，不改变原始 entity_name）
    2. 长度过滤 + 停用词过滤
    3. 频率过滤（高频保留，低频删除）
    4. 相似度聚类 + 别名合并（Embedding 语义相似度）
    5. 白名单/黑名单二次过滤

    Args:
        entity_raw: 原始 entity_raw 列表
        embedder: embedding 函数，默认使用 encode_batch
        sim_threshold: 相似度阈值，默认 0.85
        freq_low_threshold: 频率下限，默认 2

    Returns:
        (清洗后的 entity_raw 列表, 合并记录列表)
        合并记录: [{"primary": "主名", "merged": ["别名1", "别名2", ...]}, ...]
    """
    from loguru import logger

    logger.info(f"[EntityCleaner] 开始清洗，原始实体数: {len(entity_raw)}")

    # ── 步骤1&2：长度 + 停用词过滤 ─────────────────────────
    filtered = []
    for ent in entity_raw:
        name = ent.get("entity_name", "")
        if _passes_length_and_stopword(name):
            filtered.append(ent)

    logger.info(f"[EntityCleaner] 步骤1&2（长度+停用词）后: {len(filtered)}")

    # ── 步骤3：频率过滤 ────────────────────────────────────
    freq_filtered = []
    for ent in filtered:
        if _passes_frequency(ent):
            freq_filtered.append(ent)

    logger.info(f"[EntityCleaner] 步骤3（频率）后: {len(freq_filtered)}")

    # ── 步骤4：相似度聚类 + 别名合并 ───────────────────────
    if embedder is None:
        embedder = _get_embedder()

    merge_records: list[dict] = []

    if embedder is None:
        logger.warning("[EntityCleaner] embedder 不可用，跳过步骤4（相似度聚类）")
        clustered = freq_filtered
    else:
        groups = _compute_similarity_groups(freq_filtered, embedder, sim_threshold)
        clustered = [_merge_entity_cluster(g) for g in groups]

        # 记录合并
        for g in groups:
            if len(g) > 1:
                primary = max(g, key=lambda e: e.get("frequency", 0))
                merged = [e["entity_name"] for e in g if e["entity_name"] != primary["entity_name"]]
                merge_records.append({
                    "primary": primary["entity_name"],
                    "primary_type": primary.get("entity_type", "UNKNOWN"),
                    "merged": merged,
                    "merged_count": len(merged),
                })

        logger.info(
            f"[EntityCleaner] 步骤4（相似度聚类）后: {len(clustered)}，合并了 {len(freq_filtered) - len(clustered)} 个实体"
        )

    # ── 步骤5：白名单/黑名单二次过滤 ─────────────────────
    final = _apply_whitelist_filter(clustered)
    logger.info(f"[EntityCleaner] 步骤5（白/黑名单）后: {len(final)}")

    # 按频次降序排列
    final.sort(key=lambda e: e.get("frequency", 0), reverse=True)

    return final, merge_records


# ── 保存 ─────────────────────────────────────────────────


def save_cleaned(
    entities: list[dict],
    output_path: str,
    merge_records: Optional[list[dict]] = None,
) -> dict[str, str]:
    """
    保存清洗后的 entity_raw 及合并记录。

    Returns:
        {"entities": path, "merge_records": path}
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entities_path = output_path
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)

    result = {"entities": str(entities_path)}

    if merge_records:
        merge_path = output_path.parent / "step2_entity_merge_records.json"
        with open(merge_path, "w", encoding="utf-8") as f:
            json.dump(merge_records, f, ensure_ascii=False, indent=2)
        result["merge_records"] = str(merge_path)

    from loguru import logger

    logger.info(f"[EntityCleaner] 清洗结果已保存: {result}")
    return result
