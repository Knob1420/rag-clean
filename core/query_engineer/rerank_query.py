"""
增强 Rerank Query 构建 — 关键词按权重重复

将原始 query 通过关键词提取 + 权重重复合并，构建增强版 rerank query，
使 rerank 模型更关注高权重关键词。
"""

from core.query_engineer.keyword_extractor import get_keyword_extractor
from core.query_engineer.term_weight import WEIGHT_HIGH, WEIGHT_MEDIUM


def build_rerank_query(query: str) -> str:
    """构建增强 rerank query：关键词按权重重复

    Args:
        query: 原始查询字符串

    Returns:
        增强后的 rerank query（如 "G1功耗 关键词: G1 G1 G1 功耗 功耗"）
    """
    keywords = get_keyword_extractor().extract(query)
    if keywords:
        repeated = []
        for kw, weight in keywords:
            if weight >= WEIGHT_HIGH:
                repeat = 3
            elif weight >= WEIGHT_MEDIUM:
                repeat = 2
            else:
                repeat = 1
            repeated.extend([kw] * repeat)
        return f"{query} 关键词: {' '.join(repeated)}"
    return query
