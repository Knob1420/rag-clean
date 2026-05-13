#!/usr/bin/env python3
"""
BM25 query_string 生成调试脚本 — 输入查询语句，输出 Lucene query_string + 完整 ES DSL

用法:
    python scripts/dsl_debug.py "智算机的算力是多少？"
    python scripts/dsl_debug.py "智加G3支持什么接口" --synonym
    python scripts/dsl_debug.py --file scripts/queries.txt
    python scripts/dsl_debug.py "智算机的算力" --rerank
"""

import argparse
import json
import re
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.query_engineer.keyword_extractor import get_keyword_extractor
from core.query_engineer.synonym import get_synonym_lookup
from core.query_engineer.term_weight import WEIGHT_HIGH, WEIGHT_MEDIUM, WEIGHT_LOW

# ============================================================
# Lucene query_string 特殊字符转义
# ============================================================

_LUCENE_SPECIAL_CHARS = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\\/])')


def _sub_special_char(text: str) -> str:
    """转义 Lucene query_string 中的特殊字符"""
    return _LUCENE_SPECIAL_CHARS.sub(r'\\\1', text)


# ============================================================
# query_string 构建（与 retrieval.py _build_bm25_query 一致）
# ============================================================


def build_bm25_query_string(
    query: str,
    synonym_boost_factor: float = 0.2,
    bigram_boost_factor: float = 2.0,
    fulltext_boost: float = 5.0,
    syn_all_boost: float = 0.7,
) -> str:
    """构建 BM25 query_string

    参数:
        query: 原始查询字符串
        synonym_boost_factor: 同义词相对于原词的 boost 系数（默认 0.2）
        bigram_boost_factor: bigram 近邻相对于 max(w1,w2) 的倍数（默认 2.0）
        fulltext_boost: 整句加权的 boost（默认 5.0）
        syn_all_boost: 整句同义词组的 boost（默认 0.7）

    返回:
        Lucene query_string 语法字符串
    """
    # ── 关键词提取 + 权重 ──
    raw_keywords = get_keyword_extractor().extract(query)

    # ── 同义词归一化 ──
    synonym_lookup = get_synonym_lookup()
    norm_keywords = []
    word_synonyms: dict[str, list[str]] = {}
    existing_norm = set()

    for kw, weight in raw_keywords:
        std = synonym_lookup.normalize(kw)
        if std not in existing_norm:
            norm_keywords.append((std, weight))
            existing_norm.add(std)
            syns = synonym_lookup.get_synonyms(std)
            word_synonyms[std] = syns

    # ── 权重归一化（sum=1.0） ──
    total_weight = sum(w for _, w in norm_keywords)
    if total_weight > 0:
        norm_keywords = [(kw, w / total_weight) for kw, w in norm_keywords]

    tms_parts = []

    # ── 1. 每个原词 + 同义词 OR ──
    for kw, w in norm_keywords:
        tk = _sub_special_char(kw)
        if tk.find(" ") > 0:
            tk = f'"{tk}"'

        syns = word_synonyms.get(kw, [])
        if syns:
            syn_escaped = []
            for s in syns:
                s = _sub_special_char(s)
                if s.find(" ") > 0:
                    s = f'"{s}"'
                syn_escaped.append(s)
            tk = f"({tk} OR ({' '.join(syn_escaped)})^0.2)"

        if tk.strip():
            tms_parts.append(f"({tk})^{w:.4f}")

    # ── 2. bigram 近邻匹配（~2 允许中间插入 2 个词） ──
    for i in range(1, len(norm_keywords)):
        left, left_w = norm_keywords[i - 1]
        right, right_w = norm_keywords[i]
        if not left.strip() or not right.strip():
            continue
        bigram = f'"{_sub_special_char(left)} {_sub_special_char(right)}"~2'
        tms_parts.append(f"{bigram}^{max(left_w, right_w) * bigram_boost_factor:.4f}")

    # ── 3. 整句加权 + 4. 整句同义词 OR ──
    if len(norm_keywords) > 1:
        tms_core = f"({' '.join(tms_parts)})^{fulltext_boost}"

        # ── 4. 整句同义词 OR ──
        all_syns = []
        for kw, _ in norm_keywords:
            syns = word_synonyms.get(kw, [])
            all_syns.extend(syns)

        if all_syns:
            syns_str = " OR ".join(
                [f'"{_sub_special_char(s)}"' for s in all_syns]
            )
            tms_core = f"{tms_core} OR ({syns_str})^{syn_all_boost}"

        query_string = tms_core
    else:
        query_string = " ".join(tms_parts)

    # ── 5. 原始 query 兜底 ──
    escaped_query = _sub_special_char(query)
    query_string = f"{query_string} OR {escaped_query}"

    return query_string


def build_full_es_dsl(
    query: str,
    doc_ids: list[str] | None = None,
    dataset_ids: list[str] | None = None,
    top_k: int = 20,
) -> dict:
    """构建完整的 ES 查询 DSL（含 filter + query_string）"""
    query_string = build_bm25_query_string(query)

    filter_conditions = [
        {"term": {"is_latest": True}},
        {"terms": {"chunk_type": ["child", "summary"]}},
    ]
    if doc_ids:
        filter_conditions.append({"terms": {"doc_id": doc_ids}})
    if dataset_ids:
        filter_conditions.append({"terms": {"dataset_id": dataset_ids}})

    return {
        "query": {
            "bool": {
                "filter": filter_conditions,
                "must": [
                    {
                        "query_string": {
                            "query": query_string,
                            "fields": ["doc_title^3", "content"],
                            "type": "best_fields",
                            "minimum_should_match": 1,
                        }
                    }
                ],
            }
        },
        "size": top_k,
    }


def build_rerank_query(query: str) -> str:
    """构建 rerank 增强查询（关键词按权重重复）

    HIGH 词重复 3 次，MEDIUM 2 次，LOW 1 次
    """
    keywords = get_keyword_extractor().extract(query)
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


# ============================================================
# 格式化输出
# ============================================================


def weight_label(w: float) -> str:
    if w >= WEIGHT_HIGH:
        return "HIGH"
    elif w >= WEIGHT_MEDIUM:
        return "MEDIUM"
    else:
        return "LOW"


def print_analysis(query: str, show_synonym: bool = False, show_rerank: bool = False):
    """打印查询分析结果"""
    keywords = get_keyword_extractor().extract(query)
    synonym_lookup = get_synonym_lookup()

    # ── 关键词 ──
    print(f"\n{'='*60}")
    print(f"查询: {query}")
    print(f"{'='*60}")
    print(f"\n关键词 ({len(keywords)} 个):")
    for kw, w in keywords:
        label = weight_label(w)
        syns = synonym_lookup.get_synonyms(synonym_lookup.normalize(kw))
        syn_str = f"  同义词: {syns}" if syns else ""
        print(f"  {kw:<20s}  权重={w:<4.1f}  [{label}]{syn_str}")

    # ── 归一化后 ──
    norm_keywords = []
    word_synonyms: dict[str, list[str]] = {}
    existing = set()
    for kw, w in keywords:
        std = synonym_lookup.normalize(kw)
        if std not in existing:
            norm_keywords.append((std, w))
            existing.add(std)
            word_synonyms[std] = synonym_lookup.get_synonyms(std)

    # 归一化
    total_weight = sum(w for _, w in norm_keywords)
    if total_weight > 0:
        norm_keywords = [(kw, w / total_weight) for kw, w in norm_keywords]

    print(f"\n归一化关键词 ({len(norm_keywords)} 个):")
    for kw, w in norm_keywords:
        syns = word_synonyms.get(kw, [])
        syn_str = f"  同义词: {syns}" if syns else ""
        print(f"  {kw:<20s}  权重={w:<6.4f}{syn_str}")

    # ── query_string ──
    qs = build_bm25_query_string(query)
    print(f"\nquery_string:")
    print(f"  {qs}")

    # ── 完整 DSL ──
    dsl = build_full_es_dsl(query)
    print(f"\n完整 ES DSL:")
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    # ── 同义词扩展详情 ──
    if show_synonym:
        expanded = synonym_lookup.expand_with_synonyms(keywords, factor=0.5)
        print(f"\n同义词扩展 ({len(expanded)} 个):")
        for kw, w in expanded:
            print(f"  {kw:<20s}  权重={w:<4.2f}")

    # ── Rerank query ──
    if show_rerank:
        rq = build_rerank_query(query)
        print(f"\nRerank 查询:")
        print(f"  {rq}")


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="BM25 query_string 生成调试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/dsl_debug.py "智算机的算力是多少？"
  python scripts/dsl_debug.py "智加G3支持什么接口" --synonym --rerank
  python scripts/dsl_debug.py --file scripts/queries.txt
  python scripts/dsl_debug.py --file scripts/queries.txt --dsl-only
        """,
    )
    parser.add_argument("query", nargs="?", help="查询语句")
    parser.add_argument("--file", "-f", type=str, help="从文件读取查询（每行一条）")
    parser.add_argument("--synonym", "-s", action="store_true", help="显示同义词扩展详情")
    parser.add_argument("--rerank", "-r", action="store_true", help="显示 rerank 查询")
    parser.add_argument(
        "--dsl-only", action="store_true", help="仅输出 DSL JSON（不打印分析）"
    )

    args = parser.parse_args()

    if not args.query and not args.file:
        parser.print_help()
        sys.exit(1)

    queries = []
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
    elif args.query:
        queries = [args.query]

    if args.dsl_only:
        for q in queries:
            dsl = build_full_es_dsl(q)
            print(json.dumps({"query": q, "dsl": dsl}, ensure_ascii=False))
    else:
        for q in queries:
            print_analysis(q, show_synonym=args.synonym, show_rerank=args.rerank)


if __name__ == "__main__":
    main()
