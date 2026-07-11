#!/usr/bin/env python
"""
诊断脚本 — 统计 ES chunks 索引中 child chunk 的长度分布 + 拉短样本。

用法:
    python scripts/diagnose_chunks.py
    python scripts/diagnose_chunks.py --short-max 120 --sample 30
    python scripts/diagnose_chunks.py --dataset my_kb            # 只看某个数据集
"""
import argparse
import sys
from pathlib import Path
from collections import Counter

# 项目根加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from elasticsearch import Elasticsearch

from config import settings


def percentiles(values, qs=(10, 25, 50, 75, 90, 95, 99)):
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {q: float(np.percentile(arr, q)) for q in qs}


def fetch_all_lengths(es: Elasticsearch, dataset_id: str | None):
    """扫描所有 child chunk 的 content 长度（用 scroll 避免大数据集超限）"""
    body = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"chunk_type": "child"}},
                    {"term": {"is_latest": True}},
                ]
            }
        },
        "_source": ["content", "doc_title", "dataset_id"],
        "size": 2000,
    }
    if dataset_id:
        body["query"]["bool"]["filter"].append({"term": {"dataset_id": dataset_id}})

    resp = es.search(index=settings.es_index_chunks, body=body, scroll="2m")
    scroll_id = resp.get("_scroll_id")
    hits = resp["hits"]["hits"]

    lengths = []
    samples_short = []  # (length, doc_title, content)
    samples_all = []

    while hits:
        for h in hits:
            src = h["_source"]
            content = src.get("content", "") or ""
            n = len(content)
            lengths.append(n)
            samples_all.append((n, src.get("doc_title", ""), content))
        hits = es.scroll(scroll_id=scroll_id, scroll="2m")["hits"]["hits"]

    es.clear_scroll(scroll_id=scroll_id)
    return lengths, samples_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="过滤 dataset_id")
    ap.add_argument("--short-max", type=int, default=120, help="短 chunk 阈值（字符）")
    ap.add_argument("--sample", type=int, default=30, help="短样本展示条数")
    args = ap.parse_args()

    es = Elasticsearch([settings.es_url])
    if not es.indices.exists(index=settings.es_index_chunks):
        print(f"索引不存在: {settings.es_index_chunks}")
        return

    print(f"\n=== 扫描 chunks 索引: {settings.es_index_chunks} ===")
    if args.dataset:
        print(f"过滤 dataset_id = {args.dataset}")

    lengths, samples = fetch_all_lengths(es, args.dataset)
    if not lengths:
        print("没有 child chunk 数据")
        return

    # ── 1. 长度分布 ─────────────────────────────────────────
    pct = percentiles(lengths)
    print(f"\n=== 长度分布（字符数）N={len(lengths)} ===")
    print(f"  min={min(lengths)}  max={max(lengths)}  mean={sum(lengths)/len(lengths):.1f}")
    for q, v in pct.items():
        print(f"  P{q:>2} = {v:>7.1f}")

    # ── 2. 区间统计 ─────────────────────────────────────────
    bins = [(0, 30), (30, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 10**9)]
    print(f"\n=== 区间分布 ===")
    for lo, hi in bins:
        cnt = sum(1 for L in lengths if lo <= L < hi)
        bar = "#" * min(50, cnt * 50 // max(lengths))
        label = f"{lo}-{hi if hi < 10**9 else '∞'}"
        print(f"  {label:>10}: {cnt:>6}  ({cnt*100/len(lengths):>5.1f}%) {bar}")

    # ── 3. 短 chunk 样本 ─────────────────────────────────────
    short = [(n, title, c) for n, title, c in samples if n <= args.short_max]
    short.sort(key=lambda x: x[0])
    print(f"\n=== 短 chunk 样本（≤ {args.short_max} 字符，共 {len(short)} 条 / {len(lengths)} = {len(short)*100/len(lengths):.1f}%） ===")
    for i, (n, title, c) in enumerate(short[:args.sample]):
        # 单行化便于阅读
        preview = c.replace("\n", "\\n")[:200]
        print(f"  [{i+1:>3}] len={n:>3} | {title[:30]:<30} | {preview}")

    # ── 4. 按文档统计短 chunk 占比 ─────────────────────────
    print(f"\n=== 各文档短 chunk 占比 TOP 10（≤ {args.short_max} 字符） ===")
    per_doc_total = Counter()
    per_doc_short = Counter()
    for n, title, _ in samples:
        per_doc_total[title] += 1
        if n <= args.short_max:
            per_doc_short[title] += 1
    rows = []
    for title, tot in per_doc_total.items():
        s = per_doc_short.get(title, 0)
        rows.append((title, s, tot, s * 100 / tot if tot else 0))
    rows.sort(key=lambda x: -x[3])
    print(f"  {'doc_title':<40} {'short':>6} {'total':>6} {'ratio':>6}")
    for title, s, tot, r in rows[:10]:
        print(f"  {title[:40]:<40} {s:>6} {tot:>6} {r:>5.1f}%")


if __name__ == "__main__":
    main()
