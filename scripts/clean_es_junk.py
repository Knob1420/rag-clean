"""
清理 ES 索引中的低质量 HTML 碎片 chunk

用法：
    # 干跑模式（只统计不删除）
    python scripts/clean_es_junk.py --dry-run

    # 实际删除碎片
    python scripts/clean_es_junk.py --clean
"""

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from store import get_store, DocumentStore

from core.ingestion.cleaner import is_low_quality_content, _HTML_TAG_RE


def clean_junk_chunks(store: DocumentStore, index_name: str, dry_run: bool = True):
    """扫描并删除 HTML 碎片 chunk"""
    es = store.es

    total_scanned = 0
    junk_count = 0
    junk_ids = []

    query = {
        "query": {"wildcard": {"chunk_id": "*_c*"}},
        "_source": ["chunk_id", "content"],
        "size": 500,
    }

    resp = es.search(index=index_name, body=query, scroll="5m")
    scroll_id = resp.get("_scroll_id")
    hits = resp["hits"]["hits"]

    while hits:
        for hit in hits:
            total_scanned += 1
            content = hit["_source"].get("content", "")
            chunk_id = hit["_source"].get("chunk_id", hit["_id"])

            if is_low_quality_content(content):
                junk_count += 1
                junk_ids.append(hit["_id"])
                if junk_count <= 10:
                    plain = _HTML_TAG_RE.sub("", content).strip()[:80]
                    logger.info(f"  碎片: {chunk_id} | plain: '{plain}'")

        resp = es.scroll(scroll_id=scroll_id, scroll="5m")
        scroll_id = resp.get("_scroll_id")
        hits = resp["hits"]["hits"]

    try:
        es.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass

    logger.info(f"扫描完成: {total_scanned} child chunks, {junk_count} 碎片 ({junk_count*100/max(total_scanned,1):.1f}%)")

    if not dry_run and junk_ids:
        batch_size = 500
        deleted = 0
        for i in range(0, len(junk_ids), batch_size):
            batch = junk_ids[i : i + batch_size]
            try:
                es.delete_by_query(index=index_name, body={"query": {"terms": {"_id": batch}}})
                deleted += len(batch)
                logger.info(f"  已删除 {deleted}/{len(junk_ids)}")
            except Exception as e:
                logger.error(f"  批量删除失败: {e}")

        # 刷新索引
        try:
            es.indices.refresh(index=index_name)
        except Exception:
            pass
        logger.info(f"清理完成: 删除 {deleted} 个碎片 chunk")
    elif dry_run and junk_ids:
        logger.info("干跑模式: 未删除。使用 --clean 执行实际删除。")

    return junk_count


def main():
    parser = argparse.ArgumentParser(description="清理 ES 索引中的 HTML 碎片 chunk")
    parser.add_argument("--dry-run", action="store_true", help="干跑模式")
    parser.add_argument("--clean", action="store_true", help="实际删除碎片")
    parser.add_argument("--index", type=str, default="rag_chunk_0506", help="ES 索引名")
    args = parser.parse_args()

    store = get_store()
    clean_junk_chunks(store, args.index, dry_run=not args.clean)


if __name__ == "__main__":
    main()
