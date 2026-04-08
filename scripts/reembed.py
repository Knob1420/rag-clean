#!/usr/bin/env python3
"""
重建向量索引脚本

将 ES 索引从 512 维向量 (bge-small-zh-v1.5) 迁移到 1024 维 (bge-m3)。

流程：
1. 创建新 index（1024 维 mapping）
2. 遍历旧 index 的所有 chunks，重新向量化后写入新 index
3. 完成

策略：
- delete + reindex 无效（ES 不允许修改已存在 index 的 dims）
- 必须在创建新 index 时指定正确的 dims
"""

import sys
import time
import re
from typing import Any, Dict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from elasticsearch.helpers import bulk
from loguru import logger
from tqdm import tqdm

from config import settings
from core.retrieve.embedder import encode_batch
from store import get_store, CHUNKS_MAPPING


def _fix_mapping_dims(mapping: dict, new_dims: int) -> dict:
    """将 mapping 中的 embedding_vector dims 替换为新值"""
    import copy
    m = copy.deepcopy(mapping)
    if "properties" in m and "embedding_vector" in m["properties"]:
        m["properties"]["embedding_vector"]["dims"] = new_dims
    return m


def _create_new_index(store, base_name: str, new_dims: int) -> str:
    """创建新的 index，返回新 index 名称"""
    # 清理非法字符，生成新 index 名
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)
    new_index = f"{safe_name}_v{new_dims}d"

    if store.es.indices.exists(index=new_index):
        print(f"⚠️  新索引已存在: {new_index}")
        confirm = input("  覆盖已有索引？(y/N): ").strip().lower()
        if confirm != "y":
            print("取消")
            sys.exit(0)
        store.es.indices.delete(index=new_index)
        print(f"  已删除旧索引: {new_index}")

    new_mapping = _fix_mapping_dims(CHUNKS_MAPPING, new_dims)
    store.es.indices.create(index=new_index, mappings=new_mapping)
    print(f"✅ 创建新索引: {new_index}（dims={new_dims}）")
    return new_index


def reindex_all(batch_size: int = 16, dry_run: bool = False, new_index: str = None):
    """
    批量重新向量化并重建索引

    Args:
        batch_size: 每批向量化数量
        dry_run: True 则只统计数量，不实际写入
        new_index: 指定新索引名称，若已存在则跳过创建
    """
    store = get_store()
    old_index = settings.es_index_chunks
    new_dims = settings.embedding_dim

    # 1. 统计总数
    total = store.es.count(index=old_index)["count"]
    logger.info(f"待处理 chunks 总数: {total}")
    print(f"\n源索引: {old_index}")
    print(f"向量维度: {new_dims}")
    print(f"批次大小: {batch_size}")

    if dry_run:
        print(f"\n[DRY RUN] 共 {total} 个 chunks 需要重新向量化并写入新索引")
        return

    # 2. 创建新索引（若已存在则复用）
    if new_index and store.es.indices.exists(index=new_index):
        print(f"✅ 复用已有索引: {new_index}")
    else:
        new_index = _create_new_index(store, old_index, new_dims)

    # 3. 遍历 + 向量化 + 写入新索引
    pit_id = store.es.open_point_in_time(
        index=old_index, keep_alive="10m"
    )["id"]

    success_count = 0
    fail_count = 0
    skip_count = 0
    pbar = tqdm(total=total, desc="重建索引", unit="chunk")

    # sort 字段用于 search_after 分页（ES 9.x 不支持 _doc sort，用 _score + _id 代替）
    search_body: Dict[str, Any] = {
        "size": batch_size,
        "pit": {"id": pit_id, "keep_alive": "10m"},
        "sort": [{"created_at": "asc"}, {"chunk_id": "asc"}],
    }

    try:
        while True:
            resp = store.es.search(body=search_body)

            hits = resp["hits"]["hits"]
            if not hits:
                break

            # 获取最后一条的 sort values，用于下一次 search_after
            last_sort = hits[-1]["sort"]
            search_body = {
                "size": batch_size,
                "pit": {"id": pit_id, "keep_alive": "10m"},
                "sort": [{"created_at": "asc"}, {"chunk_id": "asc"}],
                "search_after": last_sort,
            }

            # 提取 content
            ids = []
            contents = []
            raw_docs = []

            for hit in hits:
                chunk_id = hit["_id"]
                content = hit["_source"].get("content", "")
                if not content:
                    skip_count += 1
                    pbar.update(1)
                    continue
                ids.append(chunk_id)
                contents.append(content)
                raw_docs.append(hit["_source"])

            # 向量化
            t0 = time.time()
            vectors = encode_batch(contents)
            elapsed = time.time() - t0

            # 构建新文档
            index_docs = []
            for chunk_id, raw, vec in zip(ids, raw_docs, vectors):
                if vec is None:
                    fail_count += 1
                    pbar.update(1)
                    continue

                new_doc = dict(raw)
                new_doc["embedding_vector"] = vec.tolist()

                index_docs.append({
                    "_index": new_index,
                    "_id": chunk_id,
                    "_source": new_doc,
                })

            # 批量写入新索引
            if index_docs:
                ok, errors = bulk(
                    store.es, index_docs,
                    raise_on_error=False,
                    raise_on_exception=False,
                )
                success_count += ok
                if errors:
                    fail_count += len(errors)
                    for err in list(errors)[:2]:
                        logger.error(f"写入失败: {err}")

            pbar.update(len(hits))
            speed = len(hits) / elapsed if elapsed > 0 else 0
            pbar.set_postfix({"ok": success_count, "fail": fail_count, f"{speed:.0f}/s": ""})

    finally:
        try:
            store.es.close_point_in_time(body={"id": pit_id})
        except Exception:
            pass

    pbar.close()

    print(f"\n{'='*50}")
    print(f"索引重建完成!")
    print(f"  新索引: {new_index}")
    print(f"  写入成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  跳过(无内容): {skip_count}")
    print(f"  总计: {total}")
    print(f"{'='*50}")

    if success_count == total:
        print(f"\n✅ 所有 chunks 已迁移到新索引 {new_index}")
        print(f"   如需切换，可在 config.py 中修改 es_index_chunks = '{new_index}'")
        print(f"   旧索引 {old_index} 可手动删除")
    else:
        print(f"\n⚠️  部分 chunks 迁移失败，请检查后重试")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="重建向量索引")
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="每批向量化数量 (default: 16)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只统计数量，不实际写入"
    )
    parser.add_argument(
        "--new-index", type=str, default=None,
        help="新索引名称（默认自动生成）"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="跳过确认直接开始"
    )
    args = parser.parse_args()

    print(f"""
重建向量索引
============
当前配置:
  索引: {settings.es_index_chunks}
  新索引: {args.new_index or '(自动生成)'}
  Embedding: {settings.embedding_model}
  向量维度: {settings.embedding_dim}
批次大小: {args.batch_size}
模式: {'DRY RUN' if args.dry_run else '实际写入'}
    """)

    if not args.dry_run and not args.force:
        confirm = input("确认开始？（y/N）: ").strip().lower()
        if confirm != "y":
            print("取消")
            sys.exit(0)

    reindex_all(batch_size=args.batch_size, dry_run=args.dry_run, new_index=args.new_index)
