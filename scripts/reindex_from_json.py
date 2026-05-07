#!/usr/bin/env python3
"""
从 processed-0506 目录直接读取 JSON 文件，重新索引到 ES

用法：
    cd /home/zjlab/Documents/build_LLMs/NLP_course_hf/RAG/rag-clean
    python scripts/reindex_from_json.py
    python scripts/reindex_from_json.py --limit 10  # 限制数量
    python scripts/reindex_from_json.py --dry-run
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from store import get_store
from core.model.models import Document

DATA_DIR = Path("/home/zjlab/Documents/build_LLMs/NLP_course_hf/RAG/data")
PROCESSED_DIR = DATA_DIR / "processed-0506"


def load_documents_from_json(json_path: Path) -> List[Document]:
    """从 JSON 文件加载 Document 列表"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [Document.from_dict(d) for d in data]


def main():
    parser = argparse.ArgumentParser(description="从 processed-0506 JSON 重建 ES 索引")
    parser.add_argument("--dry-run", action="store_true", help="只列出文件，不索引")
    parser.add_argument("--limit", type=int, default=0, help="限制数量（0=全部）")
    parser.add_argument("--dataset-id", type=str, default="", help="dataset_id（默认从目录结构推断）")
    args = parser.parse_args()

    # 收集所有 JSON 文件
    json_files = sorted(PROCESSED_DIR.glob("*.json"))
    if not json_files:
        print(f"未找到 JSON 文件: {PROCESSED_DIR}/*.json")
        return

    if args.limit > 0:
        json_files = json_files[:args.limit]

    print(f"找到 {len(json_files)} 个 JSON 文件")
    print(f"目录: {PROCESSED_DIR}")
    print(f"模式: {'dry-run' if args.dry_run else '索引'}")
    print()

    store = get_store()
    store.ensure_indices()

    success = 0
    failed = 0
    skipped = 0

    for json_file in json_files:
        title = json_file.stem
        print(f"[{'DRY-RUN' if args.dry_run else 'INDEX'}] {title}")

        if args.dry_run:
            continue

        try:
            documents = load_documents_from_json(json_file)
            if not documents:
                print(f"  -> 跳过（空文档）")
                skipped += 1
                continue

            # 从文件名推断 dataset_id（目录名）
            # JSON 文件在 processed-0506/{title}.json，dataset_id 从上一级推断
            dataset_id = json_file.parent.name  # = "processed-0506"
            for doc in documents:
                doc.metadata["dataset_id"] = dataset_id

            doc_id = documents[0].metadata.get("doc_id", title)
            store.index_document(doc_id, documents)
            print(f"  -> 索引 {len(documents)} documents")
            success += 1

        except Exception as e:
            print(f"  -> 失败: {e}")
            failed += 1

    print()
    print(f"完成: success={success}, failed={failed}, skipped={skipped}")


if __name__ == "__main__":
    main()
