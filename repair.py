"""
修复脚本 — 不重跑 pipeline，只做：
1. 重命名 imported 文件：{doc_id}.json → {title}__{doc_id}.json
2. 从 checkpoint 重新索引到 ES（修复之前因嵌套 list 失败的 chunks）

用法：
    # 只重命名文件，不碰 ES
    python repair.py --rename

    # 只重新索引 ES，不重命名
    python repair.py --reindex

    # 两步都做
    python repair.py --rename --reindex
"""

import json
import sys
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _safe_filename(title: str) -> str:
    name = title.strip()
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, '_')
    if len(name) > 80:
        name = name[:80]
    return name


def rename_files(imported_dir: Path):
    """将 {doc_id}.json 重命名为 {title}__{doc_id}.json"""
    renamed = 0
    skipped = 0

    for f in sorted(imported_dir.glob("*.json")):
        if f.name == "_summary.json":
            continue
        # 已经包含 __ 的说明已重命名过
        if "__" in f.stem:
            skipped += 1
            continue

        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            title = data.get("title", f.stem)
            doc_id = data.get("doc_id", f.stem)
            safe_title = _safe_filename(title)
            new_name = f"{safe_title}__{doc_id}.json"
            new_path = f.parent / new_name

            if new_path.exists():
                print(f"  跳过（目标已存在）: {new_name}")
                skipped += 1
                continue

            f.rename(new_path)
            print(f"  {f.name} → {new_name}")
            renamed += 1
        except Exception as e:
            print(f"  失败 {f.name}: {e}")

    print(f"\n  重命名完成: {renamed} 个文件, 跳过 {skipped} 个")


def reindex_from_cache(cache_dir: Path):
    """从 checkpoint 重新索引所有文档到 ES"""
    from models import Chunk, DocumentAnalysis, ProcessedDocument
    from store import get_store
    from core.retrieve.embedder import encode

    cache_files = sorted(cache_dir.glob("*.json"))
    if not cache_files:
        print("  未找到 checkpoint 文件")
        return

    print(f"  找到 {len(cache_files)} 个 checkpoint 文件")

    # 同时读 imported 目录获取 title 映射
    imported_dir = Path(__file__).parent / "data" / "imported"
    title_map: dict[str, str] = {}
    for f in imported_dir.glob("*.json"):
        if f.name == "_summary.json":
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            doc_id = data.get("doc_id", "")
            title = data.get("title", "")
            if doc_id:
                title_map[doc_id] = title
        except Exception:
            pass

    store = get_store()
    store.ensure_indices()

    total_success = 0
    total_chunks = 0
    failed_docs = []

    for i, cache_file in enumerate(cache_files):
        doc_id = cache_file.stem
        title = title_map.get(doc_id, doc_id)

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            analysis = DocumentAnalysis.from_dict(data["analysis"])
            chunks = [Chunk.from_dict(c) for c in data["chunks"]]

            # 重新计算 embedding（checkpoint 不存向量）
            for chunk in chunks:
                parts = []
                if chunk.keywords:
                    flat = []
                    for k in chunk.keywords:
                        if isinstance(k, list):
                            flat.extend(str(x) for x in k)
                        else:
                            flat.append(str(k))
                    parts.append(" ".join(flat))
                parts.append(chunk.context_summary or chunk.content)
                text_for_embedding = " ".join(parts)

                vector = encode(text_for_embedding)
                if vector is not None:
                    chunk._embedding_vector = vector.tolist()
                else:
                    chunk._embedding_vector = None

            # 构建 ProcessedDocument 并索引
            doc = ProcessedDocument(
                doc_id=doc_id,
                title=title,
                analysis=analysis,
                chunks=chunks,
                content="",
            )

            success = store.index_document(doc)
            total_success += success
            total_chunks += len(chunks)

            chunk_types = Counter(c.chunk_type for c in chunks)
            types_str = ", ".join(f"{k}:{v}" for k, v in chunk_types.most_common())
            print(f"  [{i+1}/{len(cache_files)}] {title}: {success}/{len(chunks)} chunks ({types_str})")

        except Exception as e:
            print(f"  [{i+1}/{len(cache_files)}] FAILED {title}: {e}")
            failed_docs.append((title, str(e)))

    print(f"\n  重新索引完成: {total_success}/{total_chunks} chunks 成功")
    if failed_docs:
        print(f"  失败文档 ({len(failed_docs)}):")
        for t, e in failed_docs:
            print(f"    {t}: {e}")


def main():
    parser = argparse.ArgumentParser(description="修复已导入的数据（不重跑 pipeline）")
    parser.add_argument("--rename", action="store_true", help="重命名 imported 文件")
    parser.add_argument("--reindex", action="store_true", help="从 checkpoint 重新索引到 ES")
    args = parser.parse_args()

    if not args.rename and not args.reindex:
        parser.print_help()
        return

    base_dir = Path(__file__).parent
    imported_dir = base_dir / "data" / "imported"
    cache_dir = base_dir / "data" / "cache"

    if args.rename:
        print(f"\n{'='*60}")
        print(f"  重命名 imported 文件")
        print(f"  目录: {imported_dir}")
        print(f"{'='*60}\n")
        rename_files(imported_dir)

    if args.reindex:
        print(f"\n{'='*60}")
        print(f"  从 checkpoint 重新索引到 ES")
        print(f"  cache 目录: {cache_dir}")
        print(f"{'='*60}\n")
        reindex_from_cache(cache_dir)


if __name__ == "__main__":
    main()
