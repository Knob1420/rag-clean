#!/usr/bin/env python
"""
批量 chunker 脚本 — 跑 extractor + chunker，保存 Document 列表到 JSON

不跑 LLM summary / embedder / ES index，用于：
1. 验证 chunker 在真实数据上的分块效果
2. 节省 LLM/embed 资源（chunker 是纯本地计算）
3. 后续 batch_import 时可加载这些 JSON 跳过 chunker 步骤

输出：data/processed/{dataset_id}/{title}.json
每个 JSON 是 List[Document] 序列化（含 parent + children metadata）

用法:
    python scripts/batch_chunker.py --dry-run                   # 看会处理哪些文件
    python scripts/batch_chunker.py                             # 跑全部
    python scripts/batch_chunker.py --dataset 2025-气象卫星      # 单 dataset
    python scripts/batch_chunker.py --limit 10                  # 前 10 个测试
    python scripts/batch_chunker.py --force                     # 强制重跑（默认跳过已存在）
"""
import sys
import json
import time
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# 复用 batch_import 的扫描逻辑
from scripts.batch_import import collect_files, RAW_DIR
from core.ingestion.extractor import convert_to_markdown
from core.ingestion.chunker import SmartChunker

# 输出目录
PROCESSED_DIR = Path("data/processed")


def _safe_filename(name: str) -> str:
    """转义文件名特殊字符"""
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        name = name.replace(ch, "_")
    return name


def _output_path(dataset_id: str, title: str) -> Path:
    """计算输出 JSON 路径：data/processed/{dataset}/{title}.json"""
    return PROCESSED_DIR / dataset_id / f"{_safe_filename(title)}.json"


def process_one(file_path: Path, dataset_id: str, source_key: str, chunker: SmartChunker) -> dict:
    """跑 extractor + chunker 一个文件，返回统计"""
    t0 = time.time()

    # 1. extractor（命中 cache 秒过）
    md = convert_to_markdown(str(file_path))
    if not md.strip():
        return {"status": "empty", "elapsed": time.time() - t0}

    # 2. chunker
    title = file_path.stem
    doc_id = f"{dataset_id}::{title}"  # 临时 doc_id（避免不同 dataset 同名冲突）
    documents = chunker.chunk(md, title=title, doc_id=doc_id, source_key=source_key)

    # 3. 统计
    parent_count = len(documents)
    child_count = sum(len(d.children or []) for d in documents)
    table_child_count = sum(
        1 for d in documents for c in (d.children or []) if c.content.strip().startswith("|")
    )

    # 4. 保存
    out_path = _output_path(dataset_id, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 序列化（不含 vector，chunker 没算 vector）
    data = []
    for doc in documents:
        d = doc.to_dict()
        d["vector"] = None
        for c in (d.get("children") or []):
            c["vector"] = None
        data.append(d)

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "ok",
        "parents": parent_count,
        "children": child_count,
        "tables": table_child_count,
        "out_path": str(out_path),
        "elapsed": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description="批量 extractor + chunker")
    ap.add_argument("--dataset", type=str, help="指定 dataset_id")
    ap.add_argument("--limit", type=int, help="只跑前 N 个文件")
    ap.add_argument("--force", action="store_true", help="强制重跑（默认跳过已存在 JSON）")
    ap.add_argument("--dry-run", action="store_true", help="只列文件不跑")
    args = ap.parse_args()

    files = collect_files(args.dataset)
    if args.limit:
        files = files[: args.limit]

    if not files:
        print("未找到任何文件")
        return

    by_fmt = Counter(f[1] for f in files)
    by_ds = Counter(f[2] for f in files)

    print(f"\n{'=' * 70}")
    print(f"  批量 Extractor + Chunker")
    print(f"  RAW_DIR: {RAW_DIR}")
    print(f"  OUTPUT:  {PROCESSED_DIR}")
    print(f"  文件总数: {len(files)}")
    print(f"  格式分布: {dict(by_fmt)}")
    print(f"  数据集数: {len(by_ds)}")
    print(f"{'=' * 70}\n")

    if args.dry_run:
        for i, (f, fmt, ds) in enumerate(files):
            out = _output_path(ds, f.stem)
            exists = "✓" if out.exists() else " "
            print(f"  [{i+1:4d}] [{fmt:4s}] [{ds[:30]:<30}] {exists} {f.name[:50]}")
        print(f"\n  共 {len(files)} 个文件 (--dry-run)")
        return

    # 跑
    chunker = SmartChunker()
    success, cached, failed = 0, 0, 0
    total_parents, total_children, total_tables = 0, 0, 0
    failed_list = []
    t_start = time.time()

    # 加载各 dataset 的 version_map（用于 source_key）
    from scripts.batch_import import load_version_map
    version_map_cache: dict[str, dict[str, str]] = {}

    for i, (path, fmt, ds) in enumerate(files):
        elapsed = time.time() - t_start
        rate = (i / elapsed) if elapsed > 0 else 0
        eta = (len(files) - i) / rate if rate > 0 else 0
        print(
            f"[{i+1}/{len(files)}] [{fmt:4s}] [{ds[:25]:<25}] {path.name[:50]:<50} "
            f"(累计 {elapsed:.0f}s, ETA {eta/60:.1f}min)"
        )

        # 跳过已存在（除非 --force）
        out = _output_path(ds, path.stem)
        if out.exists() and not args.force:
            cached += 1
            continue

        # source_key（从 _versions.yaml 来）
        if ds not in version_map_cache:
            version_map_cache[ds] = load_version_map(ds)
        source_key = version_map_cache[ds].get(path.name, "")

        try:
            result = process_one(path, ds, source_key, chunker)
            if result["status"] == "ok":
                success += 1
                total_parents += result["parents"]
                total_children += result["children"]
                total_tables += result["tables"]
            else:
                failed += 1
                failed_list.append((path, result["status"]))
        except Exception as e:
            failed += 1
            failed_list.append((path, str(e)[:200]))
            logger.error(f"\nFAILED: {path.name}: {e}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  完成（耗时 {elapsed/60:.1f} 分钟）")
    print(f"  转换: {success}, 已存在跳过: {cached}, 失败: {failed}, 总计: {len(files)}")
    print(f"  parents: {total_parents}, children: {total_children}, 表格 child: {total_tables}")
    print(f"{'=' * 70}")

    if failed_list:
        print(f"\n失败详情（前 20）:")
        for path, err in failed_list[:20]:
            print(f"  {path.name}: {err}")


if __name__ == "__main__":
    main()
