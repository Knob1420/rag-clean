"""
批量导入 — 按数据集批量处理文件

文件结构：
    /data/raw/{dataset_id}/
        file1.md
        file2.pdf

    /data/processed/{dataset_id}/
        file1.json      # 中间结果：List[Document] 序列化
        file2.json

用法：
    cd /home/zjlab/Documents/build_LLMs/NLP_course_hf/RAG/rag-clean

    # 预览模式（只列出文件，不执行）
    python batch_import.py --dry-run

    # 导入全部数据集（单线程）
    python batch_import.py

    # 导入指定数据集
    python batch_import.py --dataset my_knowledge_base

    # 只索引已有中间结果
    python batch_import.py --index-only
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from core.ingestion.extractor import SUPPORTED_FORMATS, detect_format
from core.ingestion.document_processor import process_document

# ── 目录配置 ──────────────────────────────────────────
DATA_ROOT = Path("/home/zjlab/Documents/build_LLMs/NLP_course_hf/RAG/data")
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed-0509"


def collect_files(dataset_id: Optional[str] = None) -> list[tuple[Path, str, str]]:
    """
    扫描 raw 目录下所有支持格式的文件。

    Returns:
        [(文件路径, 格式, dataset_id), ...]
    """
    files: list[tuple[Path, str, str]] = []

    search_dirs = []
    if dataset_id:
        d = RAW_DIR / dataset_id
        if d.exists():
            search_dirs.append(d)
    else:
        if RAW_DIR.exists():
            for subdir in RAW_DIR.iterdir():
                if subdir.is_dir():
                    search_dirs.append(subdir)

    for search_dir in search_dirs:
        current_dataset_id = search_dir.name
        # 只收集 .md 文件，排除重复（以 filename 为准）
        seen_names: set[str] = set()
        for f in sorted(search_dir.rglob("*.md")):
            if f.name in seen_names:
                continue
            seen_names.add(f.name)
            files.append((f, "md", current_dataset_id))

    return files


def is_processed(dataset_id: str, filename: str) -> bool:
    """检查中间结果是否已存在"""
    safe_name = Path(filename).stem
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        safe_name = safe_name.replace(ch, "_")
    json_path = PROCESSED_DIR / dataset_id / f"{safe_name}.json"
    return json_path.exists()


def process_file(
    file_info: tuple[Path, str, str],
    index_only: bool = False,
    use_summary: bool = True,
    chunk_mode: str = "recursive",
) -> Dict[str, Any]:
    """
    处理单个文件（在线程中执行）

    Returns:
        {"status": "ok"|"skipped"|"failed", "dataset_id": ..., "title": ..., "error": ...}
    """
    f, _fmt, dataset_id = file_info

    # 跳过检查
    if is_processed(dataset_id, f.name):
        return {
            "dataset_id": dataset_id,
            "title": f.stem,
            "status": "skipped",
        }

    try:
        documents = process_document(
            file_path=str(f),
            dataset_id=dataset_id,
            load_intermediate=index_only,
            processed_dir=PROCESSED_DIR,
            use_summary=use_summary,
            chunk_mode=chunk_mode,
        )
        return {
            "dataset_id": dataset_id,
            "title": f.stem,
            "status": "ok",
            "doc_count": len(documents),
        }
    except Exception as e:
        logger.exception(f"处理失败: {f.name}")
        return {
            "dataset_id": dataset_id,
            "title": f.stem,
            "status": "failed",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="批量导入文件到 RAG")
    parser.add_argument("--dry-run", action="store_true", help="只列出文件，不执行导入")
    parser.add_argument("--dataset", type=str, help="指定数据集 ID（默认全部）")
    parser.add_argument(
        "--skip-existing", action="store_true", help="跳过已有中间结果的文件"
    )
    parser.add_argument("--index-only", action="store_true", help="只索引已有中间结果")
    parser.add_argument(
        "--no-summary", action="store_true", help="关闭 summary 生成（加快处理速度）"
    )
    parser.add_argument(
        "--chunk-mode",
        type=str,
        default="recursive",
        choices=["recursive", "semantic"],
        help="分块模式：recursive（默认）或 semantic",
    )
    args = parser.parse_args()

    # 收集文件
    files = collect_files(args.dataset)

    if not files:
        print("未找到任何文件")
        return

    print(f"\n{'='*70}")
    print(f"  批量导入")
    print(f"  数据目录: {RAW_DIR}")
    print(f"  输出目录: {PROCESSED_DIR}")
    print(
        f"  模式: {'dry-run' if args.dry_run else ('index-only' if args.index_only else 'full')}"
    )
    print(f"  发现文件: {len(files)}")
    print(f"{'='*70}\n")

    if args.dry_run:
        for i, (f, fmt, dataset_id) in enumerate(files):
            rel = (
                f.relative_to(RAW_DIR / dataset_id)
                if (RAW_DIR / dataset_id) in f.parents
                else f
            )
            print(f"  [{i+1:3d}] [{fmt:4s}] [{dataset_id}] {rel.name}")
        print(f"\n  共 {len(files)} 个文件 (--dry-run)")
        return

    # 统计
    success = 0
    failed = 0
    skipped = 0
    results = []

    # 单线程顺序处理
    for i, file_info in enumerate(files):
        f, fmt, dataset_id = file_info
        print(f"\n{'─'*70}")
        print(f"[{i+1}/{len(files)}] [{dataset_id}] {f.name}")

        result = process_file(
            file_info,
            args.index_only,
            use_summary=not args.no_summary,
            chunk_mode=args.chunk_mode,
        )
        results.append(result)

        if result["status"] == "ok":
            print(f"  处理完成: {result.get('doc_count', 0)} documents")
            success += 1
        elif result["status"] == "skipped":
            print(f"  已跳过（中间结果已存在）")
            skipped += 1
        else:
            print(f"  FAILED: {result.get('error', 'unknown')}")
            failed += 1

    # 汇总
    print(f"\n{'='*70}")
    print(f"  完成")
    print(f"  成功: {success}  失败: {failed}  跳过: {skipped}  总计: {len(files)}")
    print(f"{'='*70}")

    # 按 dataset_id 统计
    dataset_counts = Counter(
        r.get("dataset_id") for r in results if r.get("status") == "ok"
    )
    if dataset_counts:
        print(f"\n  按数据集分布:")
        for ds, cnt in dataset_counts.most_common():
            print(f"    {ds}: {cnt}")


if __name__ == "__main__":
    main()
