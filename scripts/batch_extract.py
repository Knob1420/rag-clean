#!/usr/bin/env python
"""
批量转换脚本 — 只跑 extractor（PDF/DOCX/PPTX/XLSX/CSV → Markdown）

不跑 chunker / embed / ES 索引，用于：
1. 批量预热 cache（之后跑 batch_import 时 MinerU 部分秒过）
2. 单独验证 extractor 输出质量
3. chunker 还没修好时也能积累 markdown 缓存

用法:
    python scripts/batch_extract.py --dry-run                       # 看会处理哪些文件
    python scripts/batch_extract.py                                 # 跑全部数据集
    python scripts/batch_extract.py --dataset 2025-气象卫星           # 只跑某数据集
    python scripts/batch_extract.py --limit 10                      # 只跑前 10 个文件（测试用）
    python scripts/batch_extract.py --force                         # 清缓存重跑（默认走缓存跳过）
"""
import sys
import argparse
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# 复用 batch_import 的扫描逻辑（同一个 collect_files，避免重复代码）
from scripts.batch_import import collect_files, RAW_DIR
from core.ingestion.extractor import convert_to_markdown, detect_format


def main():
    ap = argparse.ArgumentParser(description="批量 extractor — 只转换不索引")
    ap.add_argument("--dataset", type=str, help="指定 dataset_id（默认全部）")
    ap.add_argument("--limit", type=int, help="只跑前 N 个文件（测试用）")
    ap.add_argument("--force", action="store_true", help="清缓存重跑（默认走 cache 跳过已转过的）")
    ap.add_argument("--dry-run", action="store_true", help="只列文件不跑")
    args = ap.parse_args()

    files = collect_files(args.dataset)
    if args.limit:
        files = files[: args.limit]

    if not files:
        print("未找到任何文件")
        return

    # 统计格式分布
    by_fmt = Counter(f[1] for f in files)
    by_ds = Counter(f[2] for f in files)

    print(f"\n{'=' * 70}")
    print(f"  批量 Extractor")
    print(f"  数据目录: {RAW_DIR}")
    print(f"  文件总数: {len(files)}")
    print(f"  格式分布: {dict(by_fmt)}")
    print(f"  数据集数: {len(by_ds)}")
    print(f"{'=' * 70}\n")

    if args.dry_run:
        for i, (f, fmt, ds) in enumerate(files):
            print(f"  [{i+1:4d}] [{fmt:4s}] [{ds}] {f.name}")
        print(f"\n  共 {len(files)} 个文件 (--dry-run)")
        return

    # 跑转换
    success, cached, failed = 0, 0, 0
    failed_list = []
    t_start = time.time()

    for i, (path, fmt, ds) in enumerate(files):
        elapsed = time.time() - t_start
        rate = (i / elapsed) if elapsed > 0 else 0
        eta = (len(files) - i) / rate if rate > 0 else 0
        # 用换行式而非 \r 覆盖（便于 tee 到日志文件）
        print(f"[{i+1}/{len(files)}] [{fmt:4s}] [{ds}] {path.name[:50]:<50} "
              f"(累计 {elapsed:.0f}s, ETA {eta/60:.1f}min)")

        if args.force:
            # 清 cache（converter cache + mineru parse_backup）
            from core.ingestion.extractor import _cache_path, _content_hash
            from config import settings
            cache_md = _cache_path(path)
            if cache_md.exists():
                cache_md.unlink()
            backup = Path(settings.parse_backup_dir) / f"{path.stem}_{_content_hash(path)}"
            if backup.exists():
                import shutil
                shutil.rmtree(backup)

        try:
            t0 = time.time()
            md = convert_to_markdown(str(path))
            dt = time.time() - t0
            # 区分 cache 命中（<0.1s）和真实转换
            if dt < 0.1:
                cached += 1
            else:
                success += 1
        except Exception as e:
            failed += 1
            failed_list.append((path, fmt, str(e)[:200]))
            logger.error(f"\nFAILED: {path.name}: {e}")

    elapsed = time.time() - t_start
    print(f"\n\n{'=' * 70}")
    print(f"  完成（耗时 {elapsed/60:.1f} 分钟）")
    print(f"  转换: {success}, 缓存命中: {cached}, 失败: {failed}, 总计: {len(files)}")
    print(f"{'=' * 70}")

    if failed_list:
        print(f"\n失败详情（前 20）:")
        for path, fmt, err in failed_list[:20]:
            print(f"  [{fmt}] {path}: {err}")
        if len(failed_list) > 20:
            print(f"  ... 还有 {len(failed_list)-20} 个失败未显示")


if __name__ == "__main__":
    main()
