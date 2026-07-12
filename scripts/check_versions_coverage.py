#!/usr/bin/env python
"""
检查 detect_versions 漏检 — 基于内容 trigram 相似度

策略：
1. 读 data/cache/converters/*.md（batch_extract 已跑过）
2. 按 raw 文件夹分组
3. 文件夹内两两计算 trigram Jaccard 相似度
4. > 阈值但 detect_versions 没识别的 → 输出漏检候选

trigram 比文件名相似度更准：M1 vs MT1 文件名相似但内容不同 → 不聚类。

用法:
    python scripts/check_versions_coverage.py                     # 默认阈值 0.7
    python scripts/check_versions_coverage.py --threshold 0.8     # 更严格
    python scripts/check_versions_coverage.py --apply             # 生成补充 yaml
"""
import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.detect_versions import (
    extract_base_name,
    has_version_marker,
    is_generic_name,
    SUPPORTED_EXTS,
)
from scripts.batch_import import collect_files


def trigram_set(text: str) -> set[str]:
    """生成 trigram 集合（3 个连续字符）"""
    return set(text[i : i + 3] for i in range(len(text) - 2))


def jaccard(s1: set, s2: set) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def read_cache_md(file_path: Path) -> str:
    """从 cache/converters/ 读对应的 md（用 content hash 匹配）"""
    from core.ingestion.extractor import _content_hash, _cache_path

    cache = _cache_path(file_path)
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="ignore")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.7, help="trigram Jaccard 阈值（默认 0.7）")
    ap.add_argument("--apply", action="store_true", help="直接补充到 _versions.yaml")
    ap.add_argument("--dataset", type=str, help="指定 dataset")
    args = ap.parse_args()

    raw_root = Path("data/raw")

    # 1. 扫描 raw，按文件夹分组
    files = collect_files(args.dataset)
    folder_to_files: dict[Path, list[Path]] = defaultdict(list)
    for path, fmt, ds in files:
        folder_to_files[path.parent].append(path)

    # 过滤：只看 >= 2 个文件的文件夹
    folders_with_pairs = {f: v for f, v in folder_to_files.items() if len(v) >= 2}

    print(f"扫描 {len(files)} 个文件，{len(folders_with_pairs)} 个文件夹有 >=2 个文件")
    print(f"trigram Jaccard 阈值: {args.threshold}\n")

    # 2. 每个文件夹内，读 md + 算 trigram + 两两比较
    total_missed = 0
    total_files = 0
    results: dict[Path, list[tuple[str, list[tuple[Path, float]]]]] = {}

    for folder in sorted(folders_with_pairs.keys()):
        folder_files = folder_to_files[folder]

        # 读每个文件的 cache md
        file_data: list[tuple[Path, set[str], str]] = []  # (path, trigrams, content)
        for f in folder_files:
            md = read_cache_md(f)
            if len(md) > 50:  # 太短的可能是空文档
                file_data.append((f, trigram_set(md[:5000]), md[:5000]))  # 截取前 5000 字符算 trigram（加速）

        if len(file_data) < 2:
            continue

        # detect_versions 已识别的文件集合
        base_groups: dict[str, list[Path]] = defaultdict(list)
        for f, _, _ in file_data:
            base = extract_base_name(f.name)
            base_groups[base].append(f)
        detected_files: set[Path] = set()
        for base, group in base_groups.items():
            if len(group) > 1 and not is_generic_name(base) and any(
                has_version_marker(f.name, base) for f in group
            ):
                detected_files.update(group)

        # 未识别的文件，两两比较
        undetected = [(f, t) for f, t, _ in file_data if f not in detected_files]
        if len(undetected) < 2:
            continue

        # 贪心聚类
        groups: list[list[tuple[Path, float]]] = []
        remaining = list(undetected)
        while remaining:
            seed_path, seed_tri = remaining[0]
            group = [(seed_path, 1.0)]
            remaining.remove((seed_path, seed_tri))
            for f, t in list(remaining):
                sim = jaccard(seed_tri, t)
                if sim >= args.threshold:
                    group.append((f, sim))
                    remaining.remove((f, t))
            if len(group) > 1:
                groups.append(group)

        if groups:
            # 建议的 base_name：用最长的 stem（去掉版本后缀）
            for group in groups:
                # 排序：按文件名字典序（模拟接入顺序）
                group.sort(key=lambda x: x[0].name)
                suggested = extract_base_name(group[0][0].name) or group[0][0].stem
                results.setdefault(folder, []).append((suggested, group))
                total_missed += 1
                total_files += len(group)

    # 3. 输出
    print(f"{'=' * 70}")
    print(f"  漏检检查（内容 trigram 相似度 > {args.threshold}）")
    print(f"  共 {total_missed} 个漏检组，{total_files} 个文件")
    print(f"  分布在 {len(results)} 个文件夹")
    print(f"{'=' * 70}")

    for folder, groups in results.items():
        try:
            rel = folder.relative_to(raw_root)
        except ValueError:
            rel = folder
        print(f"\n=== {rel} ===")
        for base, group in groups:
            print(f"\n[建议组] base='{base}' ({len(group)} 个文件)")
            for f, sim in group:
                print(f"  - {f.name:<60} (内容相似度 {sim:.2f})")

    if args.apply and results:
        import re
        total_skipped = 0
        for folder, groups in results.items():
            yaml_path = folder / "_versions.yaml"
            existing = ""
            existing_files: set[str] = set()
            if yaml_path.exists():
                existing = yaml_path.read_text(encoding="utf-8")
                # 解析已有映射（提取 "filename: source_key" 行，排除注释）
                for line in existing.splitlines():
                    m = re.match(r"^([^:#\n][^:]*?):\s*(.+)$", line)
                    if m:
                        existing_files.add(m.group(1).strip())

            new_lines = ["\n# === 漏检补充（内容 trigram 相似度 > {:.1f}）===\n".format(args.threshold)]
            skipped = 0
            for base, group in groups:
                group_lines = []
                for f, _ in group:
                    if f.name in existing_files:
                        skipped += 1
                        continue
                    group_lines.append(f"{f.name}: {base}")
                if len(group_lines) > 1:  # 跳过后至少 2 个才有意义
                    new_lines.append(f"# 组: {base} ({len(group_lines)} 文件)")
                    new_lines.extend(group_lines)
                    new_lines.append("")
            new_content = "\n".join(new_lines)
            if new_content.strip():
                yaml_path.write_text(existing + new_content, encoding="utf-8")
                print(f"[APPEND] {yaml_path} (跳过 {skipped} 个已存在)")
            total_skipped += skipped
        print(f"\n共跳过 {total_skipped} 个已有映射，避免冲突")
    elif args.apply:
        print("\n无漏检组，不生成 yaml。")


if __name__ == "__main__":
    main()
