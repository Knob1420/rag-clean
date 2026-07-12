#!/usr/bin/env python
"""
多版本检测脚本 — 按文件名聚类识别同文件夹内的版本组

不同版本一般在同一文件夹内（用户观察）。
本脚本在每个文件夹内单独聚类，输出候选版本组，dry-run 默认。

用法:
    python scripts/detect_versions.py                              # 检测所有 dataset
    python scripts/detect_versions.py --dataset 2025-...           # 指定 dataset
    python scripts/detect_versions.py --apply                      # 生成 _versions.yaml
    python scripts/detect_versions.py --apply --dataset 2025-...   # 指定 dataset 直接生成
    python scripts/detect_versions.py --min-group-size 3           # 只看 >=3 个文件的组
"""
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

SUPPORTED_EXTS = {".docx", ".doc", ".pdf", ".pptx", ".xlsx", ".csv", ".md"}

# 通用名黑名单：这些 base_name 太宽泛（如"会议纪要"），同 base 多个文件
# 通常是不同事件，不算版本组
GENERIC_NAMES = {
    "会议纪要", "纪要", "会议记录", "通知", "备忘录",
    "周报", "月报", "日报", "总结", "汇报", "报告",
    "meeting", "minutes", "notice", "report",
}

# 版本/日期/状态后缀正则（按优先级匹配）
VERSION_PATTERNS = [
    # 版本号：-v1.5 / _v1.5 / V1.5
    r"[-_](?:v|V)\d+(?:\.\d+)*",
    # 日期：-20240607 / _20240607 / -2024-06-07
    r"[-_](?:20\d{6}|20\d{2}[-_]\d{2}[-_]\d{2})",
    # 中文版本标记：终版/最新/定稿/终稿/会签版/签字版/修改版/修订版/改版/初稿/送审稿
    r"[-_](?:终版|最新|定稿|终稿|会签版|签字版|修改版|修订版|改版|初稿|送审稿)",
    # 英文版本标记
    r"[-_](?:final|draft|reviewed|approved|latest)",
    # 副本标记：(1) (2)
    r"\s*\(\d+\)",
]

# 编译正则
_COMPILED_PATTERNS = [re.compile(p) for p in VERSION_PATTERNS]


def extract_base_name(filename: str) -> str:
    """从文件名提取基础名（去版本/日期/状态后缀）"""
    stem = Path(filename).stem
    base = stem
    for pat in _COMPILED_PATTERNS:
        base = pat.sub("", base)
    # 去掉末尾残留的 - _ 空格
    return base.rstrip("-_ 　").strip()


def has_version_marker(filename: str, base: str) -> bool:
    """文件名 stem 是否含版本标记（即 stem != base，有后缀被去掉）。

    用于排除"docx + pdf 同名跨格式对"（无版本标记 → 不算版本）。
    """
    return Path(filename).stem != base


def is_generic_name(base: str) -> bool:
    """base_name 是否在通用词黑名单（避免"会议纪要"等误判）"""
    base_lower = base.lower().strip()
    if not base_lower:
        return True
    # 精确匹配
    if base_lower in GENERIC_NAMES:
        return True
    # 包含"事件性"关键词 → 不同日期的同主题事件，不是版本
    # （不限长度，因为像"XXX项目首次会议纪要"也是事件，不是版本）
    event_keywords = {
        "会议纪要", "纪要", "会议记录", "会议", "通知", "备忘录",
        "周报", "月报", "日报", "汇报材料",
        "minutes", "meeting", "notice",
    }
    for kw in event_keywords:
        if kw in base_lower:
            return True
    return False


def scan_dir(raw_dir: Path) -> dict[Path, dict[str, list[Path]]]:
    """
    扫描 raw_dir，按 (文件夹, base_name) 聚类。

    Returns:
        {folder_path: {base_name: [file_paths]}}

    同 base_name 仅在**同一文件夹内**聚类，跨文件夹不合并
    （不同版本通常在同一文件夹内）。
    """
    # 第一层：按文件夹分组
    folder_to_files: dict[Path, list[Path]] = defaultdict(list)
    for f in raw_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in SUPPORTED_EXTS:
            continue
        folder_to_files[f.parent].append(f)

    # 第二层：每个文件夹内按 base_name 聚类
    result: dict[Path, dict[str, list[Path]]] = {}
    for folder, files in folder_to_files.items():
        base_groups: dict[str, list[Path]] = defaultdict(list)
        for f in files:
            base = extract_base_name(f.name)
            base_groups[base].append(f)

        # 应用过滤规则
        groups_filtered: dict[str, list[Path]] = {}
        for base, group_files in base_groups.items():
            if len(group_files) < 2:
                continue  # 单文件，不算版本组
            # 规则 1：base_name 不能是通用词（如"会议纪要"）
            if is_generic_name(base):
                continue
            # 规则 2：组内至少 1 个文件有版本标记（排除 docx+pdf 同名跨格式对）
            if not any(has_version_marker(f.name, base) for f in group_files):
                continue
            groups_filtered[base] = sorted(group_files)

        if groups_filtered:
            result[folder] = groups_filtered

    return result


def format_groups(groups_by_folder: dict[Path, dict[str, list[Path]]], min_size: int) -> str:
    """格式化输出"""
    lines = []
    total_groups = 0
    total_files = 0

    for folder in sorted(groups_by_folder.keys()):
        groups = {k: v for k, v in groups_by_folder[folder].items() if len(v) >= min_size}
        if not groups:
            continue

        try:
            rel_folder = folder.relative_to(folder.parents[-1] if folder.parents else Path("."))
        except ValueError:
            rel_folder = folder
        # 显示 raw 下的相对路径
        try:
            rel_folder = folder.relative_to(Path("data/raw"))
        except ValueError:
            pass

        lines.append(f"\n=== 文件夹: {rel_folder} ===")
        for base, files in sorted(groups.items(), key=lambda x: -len(x[1])):
            total_groups += 1
            total_files += len(files)
            lines.append(f"\n[组] base='{base}' ({len(files)} 个文件)")
            for f in files:
                lines.append(f"  - {f.name}")

    header = [
        f"{'=' * 70}",
        f"  多版本检测",
        f"  共 {total_groups} 个候选组，{total_files} 个文件",
        f"{'=' * 70}",
    ]
    return "\n".join(header + lines)


def generate_versions_yaml(
    groups_by_folder: dict[Path, dict[str, list[Path]]],
    min_size: int,
) -> dict[Path, str]:
    """
    生成每个文件夹的 _versions.yaml 内容（字符串）。

    返回 {folder_path: yaml_content}，由调用者写入文件。
    """
    result: dict[Path, str] = {}
    for folder, groups in groups_by_folder.items():
        groups = {k: v for k, v in groups.items() if len(v) >= min_size}
        if not groups:
            continue
        lines = [f"# 自动检测的版本组（{len(groups)} 组）"]
        lines.append("# 检查后请删除不该合并的组（假阳性）")
        lines.append("")
        for base, files in sorted(groups.items()):
            lines.append(f"# 组: {base} ({len(files)} 文件)")
            for f in files:
                lines.append(f"{f.name}: {base}")
            lines.append("")
        result[folder] = "\n".join(lines)
    return result


def main():
    ap = argparse.ArgumentParser(description="检测多版本文档（按文件名聚类）")
    ap.add_argument("--dataset", type=str, help="指定 dataset（默认全部）")
    ap.add_argument("--apply", action="store_true", help="生成 _versions.yaml（默认只显示候选）")
    ap.add_argument("--min-group-size", type=int, default=2, help="最小版本组大小（默认 2）")
    args = ap.parse_args()

    raw_root = Path("data/raw")
    if args.dataset:
        raw_root = raw_root / args.dataset

    if not raw_root.exists():
        print(f"目录不存在: {raw_root}")
        return

    print(f"扫描: {raw_root}")
    groups = scan_dir(raw_root)

    if not groups:
        print("\n未检测到候选版本组（每个 base_name 只有 1 个文件）")
        return

    if not args.apply:
        # dry-run 模式：只显示
        print(format_groups(groups, args.min_group_size))
        print(f"\n{'=' * 70}")
        print("这是候选版本组（dry-run）。")
        print("确认后用 --apply 生成 _versions.yaml（每个文件夹一个）。")
        print("假阳性（如'（从星03）'vs'（从星05）'）请手动从 yaml 删除。")
        print(f"{'=' * 70}")
        return

    # apply 模式：写 _versions.yaml
    import yaml  # type: ignore
    yaml_contents = generate_versions_yaml(groups, args.min_group_size)
    written = 0
    for folder, content in yaml_contents.items():
        out_path = folder / "_versions.yaml"
        if out_path.exists():
            print(f"[SKIP] 已存在（不覆盖）: {out_path}")
            continue
        out_path.write_text(content, encoding="utf-8")
        print(f"[WRITE] {out_path}")
        written += 1

    print(f"\n已生成 {written} 个 _versions.yaml")
    print("请检查每个文件，删除假阳性条目，然后跑 batch_import.py")


if __name__ == "__main__":
    main()
