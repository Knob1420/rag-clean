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

# 可选依赖：PyYAML（用于 _versions.yaml）
try:
    import yaml
except ImportError:
    yaml = None

# ── 目录配置 ──────────────────────────────────────────
# 项目根 = batch_import.py 所在目录的父目录
_PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = _PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"


def _version_sort_key(filename: str) -> tuple:
    """
    生成版本排序 key：旧版本在前，新版本在后。

    排序优先级（从小到大 = 从旧到新）：
      0: 日期前缀（20240607-xxx）← 最旧
      1: 版本号（v1.5 / V2.0）
      2: 无版本标记
      3: final/终版/定稿/会签版/签字版 ← 最新

    确保接入顺序正确：后接入 = latest。
    """
    import re

    name_lower = filename.lower()
    # final/终版 等标记 → 排最后
    final_keywords = [
        "final", "终版", "定稿", "终稿", "最新",
        "签字版", "会签版", "签发版", "发布版",
    ]
    if any(kw in name_lower for kw in final_keywords):
        return (3, filename)
    # 版本号 v2.0 / V1.5
    v_match = re.search(r"[vV](\d+(?:\.\d+)*)", filename)
    if v_match:
        version_tuple = tuple(int(x) for x in v_match.group(1).split("."))
        return (1, version_tuple, filename)
    # 日期 20240607
    d_match = re.search(r"(20\d{6})", filename)
    if d_match:
        return (0, d_match.group(1), filename)
    # 无版本标记
    return (2, filename)

# 支持扫描的扩展名（小写、含点）
# 注意：.ppt extractor 当前会抛"暂不支持"错误，扫描进来会让用户看到失败提示
SCAN_EXTS = {".doc", ".docx", ".pdf", ".ppt", ".pptx", ".csv", ".xlsx", ".md"}


def collect_files(dataset_id: Optional[str] = None) -> list[tuple[Path, str, str]]:
    """
    扫描 raw 目录下所有支持格式的文件。

    Args:
        dataset_id: 指定数据集（= raw 下的子目录名）；None = 扫描所有子目录

    Returns:
        [(文件路径, 格式, dataset_id), ...]
        格式由 extractor.detect_format 推断（md/doc/docx/pdf/ppt/pptx/csv/xlsx）
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
        # 扫描所有支持格式（按相对路径去重，避免软链/重复备份产生多次接入）
        seen_paths: set[str] = set()
        for f in sorted(search_dir.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix.lower() not in SCAN_EXTS:
                continue
            # 跳过隐藏文件 / 临时文件
            if f.name.startswith("~$") or f.name.startswith("."):
                continue
            rel = str(f.relative_to(search_dir))
            if rel in seen_paths:
                continue
            seen_paths.add(rel)
            files.append((f, detect_format(f), current_dataset_id))

    return files


def load_version_map(dataset_id: str) -> Dict[str, str]:
    """
    加载 _versions.yaml：文件名 → source_key 映射。

    位置：data/raw/{dataset_id}/_versions.yaml
    格式：
        G1技术规范_v1.md: G1技术规范
        G1技术规范_v2.md: G1技术规范

    缺失或解析失败 → 返回空 dict（即不启用版本管理）
    """
    if yaml is None:
        logger.warning("PyYAML 未安装，跳过 _versions.yaml 加载")
        return {}

    path = RAW_DIR / dataset_id / "_versions.yaml"
    if not path.exists():
        return {}

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            logger.warning(f"_versions.yaml 格式错误（应为 dict）: {path}")
            return {}
        logger.info(f"[版本映射] 加载 {len(data)} 条: {path}")
        return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        logger.warning(f"_versions.yaml 解析失败: {e}")
        return {}


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
    version_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    处理单个文件（在线程中执行）

    如果 data/processed/{ds}/{title}.json 存在 → 从 JSON 加载（跳过 chunker）
    否则 → 从头跑 chunker

    Returns:
        {"status": "ok"|"skipped"|"failed", "dataset_id": ..., "title": ..., "error": ...}
    """
    f, _fmt, dataset_id = file_info

    # 版本链：若文件在 version_map 中，传 source_key 启用版本管理
    source_key = (version_map or {}).get(f.name)

    try:
        documents = process_document(
            file_path=str(f),
            dataset_id=dataset_id,
            load_intermediate=True,  # 默认尝试加载 JSON（batch_chunker 产物）
            processed_dir=PROCESSED_DIR,
            use_summary=use_summary,
            chunk_mode=chunk_mode,
            source_key=source_key,
        )
        return {
            "dataset_id": dataset_id,
            "title": f.stem,
            "status": "ok",
            "doc_count": len(documents),
            "source_key": source_key or "",
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
    parser.add_argument("--limit", type=int, help="只处理前 N 个文件（测试用）")
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

    # 预加载所有 version_map，按版本排序（同 source_key 内按新旧排序）
    version_map_cache: Dict[str, Dict[str, str]] = {}
    for _, _, ds in files:
        if ds not in version_map_cache:
            version_map_cache[ds] = load_version_map(ds)

    # 排序：同 dataset + 同 source_key 的文件按版本新旧排序（旧→新）
    # 不同 dataset / 不同 source_key 之间按 dataset 名排序
    def _sort_key(file_info):
        f, fmt, ds = file_info
        sk = version_map_cache.get(ds, {}).get(f.name, "")
        return (ds, sk, _version_sort_key(f.name))

    files.sort(key=_sort_key)

    if args.limit:
        files = files[: args.limit]

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

    # 单线程顺序处理（version_map 已在排序前预加载到 version_map_cache）
    for i, file_info in enumerate(files):
        f, fmt, dataset_id = file_info
        print(f"\n{'─'*70}")
        print(f"[{i+1}/{len(files)}] [{dataset_id}] {f.name}")

        # version_map 已在排序前预加载
        vmap = version_map_cache.get(dataset_id, {})

        if f.name in vmap:
            print(f"  [版本链] source_key = {vmap[f.name]}")

        result = process_file(
            file_info,
            args.index_only,
            use_summary=not args.no_summary,
            chunk_mode=args.chunk_mode,
            version_map=vmap,
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
