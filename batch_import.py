"""
批量导入 — 扫描 data 目录下所有支持的文件格式，逐一送入 pipeline

用法：
    cd /home/zjlab/Documents/build_LLMs/NLP_course_hf/RAG/rag-clean

    # 导入全部
    python batch_import.py

    # 预览模式（只列出文件，不执行）
    python batch_import.py --dry-run

    # 只导入 raw 目录
    python batch_import.py --source raw

    # 只导入 parsed_backups 目录
    python batch_import.py --source backup

    # 指定自定义数据目录
    python batch_import.py --data-dir /path/to/data
"""

import json
import sys
import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from core.ingestion.converters import SUPPORTED_FORMATS, detect_format

# ── 默认数据根目录 ──────────────────────────────────
DEFAULT_DATA_ROOT = Path(
    "/home/zjlab/Documents/build_LLMs/NLP_course_hf/RAG/rag-knowledge-base/data/raw"
)

# 处理结果保存目录
OUTPUT_DIR = Path(__file__).parent / "data" / "imported"


def collect_files(data_root: Path, source: str = "all") -> list[tuple[Path, str]]:
    """
    递归扫描 data_root 下所有支持格式的文件，返回 (路径, 格式) 元组列表。

    策略：
    - 对 .md 文件，优先取 hybrid_auto 子目录下的（MinerU 高质量输出）
    - 对非 .md 文件，直接收集
    - 按文件名（stem）去重，优先保留 hybrid_auto 目录的 .md
    """
    files: list[tuple[Path, str]] = []

    # 确定搜索根目录
    search_dirs: list[Path] = []
    if source == "all":
        if data_root.exists():
            search_dirs.append(data_root)
    elif source == "raw":
        d = data_root / "raw"
        if d.exists():
            search_dirs.append(d)
    elif source == "backup":
        d = data_root / "parsed_backups"
        if d.exists():
            search_dirs.append(d)

    for search_dir in search_dirs:
        for ext in sorted(SUPPORTED_FORMATS):
            for f in sorted(search_dir.rglob(f"*{ext}")):
                fmt = detect_format(f)
                if fmt == "unknown":
                    continue
                # 跳过辅助文件（images 目录、_middle.json 等）
                if "images" in f.parts:
                    continue
                if f.name.endswith("_middle.json"):
                    continue
                files.append((f, fmt))

    # 去重策略：同名 stem 优先保留 hybrid_auto 路径下的 .md
    # 先按优先级排序：hybrid_auto 的 .md 排前面
    def _priority(item: tuple[Path, str]) -> int:
        f, fmt = item
        if "hybrid_auto" in f.parts and fmt == "md":
            return 0  # 最高优先级
        if fmt == "md":
            return 1
        return 2

    files.sort(key=_priority)

    seen: set[str] = set()
    unique: list[tuple[Path, str]] = []
    for f, fmt in files:
        key = f.stem
        if key not in seen:
            seen.add(key)
            unique.append((f, fmt))

    return unique


def _safe_filename(title: str) -> str:
    """将标题转为安全文件名（去除/替换不合法字符）"""
    name = title.strip()
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, '_')
    # 截断过长文件名
    if len(name) > 80:
        name = name[:80]
    return name


def save_imported_doc(doc, output_dir: Path) -> Path:
    """将处理后的文档保存为 JSON，文件名包含标题"""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_title = _safe_filename(doc.title)
    out_path = output_dir / f"{safe_title}__{doc.doc_id}.json"

    # 收集 chunk 统计
    chunk_type_counts = Counter(c.chunk_type for c in doc.chunks)
    all_keywords: list[str] = []
    for c in doc.chunks:
        all_keywords.extend(c.keywords)

    keyword_counts = Counter(all_keywords)
    top_keywords = keyword_counts.most_common(20)

    data = {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "domain": doc.analysis.domain,
        "doc_type": doc.analysis.doc_type,
        "entities": doc.analysis.entities,
        "summary": doc.analysis.summary,
        "topics": doc.analysis.topics,
        "chunk_count": len(doc.chunks),
        "chunk_types": dict(chunk_type_counts),
        "top_keywords": [(kw, cnt) for kw, cnt in top_keywords],
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "chunk_type": c.chunk_type,
                "section_title": c.section_title,
                "keywords": c.keywords,
                "context_summary": c.context_summary,
                "content_preview": c.content[:200],
            }
            for c in doc.chunks
        ],
        "content": doc.content,
    }

    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def print_doc_stats(doc) -> None:
    """打印单篇文档的统计信息"""
    a = doc.analysis
    print(f"  domain: {a.domain or '-'}")
    print(f"  doc_type: {a.doc_type or '-'}")
    if a.entities:
        entities_str = ", ".join(f"{k}={v}" for k, v in list(a.entities.items())[:5])
        print(f"  entities: {entities_str}")
    if a.topics:
        print(f"  topics: {', '.join(a.topics[:5])}")

    # Chunk 统计
    chunk_types = Counter(c.chunk_type for c in doc.chunks)
    types_str = ", ".join(f"{k}:{v}" for k, v in chunk_types.most_common())
    print(f"  chunks: {len(doc.chunks)} ({types_str})")

    # Top keywords
    all_kw: list[str] = []
    for c in doc.chunks:
        all_kw.extend(c.keywords)
    if all_kw:
        kw_counts = Counter(all_kw)
        top_kw = ", ".join(kw for kw, _ in kw_counts.most_common(10))
        print(f"  keywords: {top_kw}")


def main():
    parser = argparse.ArgumentParser(
        description="批量导入文件到 RAG（支持 MD/PDF/DOC/DOCX/PPTX）"
    )
    parser.add_argument("--dry-run", action="store_true", help="只列出文件，不执行导入")
    parser.add_argument(
        "--source",
        choices=["all", "raw", "backup"],
        default="all",
        help="数据来源: all=全部, raw=仅raw, backup=仅parsed_backups",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help=f"数据根目录（默认: {DEFAULT_DATA_ROOT}）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"处理结果保存目录（默认: {OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="从第 N 个文件开始（用于断点续传）",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    output_dir = Path(args.output)

    # 收集文件
    files = collect_files(data_root, args.source)

    # 按格式统计
    format_counts: dict[str, int] = {}
    for _, fmt in files:
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    format_summary = ", ".join(f"{k}:{v}" for k, v in sorted(format_counts.items()))

    print(f"\n{'='*70}")
    print(f"  批量导入 — 共发现 {len(files)} 个文件")
    print(f"  数据目录: {data_root}")
    print(f"  数据来源: {args.source}")
    print(f"  格式分布: {format_summary}")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*70}\n")

    if args.dry_run:
        for i, (f, fmt) in enumerate(files):
            rel = f.relative_to(data_root) if f.is_relative_to(data_root) else f
            print(f"  [{i+1:3d}] [{fmt:4s}] {rel}")
        print(f"\n  共 {len(files)} 个文件 (--dry-run，未执行导入)")
        return

    # 延迟导入 pipeline（需要 ES + Embedding 服务）
    from core.ingestion.pipeline import process_document

    success = 0
    failed = 0
    all_results: list[dict] = []

    for i in range(args.start, len(files)):
        f, fmt = files[i]
        rel = f.relative_to(data_root) if f.is_relative_to(data_root) else f
        print(f"\n{'─'*70}")
        print(f"[{i+1}/{len(files)}] [{fmt}] {f.stem}")
        print(f"  路径: {rel}")

        try:
            doc = process_document(str(f))
            saved_path = save_imported_doc(doc, output_dir)
            print(f"  doc_id: {doc.doc_id}")
            print_doc_stats(doc)
            print(f"  saved: {saved_path}")
            success += 1

            all_results.append(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "format": fmt,
                    "domain": doc.analysis.domain,
                    "doc_type": doc.analysis.doc_type,
                    "chunk_count": len(doc.chunks),
                    "status": "ok",
                }
            )

        except Exception as e:
            print(f"  FAILED: {e}")
            logger.exception(f"  导入失败: {f.stem}")
            failed += 1

            all_results.append(
                {
                    "title": f.stem,
                    "format": fmt,
                    "path": str(f),
                    "status": "failed",
                    "error": str(e),
                }
            )

    # ── 全局汇总 ──────────────────────────────────
    # 保存汇总
    summary_path = output_dir / "_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "total": len(files),
        "success": success,
        "failed": failed,
        "format_counts": format_counts,
        "results": all_results,
    }
    summary_path.write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*70}")
    print(f"  导入完成")
    print(f"  成功: {success}  失败: {failed}  总计: {len(files)}")
    print(f"  汇总: {summary_path}")

    # 按域统计
    domain_counts: dict[str, int] = Counter()
    doc_type_counts: dict[str, int] = Counter()
    for r in all_results:
        if r.get("status") == "ok":
            domain_counts[r.get("domain", "-")] += 1
            doc_type_counts[r.get("doc_type", "-")] += 1

    if domain_counts:
        print(f"\n  按域分布:")
        for domain, cnt in domain_counts.most_common():
            print(f"    {domain}: {cnt}")
    if doc_type_counts:
        print(f"\n  按文档类型分布:")
        for dtype, cnt in doc_type_counts.most_common():
            print(f"    {dtype}: {cnt}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
