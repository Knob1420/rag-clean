"""
批量导入 — 从 rag-knowledge-base 的已解析 markdown 文件导入到 ES

用法：
    python scripts/import_docs.py
    python scripts/import_docs.py --dir /path/to/markdown/files
    python scripts/import_docs.py --file /path/to/specific.md --title "自定义标题"
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from core.ingestion.pipeline import process_document


def import_from_directory(docs_dir: str):
    """
    从目录中批量导入文档。

    查找每个子目录中的 markdown 文件并处理。
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logger.error(f"目录不存在: {docs_path}")
        return

    # 查找所有 markdown 文件
    md_files = list(docs_path.rglob("*.md"))
    if not md_files:
        logger.warning(f"未找到 markdown 文件: {docs_path}")
        return

    logger.info(f"找到 {len(md_files)} 个 markdown 文件")

    success = 0
    failed = 0

    for md_file in md_files:
        try:
            logger.info(f"处理: {md_file.name}")
            doc = process_document(str(md_file))
            logger.info(f"  完成: {doc.title}, {len(doc.chunks)} chunks")
            success += 1
        except Exception as e:
            logger.error(f"  失败: {md_file.name}: {e}")
            failed += 1

    logger.info(f"导入完成: {success} 成功, {failed} 失败")


def import_from_rag_knowledge_base():
    """从 rag-knowledge-base 的 pending/output 目录导入"""
    base_dir = Path(__file__).parent.parent.parent
    pending_dir = base_dir / "rag-knowledge-base" / "data" / "raw" / "pending" / "output"

    if not pending_dir.exists():
        logger.error(f"rag-knowledge-base pending 目录不存在: {pending_dir}")
        return

    # 遍历子目录，每个子目录是一个文档
    for doc_dir in sorted(pending_dir.iterdir()):
        if not doc_dir.is_dir():
            continue

        # 查找子目录中的 md 文件（可能在嵌套目录中）
        md_files = list(doc_dir.rglob("*.md"))
        if not md_files:
            continue

        # 使用找到的第一个 md 文件
        md_file = md_files[0]
        logger.info(f"导入: {doc_dir.name}")
        try:
            doc = process_document(str(md_file))
            logger.info(f"  完成: {doc.title}, {len(doc.chunks)} chunks")
        except Exception as e:
            logger.error(f"  失败: {doc_dir.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="批量导入文档到 ES")
    parser.add_argument(
        "--dir",
        type=str,
        help="包含 markdown 文件的目录路径",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="单个 markdown 文件路径",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="文档标题（仅 --file 模式）",
    )
    parser.add_argument(
        "--from-kb",
        action="store_true",
        help="从 rag-knowledge-base 的 pending 目录导入",
    )

    args = parser.parse_args()

    if args.file:
        # 单文件模式
        doc = process_document(args.file, title=args.title)
        logger.info(f"完成: {doc.title}, {len(doc.chunks)} chunks")
    elif args.dir:
        # 目录模式
        import_from_directory(args.dir)
    elif args.from_kb:
        # 从 rag-knowledge-base 导入
        import_from_rag_knowledge_base()
    else:
        # 默认：从 rag-knowledge-base 导入
        import_from_rag_knowledge_base()


if __name__ == "__main__":
    main()
