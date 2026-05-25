"""
批量解析归档文件目录下的所有文档
支持: .pdf, .doc, .docx, .pptx, .md
"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.ingestion.extractor import convert_to_markdown, detect_format, _ensure_mineru
from loguru import logger
import traceback


def parse_directory(input_dir: str, output_dir: str = None):
    """批量解析目录下的所有文档"""
    input_path = Path(input_dir)

    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = input_path / "parsed"
    output_path.mkdir(parents=True, exist_ok=True)

    # 统计
    stats = {"total": 0, "success": 0, "failed": 0}

    # 查找所有支持的文件
    supported_exts = {".pdf", ".doc", ".docx", ".pptx", ".md"}
    files = []
    for ext in supported_exts:
        files.extend(input_path.rglob(f"*{ext}"))

    # 排除临时文件
    files = [f for f in files if not f.name.startswith("~$")]

    stats["total"] = len(files)
    logger.info(f"找到 {stats['total']} 个文件待解析")

    for file_path in sorted(files):
        relative_path = file_path.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix(".md")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fmt = detect_format(str(file_path))
        logger.info(f"解析 [{fmt}]: {relative_path}")

        try:
            md_content = convert_to_markdown(str(file_path))
            output_file.write_text(md_content, encoding="utf-8")
            logger.success(f"  -> {output_file.relative_to(output_path)} ({len(md_content)} chars)")
            stats["success"] += 1
        except Exception as e:
            logger.error(f"  失败: {e}")
            logger.debug(traceback.format_exc())
            stats["failed"] += 1

    # 打印统计
    logger.info("=" * 50)
    logger.info(f"解析完成: 总={stats['total']}, 成功={stats['success']}, 失败={stats['failed']}")
    logger.info(f"输出目录: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/batch_parse_docs.py <目录路径> [输出目录]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # 确保 MinerU 可用
    _ensure_mineru()

    parse_directory(input_dir, output_dir)