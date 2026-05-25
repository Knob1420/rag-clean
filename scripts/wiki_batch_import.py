"""
Wiki 批量导入脚本

按文件夹顺序递归导入 md 文件到 wiki

用法：
    cd /home/zjlab/Documents/build_LLMs/NLP_course_hf/RAG/rag-clean
    python scripts/wiki_batch_import.py

数据源：
    /home/zjlab/Documents/build_LLMs/NLP_course_hf/RAG/data/threebody/2024-浙江省科技厅-尖兵-zj、国星-天基分布式计算系统-9900/归档文件/parsed

输出：
    data/wiki/
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# 配置
WIKI_DIR = Path("data/wiki-ds")
DOCS_DIR = Path(
    "/home/zjlab/Documents/build_LLMs/NLP_course_hf/RAG/data/threebody/2024-浙江省科技厅-尖兵-zj、国星-天基分布式计算系统-9900/归档文件/parsed"
)

# 导入 Wiki 相关模块
from core.wiki import WikiBuilder, WikiConfig, get_template, create_project_structure


def import_folder(builder, folder_path, folder_name, skip_existing=True):
    """导入单个文件夹下的所有 md 文件（按文件名排序）"""
    md_files = sorted(folder_path.rglob("*.md"))

    if not md_files:
        return 0, 0

    success = 0
    failed = 0
    skipped = 0

    # 获取已处理的文件列表（用于跳过）
    ingested_files: set = set()
    if skip_existing:
        log_file = builder.wiki_dir / "log.md"
        if log_file.exists():
            log_content = log_file.read_text(encoding="utf-8")
            import re
            ingested_files = set(re.findall(r'- \[ingest\] (.+)', log_content))
            logger.info(f"Found {len(ingested_files)} already-ingested files, will skip them")

    for f in md_files:
        # 跳过已处理的文件
        if skip_existing and f.name in ingested_files:
            print(f"  ⊝ {f.name} → already ingested, skipped")
            skipped += 1
            continue

        try:
            # 计算 folder_context（相对路径）
            folder_context = str(f.parent.relative_to(folder_path))

            result = builder.ingest_source(
                source_path=str(f),
                folder_context=folder_context,
            )

            pages = len(result.get("pages", []))
            reviews = len(result.get("reviews", []))

            status = "✓" if pages > 0 else "⚠"
            print(f"  {status} {f.name} → {pages} pages, {reviews} reviews")
            success += 1

            # 避免 API 过载，稍微延迟
            time.sleep(0.5)

        except Exception as e:
            print(f"  ✗ {f.name} → Error: {e}")
            failed += 1

    return success, failed, skipped


def main():
    print("=" * 70)
    print("  Wiki 批量导入")
    print("=" * 70)
    print(f"  数据源: {DOCS_DIR}")
    print(f"  输出:   {WIKI_DIR}")
    print("=" * 70)

    # 创建 wiki 项目结构（如果不存在）
    if not WIKI_DIR.exists() or not (WIKI_DIR / "purpose.md").exists():
        print("\n创建 wiki 项目结构...")
        create_project_structure(str(WIKI_DIR), "aerospace")
        print()

    # 初始化 WikiBuilder
    template = get_template("aerospace")
    config = WikiConfig(
        purpose=template.purpose,
        schema=template.schema,
        output_lang="zh",
    )
    builder = WikiBuilder(str(WIKI_DIR), config)

    # 按文件夹顺序处理
    folders = sorted(DOCS_DIR.iterdir(), key=lambda x: x.name)

    total_success = 0
    total_failed = 0

    for folder in folders:
        if not folder.is_dir():
            continue

        print(f"\n{'─'*70}")
        print(f"📁 {folder.name}")
        print(f"{'─'*70}")

        success, failed, skipped = import_folder(builder, folder, folder.name)
        total_success += success
        total_failed += failed

        print(f"  → 成功: {success}, 失败: {failed}, 跳过: {skipped}")

    # 完成
    print(f"\n{'='*70}")
    print(f"  完成")
    print(f"  成功: {total_success}  失败: {total_failed}")
    print(f"  Wiki: {WIKI_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
