"""
步骤1-3 主流水线脚本

用法：
    # 完整流水线（步骤1+2+3）
    python -m core.preprocessing.scripts.run_pipeline \
        --input data/cache/converters/*.md \
        --output data/preprocessing/

    # 仅步骤1（清洗+分片）
    python -m core.preprocessing.scripts.run_pipeline \
        --input data/cache/converters/*.md \
        --output data/preprocessing/ \
        --steps 1

    # 增量更新（单个新文档）
    python -m core.preprocessing.scripts.run_pipeline \
        --input data/cache/converters/new_doc.md \
        --output data/preprocessing/ \
        --incremental

    # 禁用 LLM（仅规则抽取，用于测试）
    python -m core.preprocessing.scripts.run_pipeline \
        --input data/cache/converters/*.md \
        --output data/preprocessing/ \
        --no-llm
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional
import hashlib
from loguru import logger

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 日志配置 ──────────────────────────────────────────────


def _setup_logging(verbose: bool = False):
    """配置 loguru 日志"""
    import logging

    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


# ── 步骤1：清洗 + 分片 ────────────────────────────────────


def run_step1(
    md_files: List[Path],
    output_dir: Path,
) -> List[dict]:
    """
    步骤1：清洗 + 分片

    Returns:
        chunks 列表
    """
    from core.preprocessing.chunker_ext import SmartChunker

    all_chunks: List[dict] = []

    for md_file in md_files:
        logger.info(f"[Step1] 处理文件: {md_file.name}")

        try:
            md_text = md_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"[Step1] 读取失败: {md_file}: {e}")
            continue

        if not md_text.strip():
            logger.warning(f"[Step1] 文件内容为空: {md_file}")
            continue

        # 从文件名推断 doc_type
        doc_type = _infer_doc_type(md_file.name)  ## 可能存在问题

        # 构造 doc_id
        # doc_id = f"doc_{md_file.stem[:16]}"
        doc_id = hashlib.md5(md_text.encode()).hexdigest()[:16]

        # 分块
        chunker = SmartChunker(
            source_file=md_file.name,
            doc_type=doc_type,
            source_weight=1.0,
        )

        try:
            documents = chunker.chunk(md_text, title=md_file.stem, doc_id=doc_id)
        except Exception as e:
            logger.warning(f"[Step1] 分块失败: {md_file}: {e}")
            continue

        # 转换为 dict
        for doc in documents:
            chunk_dict = {
                "content": doc.content,
                "metadata": doc.metadata,
            }
            all_chunks.append(chunk_dict)

        logger.info(
            f"[Step1] 分块完成: {len(documents)} documents, 累计 chunks={len(all_chunks)}"
        )

    # 保存步骤1结果
    step1_output = output_dir / "step1_chunks.json"
    with open(step1_output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"[Step1] 结果已保存: {step1_output}")

    return all_chunks


def _infer_doc_type(filename: str) -> str:
    """从文件名推断文档类型"""
    lower = filename.lower()
    if any(
        kw in lower
        for kw in ["技术要求", "技术规范", "试验规范", "试验大纲", "技术指标"]
    ):
        return "技术文档"
    if any(
        kw in lower for kw in ["汇报", "报告", "工作进展", "情况报告", "请示", "建议"]
    ):
        return "汇报材料"
    if any(kw in lower for kw in ["整体方案", "实施方案", "设计方案", "建设方案"]):
        return "方案"
    if any(kw in lower for kw in ["规范", "标准", "要求", "准则"]):
        return "规范"
    if any(kw in lower for kw in ["ppt", "pptx", "幻灯片", "演示"]):
        return "PPT资料"
    return "其他"


# ── 步骤2：实体 + 别名抽取 ─────────────────────────────────


async def run_step2(
    chunks: List[dict],
    output_dir: Path,
    ner_mode: str = "llm",
    merge: bool = False,
) -> tuple[List[dict], List[dict]]:
    """
    步骤2：全域实体 + 别名抽取

    Returns:
        (entity_raw_list, alias_candidates_list)
    """
    from core.preprocessing.entity_extractor import extract_entities, save_results

    logger.info(f"[Step2] 开始实体抽取，chunks={len(chunks)}, merge={merge}")

    # 转换格式（entity_extractor 需要的格式）
    chunk_inputs = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        chunk_inputs.append(
            {
                "content": chunk["content"],
                "chunk_id": meta.get("chunk_id", f"chunk_{i}"),
                "doc_id": meta.get("doc_id", ""),
                "doc_title": meta.get("doc_title", ""),
                "source_file": meta.get("source_file", ""),
            }
        )

    entity_raw, alias_candidates = await extract_entities(
        chunk_inputs, ner_mode=ner_mode
    )

    # 保存（merge=True 时自动合并已有结果）
    paths = save_results(entity_raw, alias_candidates, str(output_dir), merge=merge)

    return entity_raw, alias_candidates


# ── 步骤3：结构化抽取 ──────────────────────────────────────


async def run_step3(
    chunks: List[dict],
    output_dir: Path,
    llm_enabled: bool = True,
    incremental: bool = False,
) -> dict:
    """
    步骤3：结构化专项抽取（product_params + cooperation）

    Returns:
        {"product_params": [...], "cooperation": [...]}
    """
    from core.preprocessing.struct_extractor import (
        extract_product_params,
        extract_cooperation,
        save_results as save_struct,
    )

    logger.info(f"[Step3] 开始结构化抽取，chunks={len(chunks)}")

    # 加载 products_specs.json
    products_specs_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "products_specs.json"
    )
    from core.preprocessing.struct_extractor import _load_products_specs

    products_specs = _load_products_specs(products_specs_path)

    # 加载 entity_raw（优先用清洗后的，否则用原始的）
    entity_raw_path = output_dir / "step2_entity_raw_cleaned.json"
    if not entity_raw_path.exists():
        entity_raw_path = output_dir / "step2_entity_raw.json"
    if entity_raw_path.exists():
        with open(entity_raw_path, "r", encoding="utf-8") as f:
            entity_raw = json.load(f)
    else:
        entity_raw = []
        logger.warning("[Step3] 未找到 entity_raw 文件，跳过结构化抽取")

    logger.info(f"[Step3] 加载 entity_raw {len(entity_raw)} 条")

    # 转换 chunk 格式
    chunk_inputs = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        chunk_inputs.append(
            {
                "content": chunk["content"],
                "chunk_id": meta.get("chunk_id", f"chunk_{i}"),
                "doc_id": meta.get("doc_id", ""),
                "doc_title": meta.get("doc_title", ""),
                "source_file": meta.get("source_file", ""),
            }
        )

    # 并发执行两个抽取任务
    product_params, cooperation = await asyncio.gather(
        extract_product_params(
            products_specs,
            entity_raw,
            chunk_inputs,
            llm_enabled=llm_enabled,
        ),
        extract_cooperation(
            entity_raw,
            chunk_inputs,
            llm_enabled=llm_enabled,
        ),
    )

    # 保存（支持增量合并）
    paths = save_struct(
        product_params,
        cooperation,
        str(output_dir),
        merge=incremental,
    )

    return {"product_params": product_params, "cooperation": cooperation}


# ── 步骤4：本体构建 ───────────────────────────────────────────


async def run_step4(
    output_dir: Path,
    incremental: bool = False,
) -> dict:
    """
    步骤4：本体构建

    从 entity_raw + product_params + cooperation 构建本体图。

    Returns:
        {"metadata": {...}, "nodes": [...], "edges": [...]}
    """
    from core.preprocessing.ontology_builder import (
        load_entity_raw,
        load_product_params,
        load_cooperation,
        build_graph,
        save_graph,
    )

    logger.info("[Step4] 开始本体构建")

    # 加载三个数据源
    entity_raw_path = output_dir / "step2_entity_raw_cleaned.json"
    product_params_path = output_dir / "step3_product_params.json"
    cooperation_path = output_dir / "step3_cooperation.json"

    if not entity_raw_path.exists():
        logger.warning(f"[Step4] 未找到 entity_raw: {entity_raw_path}，跳过")
        return {}
    if not product_params_path.exists():
        logger.warning(f"[Step4] 未找到 product_params: {product_params_path}，跳过")
        return {}
    if not cooperation_path.exists():
        logger.warning(f"[Step4] 未找到 cooperation: {cooperation_path}，跳过")
        return {}

    entity_raw = load_entity_raw(str(entity_raw_path))
    product_params = load_product_params(str(product_params_path))
    cooperation = load_cooperation(str(cooperation_path))

    logger.info(f"[Step4] 加载数据: entity_raw={len(entity_raw)}, product_params={len(product_params)}, cooperation={len(cooperation)}")

    # 构建图
    graph = build_graph(entity_raw, product_params, cooperation)

    # 保存
    out_path = output_dir / "step4_ontology.json"
    save_graph(graph, str(out_path))

    logger.info(
        f"[Step4] 本体构建完成: {graph['metadata']['total_nodes']} 节点, "
        f"{graph['metadata']['total_edges']} 边"
    )

    return graph


# ── 主流水线 ──────────────────────────────────────────────


async def run_pipeline(
    input_path: str,
    output_dir: str,
    steps: str = "1,2,3",
    incremental: bool = False,
    ner_mode: str = "llm",
    llm_enabled: bool = True,
    verbose: bool = False,
):
    """
    主流水线

    Args:
        input_path: 输入文件/目录（支持 glob 模式如 data/*.md）
        output_dir: 输出目录
        steps: 执行的步骤，如 "1,2,3" 或 "1,2" 或 "3"
        incremental: 增量模式（追加而非覆盖）
        ner_mode: NER 模式，"llm"（默认）/ "hanlp" / "none"
        llm_enabled: 是否启用 LLM
        verbose: 详细日志
    """
    _setup_logging(verbose)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 解析步骤
    step_list = [int(s.strip()) for s in steps.split(",")]

    logger.info(f"=" * 60)
    logger.info(f"MD 预处理流水线启动")
    logger.info(f"输入: {input_path}")
    logger.info(f"输出: {output_dir}")
    logger.info(f"步骤: {steps}")
    logger.info(f"增量: {incremental}")
    logger.info(f"=" * 60)

    # 1. 收集输入文件
    input_p = Path(input_path)
    if input_p.is_dir():
        md_files = sorted(input_p.glob("*.md"))
    elif input_p.is_file():
        md_files = [input_p]
    else:
        # glob 模式
        md_files = sorted(Path(".").glob(input_path))

    if not md_files:
        logger.error(f"未找到输入文件: {input_path}")
        return

    logger.info(f"找到 {len(md_files)} 个 MD 文件")

    # 2. 执行步骤
    chunks: List[dict] = []
    entity_raw: List[dict] = []
    alias_candidates: List[dict] = []
    struct_results: dict = {}

    if 1 in step_list:
        chunks = run_step1(md_files, output_path)

    if 2 in step_list:
        if not chunks:
            # 从 step1 输出加载
            step1_file = output_path / "step1_chunks.json"
            if step1_file.exists():
                with open(step1_file, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                logger.info(f"[Step2] 从缓存加载 {len(chunks)} chunks")

        entity_raw, alias_candidates = await run_step2(
            chunks, output_path, ner_mode=ner_mode, merge=incremental
        )

    if 3 in step_list:
        if not chunks:
            step1_file = output_path / "step1_chunks.json"
            if step1_file.exists():
                with open(step1_file, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                logger.info(f"[Step3] 从缓存加载 {len(chunks)} chunks")

        struct_results = await run_step3(
            chunks,
            output_path,
            llm_enabled=llm_enabled,
            incremental=incremental,
        )

    if 4 in step_list:
        graph = await run_step4(output_path, incremental=incremental)
        if graph:
            logger.info(f"步骤4: {graph['metadata']['total_nodes']} 节点, {graph['metadata']['total_edges']} 边")

    # 3. 完成汇总
    logger.info(f"=" * 60)
    logger.info(f"流水线完成")
    logger.info(f"步骤1: {len(chunks)} chunks")
    logger.info(f"步骤2: {len(entity_raw)} 实体, {len(alias_candidates)} 别名候选")
    logger.info(
        f"步骤3: product_params={len(struct_results.get('product_params', []))}, "
        f"cooperation={len(struct_results.get('cooperation', []))}"
    )
    if 4 in step_list:
        logger.info(f"步骤4: {graph.get('metadata', {}).get('total_nodes', 0)} 节点, {graph.get('metadata', {}).get('total_edges', 0)} 边")
    logger.info(f"=" * 60)


# ── CLI 入口 ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="MD 预处理流水线（步骤1-4）")
    parser.add_argument(
        "--input",
        default="/home/zjlab/Documents/build_LLMs/NLP_course_hf/RAG/data/raw/产品",
        help="输入文件/目录/glob 模式",
    )
    parser.add_argument("--output", default="data/preprocessing/", help="输出目录")
    parser.add_argument("--steps", default="1,2,3", help="执行的步骤，如 1,2,3,4")
    parser.add_argument("--incremental", action="store_true", help="增量模式")
    parser.add_argument(
        "--ner-mode",
        default="llm",
        choices=["llm", "hanlp", "none"],
        help="NER 模式（默认 llm）",
    )
    parser.add_argument(
        "--no-llm", dest="llm_enabled", action="store_false", help="禁用 LLM"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    parser.add_argument("--dry-run", action="store_true", help="仅检查输入不执行")

    args = parser.parse_args()

    if args.dry_run:
        input_p = Path(args.input)
        if input_p.is_dir():
            files = list(input_p.glob("*.md"))
        elif input_p.is_file():
            files = [input_p]
        else:
            files = sorted(Path(".").glob(args.input))
        print(f"找到 {len(files)} 个文件:")
        for f in files[:10]:
            print(f"  {f}")
        if len(files) > 10:
            print(f"  ... 还有 {len(files) - 10} 个")
        return

    import asyncio

    asyncio.run(
        run_pipeline(
            input_path=args.input,
            output_dir=args.output,
            steps=args.steps,
            incremental=args.incremental,
            ner_mode=args.ner_mode,
            llm_enabled=args.llm_enabled,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
