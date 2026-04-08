"""
管道编排 — 数据处理单一入口

核心设计：
- 一条路径：不像旧项目有 upload_document 和 upload_document_with_enrichment 两条路
- 分析在分块之前：enrichment 只做 chunk 级别（doc 级别已经由 analyzer 完成）
- 简单脚本可以一行调用：process_document("path/to/G1.md")
- 多格式支持：PDF、DOC、DOCX、PPTX → 统一走 Markdown pipeline
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

from models import Chunk, DocumentAnalysis, ProcessedDocument
from core.ingestion.analyzer import DocumentAnalyzer
from core.ingestion.chunker import SmartChunker
from store import DocumentStore, get_store
from core.generation.llm import LLMClient, get_llm_client
from core.retrieve.embedder import encode
from config import settings
from core.ingestion.converters import detect_format, convert_to_markdown


def process_document(
    file_path: str,
    title: Optional[str] = None,
    store: Optional[DocumentStore] = None,
    analyzer: Optional[DocumentAnalyzer] = None,
    chunker: Optional[SmartChunker] = None,
    llm: Optional[LLMClient] = None,
) -> ProcessedDocument:
    """
    唯一的数据处理入口（从文件路径）。

    流程：读取文件 → 自动格式检测与转换 → 委托 process_markdown 处理
    """
    path = Path(file_path)
    fmt = detect_format(path)

    if fmt == "md":
        md = path.read_text(encoding="utf-8")
    else:
        logger.info(f"[Pipeline] 检测到 {fmt} 格式，开始转换: {path.name}")
        md = convert_to_markdown(file_path)

    title = title or path.stem

    return process_markdown(
        content=md,
        title=title,
        store=store,
        analyzer=analyzer,
        chunker=chunker,
        llm=llm,
    )


def process_markdown(
    content: str,
    title: str,
    doc_id: Optional[str] = None,
    store: Optional[DocumentStore] = None,
    analyzer: Optional[DocumentAnalyzer] = None,
    chunker: Optional[SmartChunker] = None,
    llm: Optional[LLMClient] = None,
) -> ProcessedDocument:
    """
    直接处理 Markdown 文本（不需要文件路径）。

    Args:
        content: Markdown 文本
        title: 文档标题
        doc_id: 文档 ID（默认自动生成）
        store: ES 存储
        analyzer: 文档分析器
        chunker: 分块器
        llm: LLM 客户端

    Returns:
        处理后的完整文档
    """
    store = store or get_store()
    llm = llm or get_llm_client()
    analyzer = analyzer or DocumentAnalyzer(llm_client=llm)
    chunker = chunker or SmartChunker()

    if doc_id is None:
        doc_id = hashlib.md5(content.encode()).hexdigest()[:16]

    logger.info(
        f"[Pipeline] 开始处理文本: {title} (doc_id={doc_id}, {len(content)} chars)"
    )

    # 检查 checkpoint
    cached = _load_checkpoint(doc_id)
    if cached is not None:
        analysis, chunks = cached
        logger.info(f"  [Checkpoint] 命中缓存，跳过 analyzer + enrichment")
    else:
        analysis = analyzer.analyze(title, content)
        chunks = chunker.chunk(content, doc_id, analysis)
        _enrich_chunks(chunks, analysis, llm, title)
        _save_checkpoint(doc_id, analysis, chunks)

    _embed_chunks(chunks, analysis)

    doc = ProcessedDocument(
        doc_id=doc_id,
        title=title,
        analysis=analysis,
        chunks=chunks,
        content=content,
    )

    store.ensure_indices()
    success = store.index_document(doc)
    logger.info(f"[Pipeline] 处理完成: {success}/{len(chunks)} chunks 已索引")

    return doc


# ── 内部函数 ──────────────────────────────────────────


def _enrich_chunks(
    chunks: list[Chunk],
    analysis: DocumentAnalysis,
    llm: LLMClient,
    title: str,
    batch_size: int = 5,
):
    """批量 chunk enrichment: chunk_type, keywords, context_summary"""
    if not chunks:
        return

    logger.info(f"  [Enrichment] 开始处理 {len(chunks)} chunks")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        batch_data = [
            {
                "chunk_index": j,
                "content": chunk.content[:500],
                "section_title": chunk.section_title,
            }
            for j, chunk in enumerate(batch)
        ]

        chunks_info = llm.extract_chunk_info_batch(
            chunks=batch_data,
            doc_context={
                "doc_type": analysis.doc_type,
                "global_entities": analysis.entities,
                "title": title,
            },
        )

        for chunk_info in chunks_info:
            idx = chunk_info.get("index", 0)
            if idx < len(batch):
                chunk = batch[idx]
                # enrichment 始终设置最终 chunk_type（包括表格 chunk 的初始 hint）
                chunk.chunk_type = chunk_info.get("chunk_type", chunk.chunk_type)
                if chunk_info.get("section_title"):
                    chunk.section_title = chunk_info["section_title"]
                # keywords 可能是嵌套 list 或含非 str 元素，统一展平
                raw_kw = chunk_info.get("keywords", [])
                flat_kw = []
                for item in raw_kw if isinstance(raw_kw, list) else [raw_kw]:
                    if isinstance(item, list):
                        flat_kw.extend(str(x) for x in item)
                    else:
                        flat_kw.append(str(item))
                chunk.keywords = flat_kw
                chunk.context_summary = chunk_info.get("context_summary", "")

        logger.info(f"    进度: {min(i + batch_size, len(chunks))}/{len(chunks)}")

    logger.info(f"  [Enrichment] 完成")


def _embed_chunks(chunks: list[Chunk], analysis: DocumentAnalysis):
    """计算每个 chunk 的向量 — 只用 keywords + context_summary"""
    for chunk in chunks:
        parts = []

        # keywords（防御旧 checkpoint 中可能存在的嵌套 list）
        if chunk.keywords:
            flat = []
            for k in chunk.keywords:
                if isinstance(k, list):
                    flat.extend(str(x) for x in k)
                else:
                    flat.append(str(k))
            parts.append(" ".join(flat))

        # context_summary（enrichment 未生成时 fallback 到 content）
        parts.append(chunk.context_summary or chunk.content)

        text_for_embedding = " ".join(parts)

        vector = encode(text_for_embedding)
        if vector is not None:
            chunk._embedding_vector = vector.tolist()
        else:
            chunk._embedding_vector = None


# ── Checkpoint：缓存 enrichment 结果 ──────────────────────


def _checkpoint_path(doc_id: str) -> Path:
    return Path(settings.cache_dir) / f"{doc_id}.json"


def _save_checkpoint(doc_id: str, analysis: DocumentAnalysis, chunks: list[Chunk]):
    """enrichment 完成后保存，跳过后续重复的 LLM 调用"""
    cache_dir = Path(settings.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "analysis": analysis.to_dict(),
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    _checkpoint_path(doc_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
    )
    logger.info(f"  [Checkpoint] 已保存: {doc_id}")


def _load_checkpoint(
    doc_id: str,
) -> Optional[Tuple[DocumentAnalysis, list[Chunk]]]:
    """加载缓存的 enrichment 结果，返回 None 表示未命中"""
    path = _checkpoint_path(doc_id)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        analysis = DocumentAnalysis.from_dict(data["analysis"])
        chunks = [Chunk.from_dict(c) for c in data["chunks"]]
        logger.info(f"  [Checkpoint] 命中: {doc_id}")
        return analysis, chunks
    except Exception as e:
        logger.warning(f"  [Checkpoint] 加载失败，将重新处理: {e}")
        return None
