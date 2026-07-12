"""
管道编排 — 数据处理单一入口

核心设计：
- 一条路径：不像旧项目有 upload_document 和 upload_document_with_enrichment 两条路
- 不用 analyzer：分析信息直接从 Document.metadata 获取（chunker 已处理）
- 支持中间结果保存和加载
- 多格式支持：PDF、DOC、DOCX、PPTX → 统一走 Markdown pipeline

输出数据结构：
- Document: 主块 + children (List[ChildDocument]) + attachments
- 每个 Document 代表一个 parent chunk
- ChildDocument 是用于检索的最小单元
- 索引时同时存 parent 和 child 到 ES，通过 parent_id 关联
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, List

from loguru import logger

from core.model.models import Document, SummaryDocument, ChildDocument
from core.ingestion.chunker import SmartChunker
from core.ingestion.cleaner import clean_text
from store import DocumentStore, get_store
from core.client.embedder import encode, encode_batch
from core.ingestion.extractor import detect_format, convert_to_markdown

# ── 目录配置 ──────────────────────────────────────────
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def process_document(
    file_path: str,
    title: Optional[str] = None,
    dataset_id: Optional[str] = None,
    store: Optional[DocumentStore] = None,
    chunker: Optional[SmartChunker] = None,
    save_intermediate: bool = True,
    load_intermediate: bool = False,
    processed_dir: Optional[Path] = None,
    use_summary: bool = True,
    chunk_mode: str = "recursive",
    source_key: Optional[str] = None,
) -> List[Document]:
    """
    唯一的数据处理入口（从文件路径）。

    流程：读取文件 → 自动格式检测与转换 → 委托 process_markdown 处理

    Args:
        file_path: 文件路径
        title: 文档标题（默认用文件名）
        dataset_id: 知识库 ID（默认从父目录名获取）
        store: ES 存储
        chunker: 分块器
        save_intermediate: 是否保存中间结果到 JSON
        load_intermediate: 是否从 JSON 加载中间结果（跳过处理）
        processed_dir: 中间结果保存目录（默认使用 PROCESSED_DIR）
        use_summary: 是否生成 summary（默认 True，关闭可节省处理时间）
        chunk_mode: 分块模式，"recursive"（默认）或 "semantic"
        source_key: 版本键；None 时自动生成 f"{dataset_id}::{file_stem}"
                    同 source_key 视为同文档的新版本，旧版本会被标记 is_latest=False

    Returns:
        List[Document]: 处理后的 Document 列表
    """
    path = Path(file_path)
    fmt = detect_format(path)

    if fmt == "md":
        md = path.read_text(encoding="utf-8")
    else:
        logger.info(f"[Pipeline] 检测到 {fmt} 格式，开始转换: {path.name}")
        md = convert_to_markdown(file_path)

    title = title or path.stem

    # dataset_id 默认从父目录名获取
    if dataset_id is None:
        dataset_id = path.parent.name

    # source_key: None（默认）= 不启用版本管理；调用方显式传入才启用
    return process_markdown(
        content=md,
        title=title,
        doc_id=None,
        dataset_id=dataset_id,
        store=store,
        chunker=chunker,
        save_intermediate=save_intermediate,
        load_intermediate=load_intermediate,
        processed_dir=processed_dir,
        use_summary=use_summary,
        chunk_mode=chunk_mode,
        source_key=source_key,
    )


def process_markdown(
    content: str,
    title: str,
    doc_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    store: Optional[DocumentStore] = None,
    chunker: Optional[SmartChunker] = None,
    save_intermediate: bool = True,
    load_intermediate: bool = False,
    processed_dir: Optional[Path] = None,
    use_summary: bool = True,
    chunk_mode: str = "recursive",
    source_key: str = "",
) -> List[Document]:
    """
    直接处理 Markdown 文本（不需要文件路径）。

    流程：
    1. 清洗文本（cleaner）
    2. 父子分块（chunker）
    3. 计算向量（embedder）
    4. 生成 summary（可选，默认开启）
    5. 保存中间结果（可选）
    6. 索引到 ES

    Args:
        content: Markdown 文本
        title: 文档标题
        doc_id: 文档 ID（默认自动生成）
        dataset_id: 知识库 ID（用于中间结果路径）
        store: ES 存储
        chunker: 分块器
        save_intermediate: 是否保存中间结果到 JSON
        load_intermediate: 是否从 JSON 加载中间结果
        processed_dir: 中间结果保存目录（默认使用 PROCESSED_DIR）
        use_summary: 是否生成 summary（默认 True）
        chunk_mode: 分块模式，"recursive"（默认）或 "semantic"
        source_key: 版本键；同 key 视为同文档的新版本

    Returns:
        List[Document]: 处理后的 Document 列表
    """
    store = store or get_store()
    chunker = chunker or SmartChunker()
    processed_dir = processed_dir or PROCESSED_DIR

    if doc_id is None:
        doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    logger.info(
        f"[Pipeline] 开始处理文本: {title} (doc_id={doc_id}, dataset_id={dataset_id}, "
        f"source_key={source_key or '-'})"
    )

    # 尝试加载中间结果（跳过 chunker，但仍跑 summary + embed + index）
    if load_intermediate and dataset_id and title:
        documents = _load_intermediate(dataset_id, title, processed_dir)
        if documents is not None:
            logger.info(f"  [Intermediate] 从缓存加载: {len(documents)} documents")

            # 补跑 summary（batch_chunker 产物不含 summary，需要 LLM 生成）
            if use_summary and any(not d.summaries for d in documents):
                _generate_summaries(documents, doc_id)
                logger.info(f"  [Summary] 补跑完成（从缓存加载后）")
                # 更新 JSON（含 summary 后重新保存）
                if save_intermediate and dataset_id and title:
                    _save_intermediate(documents, dataset_id, title, processed_dir)

            # 重新计算向量（含 summary 的 vector）
            _embed_documents(documents)
            _add_dataset_id(documents, dataset_id)
            _inject_source_key(documents, source_key)
            _index_documents(store, doc_id, documents)
            logger.info(f"[Pipeline] 处理完成（从缓存）: {len(documents)} documents")
            return documents

    # 1. 父子分块（chunker 内部调用 clean_text）
    documents = chunker.chunk(content, title, doc_id, mode=chunk_mode, source_key=source_key)
    logger.info(f"  [Chunker] 分块完成: {len(documents)} parent documents")

    # 2. 生成 summary（每个 parent 一个 summary chunk，可选）
    if use_summary:
        _generate_summaries(documents, doc_id)
        logger.info(f"  [Summary] 生成完成")

    # 3. 计算向量
    _embed_documents(documents)

    # 4. 添加 dataset_id
    if dataset_id:
        _add_dataset_id(documents, dataset_id)

    # 5. 保存中间结果
    if save_intermediate and dataset_id and title:
        _save_intermediate(documents, dataset_id, title, processed_dir)

    # 6. 索引到 ES
    _index_documents(store, doc_id, documents)
    logger.info(f"[Pipeline] 处理完成: {len(documents)} documents")

    return documents


def _save_intermediate(
    documents: List[Document], dataset_id: str, title: str, processed_dir: Path
) -> Path:
    """保存中间结果到 JSON（不包含 vector，vector 仅存 ES）"""
    import copy

    output_dir = processed_dir / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = title
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        safe_name = safe_name.replace(ch, "_")
    out_path = output_dir / f"{safe_name}.json"

    # 深拷贝并清除 vector，避免 JSON 无法序列化 numpy array，也减小文件体积
    data = []
    for doc in documents:
        doc_copy = copy.deepcopy(doc)
        doc_copy.vector = None
        if doc_copy.children:
            for child in doc_copy.children:
                child.vector = None
        if doc_copy.summaries:
            for summary in doc_copy.summaries:
                summary.vector = None
        data.append(doc_copy.to_dict())

    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"  [Intermediate] 已保存: {out_path}")
    return out_path


def _load_intermediate(
    dataset_id: str, title: str, processed_dir: Path
) -> Optional[List[Document]]:
    """从 JSON 加载中间结果"""
    output_dir = processed_dir / dataset_id
    safe_name = title
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        safe_name = safe_name.replace(ch, "_")
    json_path = output_dir / f"{safe_name}.json"

    if not json_path.exists():
        return None

    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [Document.from_dict(d) for d in data]


def _add_dataset_id(documents: List[Document], dataset_id: str):
    """给所有文档添加 dataset_id"""
    for doc in documents:
        doc.metadata["dataset_id"] = dataset_id
        if doc.children:
            for child in doc.children:
                child.metadata["dataset_id"] = dataset_id
        if doc.summaries:
            for summary in doc.summaries:
                summary.metadata["dataset_id"] = dataset_id


def _inject_source_key(documents: List[Document], source_key: str):
    """给所有文档注入 source_key（从中间结果加载时，旧缓存可能没有该字段）"""
    if not source_key:
        return
    for doc in documents:
        doc.metadata["source_key"] = source_key
        if doc.children:
            for child in doc.children:
                child.metadata["source_key"] = source_key
        if doc.summaries:
            for summary in doc.summaries:
                summary.metadata["source_key"] = source_key


def _index_documents(store: DocumentStore, doc_id: str, documents: List[Document]):
    """索引文档到 ES"""
    store.ensure_indices()
    success = store.index_document(doc_id, documents)
    logger.info(f"  [ES] 已索引: {success} chunks")


# ── 内部函数 ──────────────────────────────────────────


def _generate_summaries(documents: List[Document], doc_id: str):
    """
    为每个 Document 生成 summary chunk（批量并发 LLM 调用）。

    summary 作为特殊的 child chunk 保存，关联到对应的 parent_id。
    同时设置 doc.summary、doc.primary_entity，以及每个 child 的 primary_entity。
    """
    if not documents:
        return

    from core.generation.llm import get_llm_client
    llm = get_llm_client()

    # 批量并发生成 summaries
    contents = [doc.content for doc in documents]
    results = llm.generate_summary_batch(contents)

    # 回填结果到 documents
    summary_idx = 0
    for doc, result in zip(documents, results):
        parent_id = doc.metadata.get("chunk_id", "")

        summary_content = result["summary"]
        primary_entity = result["primary_entity"]

        # 设置 parent chunk 的 summary 和 primary_entity 字段
        doc.summary = summary_content
        doc.primary_entity = primary_entity

        # 创建 summary chunk
        summary_id = f"{doc_id}_s{summary_idx}"
        summary_meta = {
            "doc_id": doc_id,
            "doc_title": doc.metadata.get("doc_title", ""),
            "chunk_id": summary_id,
            "parent_id": parent_id,
            "doc_hash": doc.metadata.get("doc_hash", ""),
        }
        # 透传 dataset_id（如果 parent 有）
        if "dataset_id" in doc.metadata:
            summary_meta["dataset_id"] = doc.metadata["dataset_id"]
        # 透传 source_key（如果 parent 有）
        if "source_key" in doc.metadata:
            summary_meta["source_key"] = doc.metadata["source_key"]

        summary_doc = SummaryDocument(
            content=summary_content,
            primary_entity=primary_entity,
            metadata=summary_meta,
        )
        summary_idx += 1

        # 设置每个 child 的 primary_entity 字段
        if doc.children:
            for child in doc.children:
                child.primary_entity = primary_entity

        # 保存 summary 到 document.summaries
        doc.summaries = [summary_doc]


def _embed_documents(documents: List[Document]):
    """
    批量计算每个 Document 及其 children、summaries 的向量。

    策略：收集所有文本，一次 encode_batch 调用，再逐个回填。
    """
    # 1. 收集所有待向量化的文本及其引用
    texts: list[str] = []
    refs: list[tuple[object, str]] = []  # (doc/child/summary, field)

    for doc in documents:
        texts.append(doc.content)
        refs.append((doc, "vector"))

        if doc.children:
            for child in doc.children:
                texts.append(child.content)
                refs.append((child, "vector"))

        if doc.summaries:
            for summary in doc.summaries:
                texts.append(summary.content)
                refs.append((summary, "vector"))

    # 2. 批量向量化
    embeddings = encode_batch(texts)

    # 3. 回填结果
    for (obj, field), embedding in zip(refs, embeddings):
        if embedding is not None:
            value = embedding.tolist() if hasattr(embedding, "tolist") else embedding
            setattr(obj, field, value)
