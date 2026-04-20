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

from core.model.models import Document
from core.ingestion.chunker import SmartChunker
from core.ingestion.cleaner import clean_text
from store import DocumentStore, get_store
from core.retrieve.embedder import encode
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

    return process_markdown(
        content=md,
        title=title,
        doc_id=None,
        dataset_id=dataset_id,
        filename=path.name,
        store=store,
        chunker=chunker,
        save_intermediate=save_intermediate,
        load_intermediate=load_intermediate,
        processed_dir=processed_dir,
    )


def process_markdown(
    content: str,
    title: str,
    doc_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    filename: Optional[str] = None,
    store: Optional[DocumentStore] = None,
    chunker: Optional[SmartChunker] = None,
    save_intermediate: bool = True,
    load_intermediate: bool = False,
    processed_dir: Optional[Path] = None,
) -> List[Document]:
    """
    直接处理 Markdown 文本（不需要文件路径）。

    流程：
    1. 清洗文本（cleaner）
    2. 父子分块（chunker）
    3. 计算向量（embedder）
    4. 保存中间结果（可选）
    5. 索引到 ES

    Args:
        content: Markdown 文本
        title: 文档标题
        doc_id: 文档 ID（默认自动生成）
        dataset_id: 知识库 ID（用于中间结果路径）
        filename: 原文件名（用于中间结果路径）
        store: ES 存储
        chunker: 分块器
        save_intermediate: 是否保存中间结果到 JSON
        load_intermediate: 是否从 JSON 加载中间结果
        processed_dir: 中间结果保存目录（默认使用 PROCESSED_DIR）

    Returns:
        处理后的 Document 列表
    """
    store = store or get_store()
    chunker = chunker or SmartChunker()
    processed_dir = processed_dir or PROCESSED_DIR

    if doc_id is None:
        doc_id = hashlib.md5(content.encode()).hexdigest()[:16]

    logger.info(
        f"[Pipeline] 开始处理文本: {title} (doc_id={doc_id}, dataset_id={dataset_id})"
    )

    # 尝试加载中间结果
    if load_intermediate and dataset_id and filename:
        documents = _load_intermediate(dataset_id, filename, processed_dir)
        if documents is not None:
            logger.info(f"  [Intermediate] 从缓存加载: {len(documents)} documents")
            # 重新计算向量（缓存中没有向量）
            _embed_documents(documents)
            _add_dataset_id(documents, dataset_id)
            _index_documents(store, doc_id, documents)
            return documents

    # 1. 清洗文本
    cleaned = clean_text(content, remove_images=True)
    logger.info(f"  [Cleaner] 清洗完成: {len(cleaned)} chars")

    # 2. 父子分块
    documents = chunker.chunk(cleaned, title, doc_id)
    logger.info(f"  [Chunker] 分块完成: {len(documents)} parent documents")

    # 3. 计算向量
    _embed_documents(documents)

    # 4. 添加 dataset_id
    if dataset_id:
        _add_dataset_id(documents, dataset_id)

    # 5. 保存中间结果
    if save_intermediate and dataset_id and filename:
        _save_intermediate(documents, dataset_id, filename, processed_dir)

    # 6. 索引到 ES
    _index_documents(store, doc_id, documents)
    logger.info(f"[Pipeline] 处理完成: {len(documents)} documents")

    return documents


def _save_intermediate(documents: List[Document], dataset_id: str, filename: str, processed_dir: Path) -> Path:
    """保存中间结果到 JSON"""
    output_dir = processed_dir / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).stem
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        safe_name = safe_name.replace(ch, '_')
    out_path = output_dir / f"{safe_name}.json"

    data = [doc.to_dict() for doc in documents]
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"  [Intermediate] 已保存: {out_path}")
    return out_path


def _load_intermediate(dataset_id: str, filename: str, processed_dir: Path) -> Optional[List[Document]]:
    """从 JSON 加载中间结果"""
    output_dir = processed_dir / dataset_id
    safe_name = Path(filename).stem
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        safe_name = safe_name.replace(ch, '_')
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


def _index_documents(store: DocumentStore, doc_id: str, documents: List[Document]):
    """索引文档到 ES"""
    store.ensure_indices()
    success = store.index_document(doc_id, documents)
    logger.info(f"  [ES] 已索引: {success} chunks")


# ── 内部函数 ──────────────────────────────────────────


def _embed_documents(documents: List[Document]):
    """计算每个 Document 及其 children 的向量"""
    for doc in documents:
        # 主块向量
        vector = encode(doc.content)
        if vector is not None:
            doc.vector = vector.tolist()

        # children 向量
        if doc.children:
            for child in doc.children:
                child_vector = encode(child.content)
                if child_vector is not None:
                    child.vector = child_vector.tolist()
