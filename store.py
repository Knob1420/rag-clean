"""
ES 存储 — mapping + 索引逻辑合一

核心变化：
- 新增 doc_type、category 作为 chunk 顶级字段
- 新增 spec_table（结构化表格）
- 新增 parent_id、children_ids 父子层级导航
- 合并 mapping + 索引管理 + 查询逻辑
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from elasticsearch import Elasticsearch
from loguru import logger

from config import settings
from models import Chunk, ProcessedDocument


# ============================================================
# 索引 Mapping
# ============================================================

CHUNKS_MAPPING = {
    "properties": {
        "chunk_id": {"type": "keyword"},
        "doc_id": {"type": "keyword"},
        "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
        # 文档级字段（冗余存储，方便 chunk 级检索）
        "doc_type": {"type": "keyword"},
        "domain": {"type": "keyword"},
        "filter_terms": {"type": "keyword"},
        # chunk 级字段
        "chunk_type": {"type": "keyword"},
        "section_title": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
        "spec_table": {"type": "object", "dynamic": True},
        "spec_rows": {"type": "object", "dynamic": True},
        # 导航关系（parent 存本地文件，ES 只存 child 的 parent_id 用于反查）
        "parent_id": {"type": "keyword"},
        # 检索辅助
        "entities_text": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
        "keywords": {"type": "keyword", "normalizer": "lowercase"},
        "context_summary": {"type": "text", "analyzer": "ik_max_word"},
        # 向量
        "embedding_vector": {
            "type": "dense_vector",
            "dims": settings.embedding_dim,
            "index": True,
            "similarity": "cosine",
        },
        # 基础
        "is_latest": {"type": "boolean"},
        "created_at": {"type": "date"},
    }
}

DOCUMENTS_MAPPING = {
    "properties": {
        "doc_id": {"type": "keyword"},
        "title": {"type": "text", "analyzer": "ik_max_word"},
        "doc_type": {"type": "keyword"},
        "domain": {"type": "keyword"},
        "filter_terms": {"type": "keyword"},
        "global_entities": {"type": "object", "dynamic": True},
        "global_summary": {"type": "text", "analyzer": "ik_max_word"},
        "chunks_count": {"type": "integer"},
        "is_latest": {"type": "boolean"},
        "created_at": {"type": "date"},
    }
}


# ============================================================
# DocumentStore
# ============================================================


class DocumentStore:
    """ES 存储层"""

    def __init__(self, es: Optional[Elasticsearch] = None):
        self._es = es

    @property
    def es(self) -> Elasticsearch:
        if self._es is None:
            self._es = Elasticsearch([settings.es_url])
        return self._es

    def ensure_indices(self):
        """确保索引存在，不存在则创建"""
        for index_name, mapping in [
            (settings.es_index_chunks, CHUNKS_MAPPING),
            (settings.es_index_documents, DOCUMENTS_MAPPING),
        ]:
            if not self.es.indices.exists(index=index_name):
                self.es.indices.create(index=index_name, mappings=mapping)
                logger.info(f"索引已创建: {index_name}")
            else:
                logger.info(f"索引已存在: {index_name}")

    def index_document(self, doc: ProcessedDocument) -> int:
        """
        索引一个完整文档（doc record + all chunks）。

        Returns:
            成功索引的 chunk 数量
        """
        now = datetime.now().isoformat()

        # 1. 索引文档记录
        doc_record = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "doc_type": doc.analysis.doc_type,
            "domain": doc.analysis.domain,
            "filter_terms": doc.analysis.filter_terms,
            "global_entities": doc.analysis.entities,
            "global_summary": doc.analysis.summary,
            "chunks_count": len(doc.chunks),
            "is_latest": True,
            "created_at": now,
        }
        self.es.index(
            index=settings.es_index_documents, id=doc.doc_id, document=doc_record
        )
        logger.info(f"  文档记录已创建: {doc.doc_id}")

        # 2. 索引所有 chunks
        success_count = 0
        for chunk in doc.chunks:
            try:
                chunk_doc = self._chunk_to_es_doc(chunk, doc, now)
                self.es.index(
                    index=settings.es_index_chunks,
                    id=chunk.chunk_id,
                    document=chunk_doc,
                )
                success_count += 1
            except Exception as e:
                logger.warning(f"  索引失败 {chunk.chunk_id}: {e}")

        logger.info(f"  索引完成: {success_count}/{len(doc.chunks)} chunks")

        # 3. 刷新索引
        try:
            self.es.indices.refresh(index=settings.es_index_chunks)
            self.es.indices.refresh(index=settings.es_index_documents)
        except Exception:
            pass

        return success_count

    def _chunk_to_es_doc(
        self, chunk: Chunk, doc: ProcessedDocument, created_at: str
    ) -> Dict[str, Any]:
        """将 Chunk 转为 ES 文档"""
        # 构建 entities_text（用于 BM25 检索）
        entity_values = list(doc.analysis.entities.values())
        # entity 值可能是 list 或 str，统一展平为 str
        flat_values = []
        for v in entity_values:
            if isinstance(v, list):
                flat_values.extend(str(i) for i in v)
            else:
                flat_values.append(str(v))
        # keywords 也可能有嵌套 list（旧 checkpoint 数据），统一展平
        flat_kw = []
        for k in chunk.keywords:
            if isinstance(k, list):
                flat_kw.extend(str(x) for x in k)
            else:
                flat_kw.append(str(k))
        entities_text = " ".join(flat_values + flat_kw)

        # 确定 embedding 字段：外部已计算则跳过
        es_doc: Dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "content": chunk.content,
            # 文档级字段
            "doc_type": doc.analysis.doc_type,
            "domain": doc.analysis.domain,
            "filter_terms": doc.analysis.filter_terms,
            # chunk 级
            "chunk_type": chunk.chunk_type,
            "section_title": chunk.section_title,
            "spec_table": chunk.spec_table,
            "spec_rows": chunk.spec_rows,
            # 导航（parent 存本地，child 通过 parent_id 反查）
            "parent_id": chunk.parent_id,
            # 检索辅助
            "entities_text": entities_text,
            "keywords": flat_kw,
            "context_summary": chunk.context_summary,
            # 基础
            "is_latest": True,
            "created_at": created_at,
        }

        # embedding_vector 如果已有则添加（由 pipeline 设置）
        if hasattr(chunk, "_embedding_vector") and chunk._embedding_vector is not None:
            es_doc["embedding_vector"] = chunk._embedding_vector

        return es_doc

    def get_parent(self, parent_id: str) -> Optional[Dict]:
        """根据 parent_id 从本地文件加载 parent chunk"""
        parent_file = Path(settings.parent_store_dir) / f"{parent_id}.json"
        if parent_file.exists():
            return json.loads(parent_file.read_text())
        return None

    def list_documents(
        self, page: int = 1, page_size: int = 20
    ) -> Dict[str, Any]:
        """分页列出文档记录"""
        from_ = (page - 1) * page_size

        resp = self.es.search(
            index=settings.es_index_documents,
            body={
                "query": {"term": {"is_latest": True}},
                "from": from_,
                "size": page_size,
                "sort": [{"created_at": {"order": "desc"}}],
                "_source": [
                    "doc_id", "title", "doc_type", "domain",
                    "chunks_count", "is_latest", "created_at",
                ],
            },
        )

        total = resp["hits"]["total"]["value"]
        docs = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            docs.append({
                "doc_id": src["doc_id"],
                "title": src.get("title", ""),
                "doc_type": src.get("doc_type", ""),
                "domain": src.get("domain", ""),
                "chunks_count": src.get("chunks_count", 0),
                "status": "completed",
                "created_at": src.get("created_at", ""),
            })

        return {"documents": docs, "total": total, "page": page, "page_size": page_size}

    def get_doc_titles(self, doc_ids: List[str]) -> Dict[str, str]:
        """批量获取 doc_id → title 映射"""
        if not doc_ids:
            return {}
        try:
            resp = self.es.search(
                index=settings.es_index_documents,
                body={
                    "query": {"terms": {"doc_id": list(doc_ids)}},
                    "size": len(doc_ids),
                    "_source": ["doc_id", "title"],
                },
            )
            return {
                hit["_source"]["doc_id"]: hit["_source"].get("title", "")
                for hit in resp.get("hits", {}).get("hits", [])
            }
        except Exception as e:
            logger.warning(f"获取 doc_titles 失败: {e}")
            return {}

    def delete_document(self, doc_id: str):
        """删除文档及其所有 chunks（ES + 本地 parent 文件）"""
        # 1. 删除 ES 中的 chunks
        self.es.delete_by_query(
            index=settings.es_index_chunks,
            query={"term": {"doc_id": doc_id}},
        )
        # 2. 删除 ES 中的文档记录
        self.es.delete(
            index=settings.es_index_documents, id=doc_id, ignore=[404]
        )
        # 3. 删除本地 parent 文件
        parent_dir = Path(settings.parent_store_dir)
        if parent_dir.exists():
            for f in parent_dir.glob(f"{doc_id}_p*.json"):
                f.unlink()
        logger.info(f"已删除文档: {doc_id}")


# ── 全局实例 ──────────────────────────────────────────

_store: Optional[DocumentStore] = None


def get_store() -> DocumentStore:
    global _store
    if _store is None:
        _store = DocumentStore()
    return _store
