"""
ES 存储 — mapping + 索引逻辑合一

核心变化：
- 只用一个 chunks 索引（parent + child 都存在这里）
- parent_id 父子层级导航（Dify 方式：parent 存在 ES 中）
- 通过 doc_id 聚合获取文档列表
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from elasticsearch import Elasticsearch
from loguru import logger

from config import settings
from core.model.models import Document, ChildDocument, SummaryDocument


# ============================================================
# 索引 Mapping
# ============================================================

CHUNKS_MAPPING = {
    "properties": {
        # 核心 ID
        "chunk_id": {"type": "keyword"},       # chunk 唯一ID (parent_id 或 child_id)
        "doc_id": {"type": "keyword"},         # 文档ID（一次上传 = 一个 doc_id）
        "doc_hash": {"type": "keyword"},       # chunk 内容哈希（去重/标识用）
        # 文档级信息
        "doc_title": {"type": "text", "analyzer": "ik_max_word"},
        "dataset_id": {"type": "keyword"},     # 知识库ID（后续扩展）
        # chunk 级
        "chunk_type": {"type": "keyword"},     # "parent" | "child" | "summary"
        "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
        # summary 相关（parent chunk 有）
        "summary": {"type": "text", "analyzer": "ik_max_word"},
        "primary_entity": {"type": "keyword"}, # 核心实体（parent 和 child 都有）
        # 父子关系
        "parent_id": {"type": "keyword"},       # child 有，parent 无
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
        if not self.es.indices.exists(index=settings.es_index_chunks):
            self.es.indices.create(index=settings.es_index_chunks, mappings=CHUNKS_MAPPING)
            logger.info(f"索引已创建: {settings.es_index_chunks}")
        else:
            logger.info(f"索引已存在: {settings.es_index_chunks}")

    def index_document(self, doc_id: str, documents: List[Document]) -> int:
        """
        索引一个完整文档（所有 chunks）。

        Args:
            doc_id: 文档 ID
            documents: Document 列表（每个代表一个 parent chunk 及 其 children）

        Returns:
            成功索引的 chunk 数量
        """
        now = datetime.now().isoformat()

        # 计算总 chunks 数
        total_chunks = 0
        for doc in documents:
            total_chunks += 1  # parent chunk
            if doc.children:
                total_chunks += len(doc.children)
            if doc.summaries:
                total_chunks += len(doc.summaries)

        # 索引所有 chunks（parent + children + summaries）
        success_count = 0
        child_idx = 0
        summary_idx = 0
        for doc in documents:
            parent_id = doc.metadata.get("chunk_id", "")

            # 索引 parent chunk
            try:
                parent_doc = self._document_to_es_doc(doc, parent_id, "parent", now)
                self.es.index(
                    index=settings.es_index_chunks,
                    id=parent_id,
                    document=parent_doc,
                )
                success_count += 1
            except Exception as e:
                logger.warning(f"  索引失败 parent {parent_id}: {e}")

            # 索引 summaries（在 children 之前索引，便于关联）
            if doc.summaries:
                for summary in doc.summaries:
                    summary_id = summary.metadata.get("chunk_id", f"{doc_id}_s{summary_idx}")
                    try:
                        summary_doc = self._summary_to_es_doc(summary, summary_id, parent_id, now)
                        self.es.index(
                            index=settings.es_index_chunks,
                            id=summary_id,
                            document=summary_doc,
                        )
                        success_count += 1
                    except Exception as e:
                        logger.warning(f"  索引失败 summary {summary_id}: {e}")
                    summary_idx += 1

            # 索引 children
            if doc.children:
                for child in doc.children:
                    child_id = child.metadata.get("chunk_id", f"{doc_id}_c{child_idx}")
                    try:
                        child_doc = self._child_to_es_doc(child, child_id, parent_id, now)
                        self.es.index(
                            index=settings.es_index_chunks,
                            id=child_id,
                            document=child_doc,
                        )
                        success_count += 1
                    except Exception as e:
                        logger.warning(f"  索引失败 child {child_id}: {e}")
                    child_idx += 1

        logger.info(f"  索引完成: {success_count}/{total_chunks} chunks")

        # 刷新索引
        try:
            self.es.indices.refresh(index=settings.es_index_chunks)
        except Exception:
            pass

        return success_count

    def _document_to_es_doc(
        self, doc: Document, chunk_id: str, chunk_type: str, created_at: str
    ) -> Dict[str, Any]:
        """将 Document（parent chunk）转为 ES 文档"""
        es_doc: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "doc_id": doc.metadata.get("doc_id", ""),
            "doc_title": doc.metadata.get("doc_title", ""),
            "doc_hash": doc.metadata.get("doc_hash", ""),
            "dataset_id": doc.metadata.get("dataset_id", ""),
            "content": doc.content,
            # chunk 级
            "chunk_type": chunk_type,
            # summary 相关（parent chunk 有）
            "summary": getattr(doc, "summary", ""),
            "primary_entity": getattr(doc, "primary_entity", ""),
            # 父子关系
            "parent_id": None,  # parent 没有父块
            # 基础
            "is_latest": True,
            "created_at": created_at,
        }

        # embedding_vector 如果已有则添加
        if hasattr(doc, "vector") and doc.vector is not None:
            es_doc["embedding_vector"] = doc.vector

        return es_doc

    def _child_to_es_doc(
        self, child: ChildDocument, chunk_id: str, parent_id: str, created_at: str
    ) -> Dict[str, Any]:
        """将 ChildDocument 转为 ES 文档"""
        es_doc: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "doc_id": child.metadata.get("doc_id", ""),
            "doc_title": child.metadata.get("doc_title", ""),
            "doc_hash": child.metadata.get("doc_hash", ""),
            "dataset_id": child.metadata.get("dataset_id", ""),
            "content": child.content,
            # chunk 级
            "chunk_type": "child",
            # primary_entity
            "primary_entity": getattr(child, "primary_entity", ""),
            # 父子关系
            "parent_id": parent_id,
            # 基础
            "is_latest": True,
            "created_at": created_at,
        }

        # embedding_vector 如果已有则添加
        if hasattr(child, "vector") and child.vector is not None:
            es_doc["embedding_vector"] = child.vector

        return es_doc

    def _summary_to_es_doc(
        self, summary, chunk_id: str, parent_id: str, created_at: str
    ) -> Dict[str, Any]:
        """将 SummaryDocument 转为 ES 文档"""
        es_doc: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "doc_id": summary.metadata.get("doc_id", ""),
            "doc_title": summary.metadata.get("doc_title", ""),
            "doc_hash": summary.metadata.get("doc_hash", ""),
            "dataset_id": summary.metadata.get("dataset_id", ""),
            "content": summary.content,
            "primary_entity": summary.primary_entity,
            # chunk 级
            "chunk_type": "summary",
            # 父子关系
            "parent_id": parent_id,
            # 基础
            "is_latest": True,
            "created_at": created_at,
        }

        # embedding_vector 如果已有则添加
        if hasattr(summary, "vector") and summary.vector is not None:
            es_doc["embedding_vector"] = summary.vector

        return es_doc

    def get_parent(self, parent_id: str) -> Optional[Dict]:
        """
        根据 parent_id 从 ES 加载 parent Document（其 content 即 parent 内容）。

        Dify 方式：parent 存在 ES 中，parent chunk 的 ID = parent_id。
        """
        try:
            resp = self.es.get(index=settings.es_index_chunks, id=parent_id)
            return resp.get("_source")
        except Exception as e:
            logger.warning(f"获取 parent 失败: {e}")
        return None

    def list_documents(
        self, page: int = 1, page_size: int = 20
    ) -> Dict[str, Any]:
        """分页列出文档（从 chunks 索引聚合 doc_id）"""
        from_ = (page - 1) * page_size

        # 用 aggregation 获取去重后的 doc_id 列表
        resp = self.es.search(
            index=settings.es_index_chunks,
            body={
                "query": {"term": {"chunk_type": "parent"}},  # 只查 parent
                "from": from_,
                "size": 0,  # 不返回 hits，只做聚合
                "aggs": {
                    "docs": {
                        "terms": {
                            "field": "doc_id",
                            "size": page_size,
                            "order": {"max_created": "desc"},
                        },
                        "aggs": {
                            "max_created": {"max": {"field": "created_at"}},
                            "titles": {"top_hits": {"size": 1, "_source": ["title"]}},
                        },
                    },
                    "total": {"value_count": {"field": "doc_id"}},
                },
            },
        )

        aggs = resp.get("aggregations", {})
        buckets = aggs.get("docs", {}).get("buckets", [])
        total = aggs.get("total", {}).get("value", 0)

        docs = []
        for bucket in buckets:
            title = bucket.get("titles", {}).get("hits", {}).get("hits", [{}])[0].get("_source", {}).get("doc_title", "")
            docs.append({
                "doc_id": bucket["key"],
                "title": title,
                "chunks_count": bucket.get("doc_count", 0),
                "created_at": bucket.get("max_created", {}).get("value_as_string", ""),
            })

        return {"documents": docs, "total": total, "page": page, "page_size": page_size}

    def get_doc_titles(self, doc_ids: List[str]) -> Dict[str, str]:
        """批量获取 doc_id → title 映射"""
        if not doc_ids:
            return {}
        try:
            resp = self.es.search(
                index=settings.es_index_chunks,
                body={
                    "query": {"terms": {"doc_id": list(doc_ids)}},
                    "size": len(doc_ids),
                    "_source": ["doc_id", "doc_title"],
                },
            )
            # 优先取 parent chunk 的 title，其次取任何 chunk 的 title
            result: Dict[str, str] = {}
            for hit in resp.get("hits", {}).get("hits", []):
                doc_id = hit["_source"]["doc_id"]
                title = hit["_source"].get("doc_title", "")
                if title and doc_id not in result:
                    result[doc_id] = title
                elif title and doc_id in result and not result[doc_id]:
                    result[doc_id] = title
            return result
        except Exception as e:
            logger.warning(f"获取 doc_titles 失败: {e}")
            return {}

    def delete_document(self, doc_id: str):
        """删除文档及其所有 chunks（ES）"""
        self.es.delete_by_query(
            index=settings.es_index_chunks,
            query={"term": {"doc_id": doc_id}},
        )
        logger.info(f"已删除文档: {doc_id}")


# ── 全局实例 ──────────────────────────────────────────

_store: Optional[DocumentStore] = None


def get_store() -> DocumentStore:
    global _store
    if _store is None:
        _store = DocumentStore()
    return _store
