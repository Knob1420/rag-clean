"""
Parent Chunk 展开逻辑

供 SimplePipeline 和 RAGPipeline 共用的父块展开功能：
- summary chunk → 展开为 parent chunk
- 同一 parent 有 >= 2 个 child → 展开为 parent chunk
- 同一 parent 只有 1 个 child → 保留该 child + 前后各 N 个 sibling
- 无 parent_id 的 chunk → 保持原样
"""

from typing import Dict, List

from loguru import logger

from config import settings
from core.retrieve.retrieval_models import RetrievedChunk


def _make_retrieved_chunk(source: Dict, score: float = 0.0) -> RetrievedChunk:
    """从 ES _source 构建 RetrievedChunk"""
    return RetrievedChunk(
        chunk_id=source.get("chunk_id", ""),
        doc_id=source.get("doc_id", ""),
        content=source.get("content", ""),
        score=score,
        doc_title=source.get("doc_title"),
        dataset_id=source.get("dataset_id"),
        chunk_type=source.get("chunk_type"),
        doc_hash=source.get("doc_hash"),
        section_title=source.get("section_title"),
        parent_id=source.get("parent_id"),
    )


def get_sibling_chunks(
    store,
    parent_id: str,
    current_chunk_id: str,
    limit: int = 1,
) -> List[RetrievedChunk]:
    """获取指定 child chunk 的前后各 N 个 sibling"""
    try:
        resp = store.es.search(
            index=settings.es_index_chunks,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"parent_id": parent_id}},
                            {"term": {"chunk_type": "child"}},
                            {"term": {"is_latest": True}},
                        ],
                        "must_not": [{"term": {"chunk_id": current_chunk_id}}],
                    }
                },
                "sort": [{"created_at": "asc"}],
                "size": 100,
            },
        )
    except Exception as e:
        logger.warning(f"获取 sibling chunks 失败: {e}")
        return []

    hits = resp.get("hits", {}).get("hits", [])
    if not hits:
        return []

    current_idx = -1
    for i, hit in enumerate(hits):
        if hit.get("_source", {}).get("chunk_id") == current_chunk_id:
            current_idx = i
            break

    if current_idx < 0:
        return []

    siblings: List[RetrievedChunk] = []
    for i in range(current_idx - limit, current_idx):
        if i >= 0:
            src = hits[i].get("_source", {})
            siblings.append(_make_retrieved_chunk(src, hits[i].get("_score", 0.0)))

    for i in range(current_idx + 1, current_idx + 1 + limit):
        if i < len(hits):
            src = hits[i].get("_source", {})
            siblings.append(_make_retrieved_chunk(src, hits[i].get("_score", 0.0)))

    return siblings


def expand_to_parent_chunks(
    chunks: List[RetrievedChunk],
    store,
) -> List[RetrievedChunk]:
    """将 child/summary chunks 做智能展开。"""
    if not chunks:
        return chunks

    # 1. 按 parent_id 分组
    parent_children: Dict[str, List[RetrievedChunk]] = {}
    for chunk in chunks:
        if chunk.parent_id:
            parent_children.setdefault(chunk.parent_id, []).append(chunk)

    if not parent_children:
        return chunks

    # 2. 收集需要展开的 parent_id
    parent_ids_to_expand = []
    single_child_ids = []
    for pid, children in parent_children.items():
        if any(c.chunk_type == "summary" for c in children):
            parent_ids_to_expand.append(pid)
        elif len(children) >= 2:
            parent_ids_to_expand.append(pid)
        else:
            single_child_ids.append(pid)

    # 3. 批量拉取需要展开的 parent chunks
    parent_map: Dict[str, Dict] = {}
    if parent_ids_to_expand:
        try:
            resp = store.es.mget(
                index=settings.es_index_chunks,
                body={"ids": parent_ids_to_expand},
            )
            for doc in resp.get("docs", []):
                if doc.get("found") and doc.get("_source"):
                    parent_map[doc["_id"]] = doc["_source"]
        except Exception as e:
            logger.warning(f"批量获取 parent chunk 失败: {e}")

    # 4. 构建结果
    result: List[RetrievedChunk] = []
    seen_ids = set()

    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            continue

        if not chunk.parent_id:
            result.append(chunk)
            seen_ids.add(chunk.chunk_id)
            continue

        children = parent_children[chunk.parent_id]

        # 4a. summary → 展开为 parent
        if chunk.chunk_type == "summary":
            if chunk.parent_id in parent_map:
                parent_src = parent_map[chunk.parent_id]
                parent_chunk = RetrievedChunk(
                    chunk_id=parent_src.get("chunk_id", chunk.parent_id),
                    doc_id=parent_src.get("doc_id", ""),
                    content=parent_src.get("content", ""),
                    score=chunk.score,
                    doc_title=parent_src.get("doc_title"),
                    dataset_id=parent_src.get("dataset_id"),
                    chunk_type="parent",
                    doc_hash=parent_src.get("doc_hash"),
                    section_title=parent_src.get("section_title"),
                    parent_id=None,
                )
                result.append(parent_chunk)
                seen_ids.add(chunk.parent_id)
            else:
                result.append(chunk)
                seen_ids.add(chunk.chunk_id)
            continue

        # 4b. >= 2 个 child from same parent → 展开为 parent
        if len(children) >= 2:
            if chunk.parent_id in parent_map:
                parent_src = parent_map[chunk.parent_id]
                max_score = max(c.score for c in children)
                parent_chunk = RetrievedChunk(
                    chunk_id=parent_src.get("chunk_id", chunk.parent_id),
                    doc_id=parent_src.get("doc_id", ""),
                    content=parent_src.get("content", ""),
                    score=max_score,
                    doc_title=parent_src.get("doc_title"),
                    dataset_id=parent_src.get("dataset_id"),
                    chunk_type="parent",
                    doc_hash=parent_src.get("doc_hash"),
                    section_title=parent_src.get("section_title"),
                    parent_id=None,
                )
                if chunk.parent_id not in seen_ids:
                    result.append(parent_chunk)
                    seen_ids.add(chunk.parent_id)
            continue

        # 4c. 只有 1 个 child → 保留 child + 前后各 1 个 sibling
        result.append(chunk)
        seen_ids.add(chunk.chunk_id)

        siblings = get_sibling_chunks(store, chunk.parent_id, chunk.chunk_id)
        for sib in siblings:
            if sib.chunk_id not in seen_ids:
                result.append(sib)
                seen_ids.add(sib.chunk_id)

    return result
