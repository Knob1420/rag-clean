"""
WeKnora Faithful Port — list_knowledge_chunks Tool (Deep Read)

Ported from WeKnora internal/agent/tools/list_knowledge_chunks.go

THE CRITICAL MISSING PIECE from the rag-clean agent.
After grep_chunks or knowledge_search returns chunk_ids / knowledge_ids,
the agent MUST call this tool to fetch full content.

This is the "Deep Read" step — reading actual document content
instead of relying on search snippets.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from core.retrieve.retrieval_models import RetrievedChunk
from store import get_store
from config import settings


def list_knowledge_chunks_handler(
    args: Dict[str, Any],
    accumulated_chunks: List[RetrievedChunk],
) -> str:
    """
    Execute list_knowledge_chunks tool: fetch full chunk content by IDs.

    This is the "Deep Read" tool — the agent MUST call this after search
    to read the actual content of matched chunks, not just the snippets.

    Args:
        args: Tool arguments with 'knowledge_ids' list
        accumulated_chunks: Shared state for accumulated results

    Returns:
        XML-formatted <knowledge_chunks> with full content per chunk
    """
    knowledge_ids = args.get("knowledge_ids", [])
    chunk_ids = args.get("chunk_ids", [])

    if not knowledge_ids and not chunk_ids:
        return (
            "<knowledge_chunks><error>"
            "knowledge_ids or chunk_ids must be provided"
            "</error></knowledge_chunks>"
        )

    store = get_store()
    chunks_data: List[Dict[str, Any]] = []

    # Fetch by knowledge_ids (doc-level: get all chunks for a document)
    if knowledge_ids:
        for kid in knowledge_ids:
            try:
                chunks = _fetch_chunks_by_knowledge_id(store, kid)
                chunks_data.extend(chunks)
            except Exception as e:
                logger.warning(
                    f"[list_knowledge_chunks] Failed to fetch knowledge_id={kid}: {e}"
                )

    # Fetch by chunk_ids (specific chunks)
    if chunk_ids:
        try:
            resp = store.es.mget(
                index=settings.es_index_chunks,
                body={"ids": chunk_ids},
            )
            for doc in resp.get("docs", []):
                if doc.get("found") and doc.get("_source"):
                    src = doc["_source"]
                    chunk = _source_to_dict(doc["_id"], src)
                    # Avoid duplicates
                    if not any(c["chunk_id"] == chunk["chunk_id"] for c in chunks_data):
                        chunks_data.append(chunk)
        except Exception as e:
            logger.warning(f"[list_knowledge_chunks] mget by chunk_ids failed: {e}")

    # Accumulate into shared state
    _accumulate_from_dicts(accumulated_chunks, chunks_data)

    # Format as XML
    return _format_knowledge_chunks(chunks_data)


def _fetch_chunks_by_knowledge_id(
    store,
    knowledge_id: str,
    max_chunks: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch all chunks for a given knowledge_id (document)."""
    try:
        resp = store.es.search(
            index=settings.es_index_chunks,
            body={
                "query": {
                    "term": {"doc_id": knowledge_id},
                },
                "size": max_chunks,
                "sort": [{"chunk_id": {"order": "asc"}}],
            },
        )
        chunks = []
        for hit in resp.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            chunk = _source_to_dict(hit["_id"], src)
            chunks.append(chunk)
        return chunks
    except Exception as e:
        logger.warning(f"[list_knowledge_chunks] ES search for knowledge_id={knowledge_id} failed: {e}")
        return []


def _source_to_dict(chunk_id: str, src: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ES _source to a chunk dict."""
    return {
        "chunk_id": src.get("chunk_id", chunk_id),
        "knowledge_id": src.get("doc_id", ""),
        "doc_id": src.get("doc_id", ""),
        "doc_title": src.get("doc_title", ""),
        "chunk_type": src.get("chunk_type", ""),
        "content": src.get("content", ""),
        "parent_id": src.get("parent_id"),
        "dataset_id": src.get("dataset_id", ""),
    }


def _accumulate_from_dicts(
    accumulated: List[RetrievedChunk],
    chunks_data: List[Dict[str, Any]],
) -> None:
    """Convert dicts to RetrievedChunk and accumulate (dedup)."""
    seen = {c.chunk_id for c in accumulated}
    for cd in chunks_data:
        if cd["chunk_id"] not in seen:
            chunk = RetrievedChunk(
                chunk_id=cd["chunk_id"],
                doc_id=cd["doc_id"],
                content=cd["content"],
                score=0.0,
                doc_title=cd.get("doc_title"),
                dataset_id=cd.get("dataset_id"),
                chunk_type=cd.get("chunk_type"),
                parent_id=cd.get("parent_id"),
            )
            accumulated.append(chunk)
            seen.add(cd["chunk_id"])


def _format_knowledge_chunks(chunks_data: List[Dict[str, Any]]) -> str:
    """Format chunks as WeKnora-style XML <knowledge_chunks>."""
    if not chunks_data:
        return "<knowledge_chunks />\n"

    parts = ["<knowledge_chunks>"]
    for cd in chunks_data:
        doc_name = cd.get("doc_title") or cd.get("knowledge_id", "")
        chunk_type = cd.get("chunk_type", "unknown")
        parent_id = cd.get("parent_id", "")

        parts.append(
            f'  <chunk knowledge_id="{_xml_escape(cd["knowledge_id"])}" '
            f'chunk_id="{_xml_escape(cd["chunk_id"])}" '
            f'type="{_xml_escape(chunk_type)}"'
        )
        if parent_id:
            parts.append(f' parent_id="{_xml_escape(parent_id)}"')
        parts.append(">")
        parts.append(f"    <doc_title>{_xml_escape(doc_name)}</doc_title>")
        parts.append(f"    <content>{_xml_escape(cd['content'])}</content>")
        parts.append("  </chunk>")

    parts.append("</knowledge_chunks>")
    return "\n".join(parts)


def _xml_escape(s: str) -> str:
    """Escape special XML characters."""
    if not s:
        return ""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ── Tool Definition for Registry ──────────────────────────────────

LIST_KNOWLEDGE_CHUNKS_DEFINITION = {
    "name": "list_knowledge_chunks",
    "description": (
        "Deep Read: fetch full chunk content by knowledge_ids or chunk_ids. "
        "MANDATORY after grep_chunks or knowledge_search returns IDs — "
        "you MUST call this to read the actual content, not rely on snippets. "
        "Call frequently for multiple IDs. Do not be lazy; fetch the content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "knowledge_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Document/knowledge IDs to fetch all chunks for. "
                    "Get these from the knowledge_id field of search results."
                ),
            },
            "chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Specific chunk IDs to fetch. "
                    "Get these from the chunk_id field of search results."
                ),
            },
        },
        "required": [],
    },
}
