"""
WeKnora Faithful Port — knowledge_search Tool

Ported from WeKnora internal/agent/tools/knowledge_search.go

Hybrid vector + BM25 search returning structured XML <knowledge_results>.
For single-query calls, automatic reranking with score threshold is applied.
"""

import re
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from core.retrieve.retrieval import RetrievalService, RetrievalOptions
from core.retrieve.retrieval_models import RetrievedChunk
from core.client.rerank_client import rerank_documents
from config import settings


def knowledge_search_handler(
    args: Dict[str, Any],
    retrieval_service: RetrievalService,
    accumulated_chunks: List[RetrievedChunk],
    current_query: str = "",
) -> str:
    """
    Execute knowledge_search tool: hybrid vector+BM25 search.

    Args:
        args: Tool arguments with 'queries' (1-5) and optional 'top_k', 'knowledge_base_ids'
        retrieval_service: RetrievalService instance
        accumulated_chunks: List to append new chunks to (shared state)
        current_query: The original user query (for reranking)

    Returns:
        XML-formatted <knowledge_results> string
    """
    queries = args.get("queries", [])
    top_k = args.get("top_k", 10)
    kb_ids = args.get("knowledge_base_ids")

    if not queries:
        return "<knowledge_results><error>queries cannot be empty</error></knowledge_results>"

    options = RetrievalOptions(
        top_k=top_k,
        use_rerank=False,
    )

    all_chunks: List[RetrievedChunk] = []
    seen_ids: Set[str] = set()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as pool:
        future_to_query = {
            pool.submit(retrieval_service.search, q, options, True): q
            for q in queries
        }
        for future in as_completed(future_to_query):
            try:
                result = future.result()
                for chunk in result.chunks:
                    if chunk.chunk_id not in seen_ids:
                        all_chunks.append(chunk)
                        seen_ids.add(chunk.chunk_id)
            except Exception as e:
                logger.warning(f"[knowledge_search] query failed: {e}")

    # Single-query automatic reranking (WeKnora pattern)
    if len(queries) == 1 and all_chunks:
        _apply_auto_rerank(all_chunks, queries[0], top_k)

    # Accumulate chunks (dedup)
    _accumulate_chunks(accumulated_chunks, all_chunks)

    # Format as XML
    return _format_knowledge_results(all_chunks)


def _apply_auto_rerank(
    chunks: List[RetrievedChunk],
    query: str,
    top_k: int,
) -> None:
    """Apply automatic reranking for single-query searches (WeKnora pattern)."""
    try:
        documents = [c.content for c in chunks]
        rerank_results = rerank_documents(
            query=query,
            documents=documents,
            top_k=top_k,
        )
        reranked_map = {doc: score for doc, score in rerank_results}
        for chunk in chunks:
            if chunk.content in reranked_map:
                chunk.score = reranked_map[chunk.content]

        chunks.sort(key=lambda c: c.score, reverse=True)

        # Score threshold filtering
        threshold = settings.rerank_score_threshold
        if threshold > 0:
            before_count = len(chunks)
            chunks[:] = [c for c in chunks if c.score >= threshold]
            logger.info(
                f"[knowledge_search] score threshold filter: "
                f"{before_count} → {len(chunks)} (threshold={threshold})"
            )

        # Trim to top_k
        del chunks[top_k:]

    except Exception as e:
        logger.warning(f"[knowledge_search] auto-rerank failed: {e}")


def _accumulate_chunks(
    accumulated: List[RetrievedChunk],
    new_chunks: List[RetrievedChunk],
) -> None:
    """Append new chunks to accumulated list (dedup)."""
    seen = {c.chunk_id for c in accumulated}
    for chunk in new_chunks:
        if chunk.chunk_id not in seen:
            accumulated.append(chunk)
            seen.add(chunk.chunk_id)


def _format_knowledge_results(chunks: List[RetrievedChunk]) -> str:
    """Format search results as WeKnora-style XML <knowledge_results>."""
    if not chunks:
        return "<knowledge_results />\n"

    parts = ["<knowledge_results>"]
    for chunk in chunks:
        doc_name = chunk.doc_title or chunk.doc_id
        score_str = f"{chunk.score:.4f}" if chunk.score else "N/A"
        chunk_type = chunk.chunk_type or "unknown"

        # Extract match snippet (first 200 chars around query terms)
        snippet = _extract_match_snippet(chunk.content)

        parts.append(
            f'  <chunk knowledge_id="{chunk.doc_id}" '
            f'chunk_id="{chunk.chunk_id}" '
            f'score="{score_str}" '
            f'type="{chunk_type}">'
        )
        parts.append(f"    <doc_title>{_xml_escape(doc_name)}</doc_title>")
        if snippet:
            parts.append(f"    <match_snippet>{_xml_escape(snippet)}</match_snippet>")
        parts.append(f"    <content>{_xml_escape(chunk.content)}</content>")
        parts.append("  </chunk>")

    parts.append("</knowledge_results>")
    return "\n".join(parts)


def _extract_match_snippet(content: str, max_len: int = 200) -> str:
    """Extract a short snippet from content for relevance preview."""
    if not content:
        return ""
    # Take first max_len characters as snippet
    snippet = content[:max_len]
    if len(content) > max_len:
        snippet += "..."
    # Clean up newlines for inline display
    snippet = snippet.replace("\n", " ").strip()
    return snippet


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

KNOWLEDGE_SEARCH_DEFINITION = {
    "name": "knowledge_search",
    "description": (
        "Semantic search: vector + BM25 hybrid retrieval across the Knowledge Base. "
        "Returns <knowledge_results> with <chunk> entries including match_snippet, "
        "score, and full content. "
        "Use for finding information by meaning/concept, not exact keyword matching. "
        "queries should be complete natural language sentences (1-5). "
        "After getting results with knowledge_ids, you MUST call list_knowledge_chunks "
        "to deep-read the full content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
                "description": (
                    "Natural language query sentences (1-5). "
                    "e.g., ['How does the satellite power system work during eclipse?']"
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results per query (default 10)",
                "default": 10,
            },
            "knowledge_base_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: restrict search to specific KB IDs",
            },
        },
        "required": ["queries"],
    },
}
