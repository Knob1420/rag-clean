"""
WeKnora Faithful Port — grep_chunks Tool

Ported from WeKnora internal/agent/tools/grep_chunks.go

Regex-based keyword search with:
- Regex compilation & validation
- Match snippet extraction
- MMR diversity (optional)
- already_seen tracking (across calls)
- query_hit counting
- aggregateByKnowledge (group results by knowledge/doc)
- XML <grep_results> output format
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from core.retrieve.retrieval import RetrievalService, RetrievalOptions
from core.retrieve.retrieval_models import RetrievedChunk
from store import get_store
from config import settings


# ── MMR helpers ───────────────────────────────────────────────────

def _mmr_diversify(
    chunks: List[RetrievedChunk],
    query_embedding: Optional[List[float]] = None,
    lambda_param: float = 0.7,
    top_k: int = 10,
) -> List[RetrievedChunk]:
    """
    Maximal Marginal Relevance for diversity.
    Without query_embedding, uses content overlap as similarity proxy.
    """
    if not chunks or top_k >= len(chunks):
        return chunks[:top_k]

    selected: List[RetrievedChunk] = [chunks[0]]
    remaining = list(chunks[1:])

    while len(selected) < top_k and remaining:
        best_idx = 0
        best_score = -float("inf")

        for i, candidate in enumerate(remaining):
            # Relevance: use original score
            relevance = candidate.score

            # Diversity: max similarity to already selected
            max_sim = 0.0
            for sel in selected:
                # Simple word overlap similarity as proxy
                sim = _word_overlap(candidate.content, sel.content)
                max_sim = max(max_sim, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def _word_overlap(s1: str, s2: str) -> float:
    """Simple word-overlap Jaccard similarity between two strings."""
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


# ── Snippet extraction ─────────────────────────────────────────────

def _extract_match_snippets(
    content: str,
    pattern: str,
    context_chars: int = 100,
    max_snippets: int = 3,
) -> List[str]:
    """
    Extract short text snippets around regex matches in content.
    Returns up to max_snippets snippets with context_chars of surrounding text.
    """
    if not content or not pattern:
        return []

    try:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
    except re.error:
        return []

    snippets = []
    for match in matches[:max_snippets]:
        start = max(0, match.start() - context_chars)
        end = min(len(content), match.end() + context_chars)
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        # Clean for inline display
        snippet = snippet.replace("\n", " ").strip()
        snippets.append(snippet)

    return snippets


# ── Main grep_chunks handler ───────────────────────────────────────

def grep_chunks_handler(
    args: Dict[str, Any],
    retrieval_service: RetrievalService,
    accumulated_chunks: List[RetrievedChunk],
    already_seen: Set[str],
    current_query: str = "",
) -> str:
    """
    Execute grep_chunks tool: regex-based keyword search.

    Args:
        args: Tool arguments with 'queries' (1-5 regex patterns)
        retrieval_service: RetrievalService instance
        accumulated_chunks: Shared state for accumulated results
        already_seen: Set of already-seen chunk IDs (across calls)
        current_query: Original user query

    Returns:
        XML-formatted <grep_results> string
    """
    queries = args.get("queries", [])
    top_k = args.get("top_k", 10)
    use_mmr = args.get("use_mmr", False)
    aggregate = args.get("aggregateByKnowledge", False)

    if not queries:
        return "<grep_results><error>queries cannot be empty</error></grep_results>"

    # Validate regex patterns
    valid_queries = []
    for q in queries:
        try:
            re.compile(q)
            valid_queries.append(q)
        except re.error as e:
            logger.warning(f"[grep_chunks] Invalid regex '{q}': {e}")

    if not valid_queries:
        return "<grep_results><error>All regex patterns invalid</error></grep_results>"

    # Use BM25-only search (keyword mode, no vector)
    options = RetrievalOptions(
        top_k=top_k,
        use_rerank=False,
        vector_weight=None,
    )

    all_chunks: List[RetrievedChunk] = []
    seen_ids: Set[str] = set()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=min(len(valid_queries), 5)) as pool:
        future_to_query = {
            pool.submit(retrieval_service.search, q, options, use_hybrid=False): q
            for q in valid_queries
        }
        for future in as_completed(future_to_query):
            try:
                result = future.result()
                for chunk in result.chunks:
                    if chunk.chunk_id not in seen_ids:
                        all_chunks.append(chunk)
                        seen_ids.add(chunk.chunk_id)
            except Exception as e:
                logger.warning(f"[grep_chunks] query failed: {e}")

    # Track already_seen
    new_chunks = []
    for chunk in all_chunks:
        if chunk.chunk_id not in already_seen:
            new_chunks.append(chunk)
            already_seen.add(chunk.chunk_id)

    # MMR diversification
    if use_mmr and new_chunks:
        new_chunks = _mmr_diversify(new_chunks, top_k=top_k)

    new_chunks = new_chunks[:top_k]

    # Accumulate chunks
    _accumulate_chunks(accumulated_chunks, new_chunks)

    # Query hit counting
    query_hits = _count_query_hits(new_chunks, valid_queries)

    # Format as XML
    return _format_grep_results(
        new_chunks, valid_queries, query_hits, aggregate
    )


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


def _count_query_hits(
    chunks: List[RetrievedChunk],
    queries: List[str],
) -> Dict[str, int]:
    """Count how many chunks each query pattern hits."""
    hits = {}
    for q in queries:
        count = 0
        try:
            pattern = re.compile(q, re.IGNORECASE)
            for chunk in chunks:
                if pattern.search(chunk.content):
                    count += 1
        except re.error:
            pass
        hits[q] = count
    return hits


def _format_grep_results(
    chunks: List[RetrievedChunk],
    queries: List[str],
    query_hits: Dict[str, int],
    aggregate: bool,
) -> str:
    """Format results as WeKnora-style XML <grep_results>."""
    if not chunks:
        return "<grep_results />\n"

    parts = ["<grep_results>"]

    # Query hit summary
    parts.append("  <query_summary>")
    for q in queries:
        hits = query_hits.get(q, 0)
        parts.append(f'    <query pattern="{_xml_escape(q)}" hits="{hits}" />')
    parts.append("  </query_summary>")

    if aggregate:
        # Group by knowledge_id (doc_id)
        by_doc: Dict[str, List[RetrievedChunk]] = {}
        for chunk in chunks:
            by_doc.setdefault(chunk.doc_id, []).append(chunk)

        parts.append("  <knowledge_groups>")
        for doc_id, doc_chunks in by_doc.items():
            doc_name = doc_chunks[0].doc_title or doc_id
            parts.append(f'    <knowledge_group knowledge_id="{doc_id}">')
            parts.append(f"      <doc_title>{_xml_escape(doc_name)}</doc_title>")
            for chunk in doc_chunks:
                _format_chunk_entry(parts, chunk, queries)
            parts.append("    </knowledge_group>")
        parts.append("  </knowledge_groups>")
    else:
        for chunk in chunks:
            _format_chunk_entry(parts, chunk, queries)

    parts.append("</grep_results>")
    return "\n".join(parts)


def _format_chunk_entry(
    parts: List[str],
    chunk: RetrievedChunk,
    queries: List[str],
) -> None:
    """Format a single chunk entry in XML."""
    doc_name = chunk.doc_title or chunk.doc_id
    score_str = f"{chunk.score:.4f}" if chunk.score else "N/A"
    chunk_type = chunk.chunk_type or "unknown"

    parts.append(
        f'    <chunk knowledge_id="{chunk.doc_id}" '
        f'chunk_id="{chunk.chunk_id}" '
        f'score="{score_str}" '
        f'type="{chunk_type}">'
    )
    parts.append(f"      <doc_title>{_xml_escape(doc_name)}</doc_title>")

    # Match snippets per query
    for q in queries:
        snippets = _extract_match_snippets(chunk.content, q)
        if snippets:
            for snippet in snippets:
                parts.append(
                    f'      <match_snippet query="{_xml_escape(q)}">'
                    f"{_xml_escape(snippet)}"
                    f"</match_snippet>"
                )

    # Full content
    parts.append(f"      <content>{_xml_escape(chunk.content)}</content>")
    parts.append("    </chunk>")


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

GREP_CHUNKS_DEFINITION = {
    "name": "grep_chunks",
    "description": (
        "Keyword/regex search: searches chunk content using regex patterns. "
        "Returns <grep_results> with <match_snippet> per hit for relevance preview. "
        "STRONGLY PREFER one alternation query (e.g., 'stardust|skyvault|psionic') "
        "over multiple single-keyword calls. Literal text also works ('engine' matches anywhere). "
        "After getting results with knowledge_ids/chunk_ids, you MUST call "
        "list_knowledge_chunks to deep-read the full content."
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
                    "Regex patterns (1-5). Use alternation for efficiency: "
                    "'term1|term2|term3'. Literal text also works."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results (default 10)",
                "default": 10,
            },
            "use_mmr": {
                "type": "boolean",
                "description": "Enable MMR diversity filtering (default false)",
                "default": False,
            },
            "aggregateByKnowledge": {
                "type": "boolean",
                "description": "Group results by document/knowledge (default false)",
                "default": False,
            },
        },
        "required": ["queries"],
    },
}
