"""
WeKnora Faithful Port — Tools Package

Registers all WeKnora-port tools with the ToolRegistry.
"""

from typing import Any, Callable, Dict, List, Optional

from core.agent.weknora_port.tools.registry import ToolRegistry, ToolDefinition
from core.agent.weknora_port.tools.knowledge_search import (
    KNOWLEDGE_SEARCH_DEFINITION,
    knowledge_search_handler,
)
from core.agent.weknora_port.tools.grep_chunks import (
    GREP_CHUNKS_DEFINITION,
    grep_chunks_handler,
)
from core.agent.weknora_port.tools.list_knowledge_chunks import (
    LIST_KNOWLEDGE_CHUNKS_DEFINITION,
    list_knowledge_chunks_handler,
)
from core.agent.weknora_port.tools.final_answer import (
    FINAL_ANSWER_DEFINITION,
    final_answer_handler,
)
from core.agent.weknora_port.tools.capabilities import (
    KBCapability,
    ToolRequirement,
    RequirementType,
)


def create_weknora_tool_registry(
    retrieval_service,
    accumulated_chunks: List,
    already_seen: set,
    current_query: str = "",
) -> ToolRegistry:
    """
    Create a ToolRegistry with all WeKnora-port tools registered.

    Tool closures capture the shared state (accumulated_chunks, already_seen, etc.).

    Args:
        retrieval_service: RetrievalService instance
        accumulated_chunks: Shared list for accumulated RetrievedChunk results
        already_seen: Shared set of chunk_ids already returned by grep_chunks
        current_query: Original user query

    Returns:
        Configured ToolRegistry
    """
    from core.agent.weknora_port.const import DEFAULT_MAX_TOOL_OUTPUT

    registry = ToolRegistry(max_tool_output=DEFAULT_MAX_TOOL_OUTPUT)

    # ── knowledge_search ─────────────────────────────────────────────
    def _ks_handler(args: Dict[str, Any]) -> str:
        return knowledge_search_handler(
            args,
            retrieval_service=retrieval_service,
            accumulated_chunks=accumulated_chunks,
            current_query=current_query,
        )

    registry.register_tool(ToolDefinition(
        name=KNOWLEDGE_SEARCH_DEFINITION["name"],
        description=KNOWLEDGE_SEARCH_DEFINITION["description"],
        parameters=KNOWLEDGE_SEARCH_DEFINITION["parameters"],
        handler=_ks_handler,
        requirements=[ToolRequirement(RequirementType.ANY_OF, {KBCapability.VECTOR, KBCapability.KEYWORD})],
    ))

    # ── grep_chunks ──────────────────────────────────────────────────
    def _gc_handler(args: Dict[str, Any]) -> str:
        return grep_chunks_handler(
            args,
            retrieval_service=retrieval_service,
            accumulated_chunks=accumulated_chunks,
            already_seen=already_seen,
            current_query=current_query,
        )

    registry.register_tool(ToolDefinition(
        name=GREP_CHUNKS_DEFINITION["name"],
        description=GREP_CHUNKS_DEFINITION["description"],
        parameters=GREP_CHUNKS_DEFINITION["parameters"],
        handler=_gc_handler,
        requirements=[ToolRequirement(RequirementType.ANY_OF, {KBCapability.KEYWORD})],
    ))

    # ── list_knowledge_chunks (Deep Read) ────────────────────────────
    def _lkc_handler(args: Dict[str, Any]) -> str:
        return list_knowledge_chunks_handler(
            args,
            accumulated_chunks=accumulated_chunks,
        )

    registry.register_tool(ToolDefinition(
        name=LIST_KNOWLEDGE_CHUNKS_DEFINITION["name"],
        description=LIST_KNOWLEDGE_CHUNKS_DEFINITION["description"],
        parameters=LIST_KNOWLEDGE_CHUNKS_DEFINITION["parameters"],
        handler=_lkc_handler,
    ))

    # ── final_answer ─────────────────────────────────────────────────
    registry.register_tool(ToolDefinition(
        name=FINAL_ANSWER_DEFINITION["name"],
        description=FINAL_ANSWER_DEFINITION["description"],
        parameters=FINAL_ANSWER_DEFINITION["parameters"],
        handler=final_answer_handler,
    ))

    # ── spec_query (reuse from rag-clean) ────────────────────────────
    _register_spec_query_tool(registry)

    # ── rerank_chunks ────────────────────────────────────────────────
    _register_rerank_tool(registry, accumulated_chunks, current_query)

    # ── resolve_entities (reuse from rag-clean) ──────────────────────
    _register_resolve_entities_tool(registry)

    return registry


def _register_spec_query_tool(registry: ToolRegistry) -> None:
    """Register spec_query tool (reuses rag-clean's spec_matcher)."""
    from core.products.spec_matcher import query_products, format_spec_context

    def _sq_handler(args: Dict[str, Any]) -> str:
        entities = args.get("entities", [])
        fields = args.get("fields", [])
        constraints = args.get("constraints", {})
        if not entities:
            return "Error: entities cannot be empty"
        results = query_products(
            target_models=entities,
            required_fields=fields,
            numerical_constraints=constraints,
        )
        if not results:
            return f"No matching product specs found for '{', '.join(entities)}'."
        return format_spec_context(results, "recommend")

    registry.register_tool(ToolDefinition(
        name="spec_query",
        description=(
            "Structured parameter query: query exact numerical parameters "
            "(power consumption, weight, compute, architecture, interfaces, etc.). "
            "entities specifies target products/series/categories, "
            "fields specifies parameter fields of interest, "
            "constraints specifies numerical constraints."
        ),
        parameters={
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target product/series/category list",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Parameter fields to query",
                },
                "constraints": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Numerical constraints, e.g. {'weight': '<=3.0', 'compute': '>100'}",
                },
            },
            "required": ["entities"],
        },
        handler=_sq_handler,
    ))


def _register_rerank_tool(
    registry: ToolRegistry,
    accumulated_chunks: List,
    current_query: str,
) -> None:
    """Register rerank_chunks tool."""
    from core.client.rerank_client import rerank_documents
    from config import settings

    def _rr_handler(args: Dict[str, Any]) -> str:
        top_k = args.get("top_k", 10)
        if not accumulated_chunks:
            return "No accumulated retrieval results to rerank. Please search first."

        # Dedup
        seen_ids = set()
        unique_chunks = []
        for chunk in accumulated_chunks:
            if chunk.chunk_id not in seen_ids:
                unique_chunks.append(chunk)
                seen_ids.add(chunk.chunk_id)

        if not current_query:
            accumulated_chunks[:] = unique_chunks[:top_k]
            return _format_rerank_result(accumulated_chunks)

        documents = [chunk.content for chunk in unique_chunks]
        rerank_results = rerank_documents(
            query=current_query,
            documents=documents,
            top_k=top_k,
        )

        reranked_map = {doc: score for doc, score in rerank_results}
        for chunk in unique_chunks:
            if chunk.content in reranked_map:
                chunk.score = reranked_map[chunk.content]

        unique_chunks.sort(key=lambda c: c.score, reverse=True)

        # Score threshold
        threshold = settings.rerank_score_threshold
        if threshold > 0:
            unique_chunks = [c for c in unique_chunks if c.score >= threshold]

        accumulated_chunks[:] = unique_chunks[:top_k]
        return _format_rerank_result(accumulated_chunks)

    registry.register_tool(ToolDefinition(
        name="rerank_chunks",
        description=(
            "Rerank: reorder all accumulated retrieval results. "
            "Use when you have accumulated many results from multiple searches "
            "and need more precise ranking. No parameters needed — "
            "automatically reranks all accumulated chunks."
        ),
        parameters={
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to keep after reranking (default 10)",
                    "default": 10,
                },
            },
            "required": [],
        },
        handler=_rr_handler,
    ))


def _register_resolve_entities_tool(registry: ToolRegistry) -> None:
    """Register resolve_entities tool (reuses rag-clean's spec_matcher)."""
    from core.products.spec_matcher import (
        _normalize_category_name,
        _normalize_series_name,
        _normalize_model_name,
        _resolve_category,
        get_product_specs,
    )

    def _re_handler(args: Dict[str, Any]) -> str:
        entities = args.get("entities", [])
        if not entities:
            return "Error: entities cannot be empty"

        specs = get_product_specs()
        if not specs:
            return "Product specs table not loaded"

        results = []
        for raw_name in entities:
            entry = {"input": raw_name}

            resolved_cats = _resolve_category(raw_name)
            actual_cats = [c for c in resolved_cats if c in specs]

            if len(actual_cats) > 1:
                sub_models = []
                for cat in actual_cats:
                    for s_item in specs.get(cat, []):
                        if not isinstance(s_item, dict):
                            continue
                        m_names = [m.get("model", "") for m in s_item.get("model_list", []) if isinstance(m, dict)]
                        sub_models.extend(m for m in m_names if m)
                entry["type"] = "组合大类"
                entry["standard_name"] = _normalize_category_name(raw_name)
                entry["sub_categories"] = actual_cats
                entry["sub_entities"] = sub_models
                results.append(entry)
                continue

            if len(actual_cats) == 1:
                cat_name = actual_cats[0]
                series_list = specs[cat_name]
                if isinstance(series_list, list):
                    sub_models = []
                    for s_item in series_list:
                        if not isinstance(s_item, dict):
                            continue
                        m_names = [m.get("model", "") for m in s_item.get("model_list", []) if isinstance(m, dict)]
                        sub_models.extend(m for m in m_names if m)
                    entry["type"] = "大类"
                    entry["standard_name"] = cat_name
                    entry["sub_entities"] = sub_models
                    results.append(entry)
                    continue

            std_series = _normalize_series_name(raw_name)
            if std_series:
                sub_models = []
                for cat, s_list in specs.items():
                    if not isinstance(s_list, list):
                        continue
                    for s_item in s_list:
                        if not isinstance(s_item, dict):
                            continue
                        if s_item.get("series") == std_series:
                            m_names = [m.get("model", "") for m in s_item.get("model_list", []) if isinstance(m, dict)]
                            sub_models.extend(m for m in m_names if m)
                entry["type"] = "系列"
                entry["standard_name"] = std_series
                entry["sub_entities"] = sub_models
                results.append(entry)
                continue

            std_model = _normalize_model_name(raw_name)
            found = False
            for cat, s_list in specs.items():
                if not isinstance(s_list, list):
                    continue
                for s_item in s_list:
                    if not isinstance(s_item, dict):
                        continue
                    for m_entry in s_item.get("model_list", []):
                        if isinstance(m_entry, dict) and m_entry.get("model") == std_model:
                            entry["type"] = "型号"
                            entry["standard_name"] = std_model
                            entry["category"] = cat
                            entry["series"] = s_item.get("series", "")
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

            if found:
                results.append(entry)
            else:
                entry["type"] = "未知"
                entry["standard_name"] = std_model
                results.append(entry)

        # Format output as XML (WeKnora style)
        parts = ["<entity_resolution_results>"]
        for r in results:
            parts.append(f'  <entity input="{_xml_escape(r["input"])}">')
            parts.append(f'    <type>{r["type"]}</type>')
            parts.append(f'    <standard_name>{_xml_escape(r["standard_name"])}</standard_name>')
            if "sub_categories" in r and r["sub_categories"]:
                parts.append(f'    <sub_categories>{", ".join(r["sub_categories"])}</sub_categories>')
            if "sub_entities" in r and r["sub_entities"]:
                parts.append(f'    <sub_entities>{", ".join(r["sub_entities"])}</sub_entities>')
            if "category" in r:
                parts.append(f'    <category>{r["category"]}</category>')
            if "series" in r:
                parts.append(f'    <series>{r["series"]}</series>')
            parts.append("  </entity>")
        parts.append("</entity_resolution_results>")
        return "\n".join(parts)

    registry.register_tool(ToolDefinition(
        name="resolve_entities",
        description=(
            "Entity resolution: query product hierarchy, normalize aliases, "
            "expand categories/series into sub-entities. "
            "Use when the query involves product-related entities to understand "
            "entity hierarchy before searching. "
            "Supports combined categories (auto-expands to sub-categories and models)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 10,
                    "description": "Entity names to resolve (category, series, model, or alias)",
                },
            },
            "required": ["entities"],
        },
        handler=_re_handler,
    ))


def _format_rerank_result(chunks) -> str:
    """Format reranked chunks as XML."""
    if not chunks:
        return "<rerank_results />"

    parts = ["<rerank_results>"]
    for chunk in chunks:
        doc_name = chunk.doc_title or chunk.doc_id
        score_str = f"{chunk.score:.4f}" if chunk.score else "N/A"
        parts.append(
            f'  <chunk chunk_id="{chunk.chunk_id}" '
            f'knowledge_id="{chunk.doc_id}" '
            f'score="{score_str}">'
        )
        parts.append(f"    <doc_title>{_xml_escape(doc_name)}</doc_title>")
        parts.append(f"    <content>{_xml_escape(chunk.content[:500])}</content>")
        parts.append("  </chunk>")
    parts.append("</rerank_results>")
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
