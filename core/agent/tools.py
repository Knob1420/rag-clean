"""
ReAct Agent 工具定义与执行

4 个工具（OpenAI Function Calling schema + 执行逻辑）：
- bm25_search    : BM25 关键词检索（擅长精准匹配型号、编号、专有名词）
- vector_search  : 语义向量检索（擅长理解原理、机制、场景、描述性查询）
- spec_query     : 结构化产品参数补充查询
- finish         : 提交最终答案（终止信号）
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from core.retrieve.retrieval import RetrievalService, RetrievalOptions
from core.retrieve.retrieval_models import RetrievedChunk
from core.products.spec_matcher import query_products, format_spec_context
from store import DocumentStore, get_store

# ═════════════════════════════════════════════════════════════════
# 工具 JSON Schema 定义（OpenAI Function Calling 格式）
# ═════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bm25_search",
            "description": (
                "BM25 关键词检索：基于精确关键词/正则匹配搜索文档。"
                "擅长：型号、编号、专有名词、缩写等精准匹配。"
                "query 写法：用关键词或正则模式，强烈推荐交替查询（如 '*|*|*'）"
                "代替多次单关键词调用；纯文本也有效（'功耗' 匹配任意位置）。"
                "支持分组并发检索：可一次性传入多个检索分组，每个分组对应一个独立子问题。"
                "检索结果按分组返回，自动包含完整文档内容。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "子问题描述",
                                },
                                "queries": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                    "maxItems": 5,
                                    "description": "该子问题的检索 query 列表",
                                },
                            },
                            "required": ["label", "queries"],
                        },
                        "minItems": 1,
                        "maxItems": 5,
                        "description": "检索分组列表，每个 group 对应一个独立子问题",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "每个 query 返回结果数量，默认 10",
                        "default": 10,
                    },
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": (
                "语义向量检索：基于语义相似度搜索文档。擅长原理、机制、场景、描述性查询、概念理解。\n"
                "**必须使用 HyDE**：每条 query 内部会通过 LLM 生成假设性文档并融合 embedding，"
                "大幅提升语义匹配精度（尤其对复杂问句效果显著）。\n"
                "**用法**：每次调用只需传入一条语义通顺的自然语言句子，不要拆分为多组。\n"
                "query 写法：完整的自然语言问题或陈述，语义越完整匹配越好。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "一条语义通顺的自然语言查询（完整句子，不要用关键词拼接）",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，默认 10",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spec_query",
            "description": (
                "产品参数精确筛选查询：用于查询产品的数值型参数（重量、功耗、算力、体积、接口数量等）。\n"
                "**参数库范围**：spec 中包含所有产品的完整参数数据。\n"
                "**输入格式（entities）**：\n"
                "  - 输入一个大类  → 返回该大类下全部产品型号及参数\n"
                "  - 输入具体型号 → 返回该型号的完整参数\n"
                "  - 输入系列名 → 返回该系列下所有型号的参数\n"
                "**约束格式（constraints）**：可选，用于数值筛选，如 {'重量': '<=2', '算力': '>250', '功耗': '<150'}，"
                "支持 >, <, >=, <=, == 等比较符。\n"
                "**使用时机**：当问题涉及具体参数数值查询或约束时，直接使用此工具，无需先通过 BM25/vector 确认型号。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "目标产品/系列/类别列表，支持组合大类和混合输入",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "需要的参数字段列表（如 ['重量', '功耗', '算力']）",
                    },
                    "constraints": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "数值约束，如 {'重量': '<=3.0', '算力': '>100'}",
                    },
                },
                "required": ["entities"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "提交最终回答：这是你的终止信号。当你已经收集到足够信息，"
                "准备向用户输出最终答案时，必须调用此工具。"
                "绝不可以在未调用此工具的情况下结束对话。"
                "answer 参数包含完整的最终回答（支持 Markdown 格式）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "完整的最终回答内容（Markdown 格式）",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


# ═════════════════════════════════════════════════════════════════
# 工具输出截断
# ═════════════════════════════════════════════════════════════════

MAX_TOOL_OUTPUT_CHARS = 8000
HEAD_RATIO = 0.8


def _truncate_output(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """截断工具输出，保留头尾"""
    if len(text) <= max_chars:
        return text
    head_len = int(max_chars * HEAD_RATIO)
    tail_len = max_chars - head_len - 50  # 50 chars for truncation marker
    return (
        text[:head_len]
        + f"\n\n... [截断：原始 {len(text)} 字符，保留 {max_chars} 字符] ...\n\n"
        + text[-tail_len:]
    )


def _format_chunks_summary(chunks: List[RetrievedChunk]) -> str:
    """将检索结果格式化为摘要供 LLM 消费（完整内容）"""
    if not chunks:
        return "未找到匹配结果。"

    parts = [f"找到 {len(chunks)} 条匹配结果：\n"]
    for i, chunk in enumerate(chunks):
        doc_name = chunk.doc_title or chunk.doc_id
        score_str = f"{chunk.score:.4f}" if chunk.score else "N/A"
        parts.append(
            f"[{i+1}] 文档={doc_name} | "
            f"分数={score_str}\n"
            f"  内容: {chunk.content}"
        )
    return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════
# 工具执行引擎
# ═════════════════════════════════════════════════════════════════


class ToolExecutor:
    """工具执行器 — 管理工具调用分发和状态"""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        store: Optional[DocumentStore] = None,
    ):
        self._retrieval = retrieval_service or RetrievalService()
        self._store = store or get_store()
        # 累积的 chunks（跨步骤）
        self.accumulated_chunks: List[RetrievedChunk] = []

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        执行工具调用，返回结果字符串。

        Args:
            tool_name: 工具名称
            arguments: 解析后的参数字典

        Returns:
            工具执行结果（已截断）
        """
        try:
            if tool_name == "bm25_search":
                result = self._exec_bm25_search(arguments)
            elif tool_name == "vector_search":
                result = self._exec_vector_search(arguments)
            elif tool_name == "search_knowledge":
                # 兼容旧调用：路由到 BM25
                logger.warning(
                    "[ToolExecutor] search_knowledge 已拆分，路由到 bm25_search"
                )
                result = self._exec_bm25_search(arguments)
            elif tool_name == "spec_query":
                result = self._exec_spec_query(arguments)
            elif tool_name == "finish":
                result = arguments.get("answer", "")
            else:
                result = f"错误：未知工具 '{tool_name}'"

            return _truncate_output(result)

        except Exception as e:
            logger.error(f"[ToolExecutor] 工具 {tool_name} 执行失败: {e}")
            return f"工具执行出错: {str(e)}。请分析错误并尝试不同方法。"

    # ── bm25_search（关键词检索）───────────────────────────────────────

    def _exec_bm25_search(self, args: Dict[str, Any]) -> str:
        """
        BM25 关键词检索：支持分组并发，按分组返回结果。
        """
        raw_queries = args.get("queries", [])
        top_k = args.get("top_k", 10)

        if not raw_queries:
            return "错误：queries 不能为空"

        # 兼容旧格式
        if raw_queries and isinstance(raw_queries[0], str):
            groups = [{"label": "查询", "queries": raw_queries}]
        else:
            groups = raw_queries

        from concurrent.futures import ThreadPoolExecutor, as_completed

        options = RetrievalOptions(top_k=top_k, use_rerank=False)

        def _search_single_bm25(q: str) -> Tuple[str, List[RetrievedChunk]]:
            try:
                # 跳过 _build_bm25_query：LLM 已决定关键词策略，直接作为 Lucene query_string 发给 ES
                # Lucene query_string 语法支持 | 交替、+ - 布尔、引号短语 等
                chunks = self._retrieval._execute_bm25(q, options, top_k)
            except Exception as e:
                logger.warning(f"[bm25_search] query='{q}' 失败: {e}")
                return (q, [])
            if chunks:
                from core.query_engineer.rerank_query import build_rerank_query

                rerank_q = build_rerank_query(q)
                chunks = self._auto_rerank(chunks, rerank_q, top_k)
            return (q, chunks)

        group_results: List[Tuple[str, str, List[RetrievedChunk]]] = []
        all_futures = {}

        with ThreadPoolExecutor(
            max_workers=min(sum(len(g["queries"]) for g in groups), 5)
        ) as pool:
            for group in groups:
                label = group.get("label", "查询")
                for q in group["queries"]:
                    future = pool.submit(_search_single_bm25, q)
                    all_futures[future] = (label, q)

            for future in as_completed(all_futures):
                label, q = all_futures[future]
                try:
                    query, chunks = future.result()
                    group_results.append((label, query, chunks))
                except Exception as e:
                    logger.warning(
                        f"[bm25_search] group='{label}' query='{q}' 执行异常: {e}"
                    )

        # 保持原始顺序
        group_order = {}
        idx = 0
        for group in groups:
            label = group.get("label", "查询")
            for q in group["queries"]:
                group_order[(label, q)] = idx
                idx += 1
        group_results.sort(key=lambda x: group_order.get((x[0], x[1]), 0))

        # 累积 chunks
        global_seen: Set[str] = set()
        for _, _, chunks in group_results:
            for chunk in chunks:
                if chunk.chunk_id not in global_seen:
                    self.accumulated_chunks.append(chunk)
                    global_seen.add(chunk.chunk_id)

        return self._format_grouped_results(group_results, search_type="BM25")

    # ── vector_search（语义检索 + HyDE）─────────────────────────────────

    def _exec_vector_search(self, args: Dict[str, Any]) -> str:
        """
        语义向量检索：单条 query + HyDE，增强语义匹配。

        Args:
            args: {"query": "完整自然语言句子", "top_k": int}
        """
        query = args.get("query", "")
        top_k = args.get("top_k", 10)

        if not query:
            return "错误：query 不能为空"

        from core.query_engineer.hyde import HyDEQueryEngine
        from core.client.embedder import encode

        hyde_engine = HyDEQueryEngine(num_hypotheses=1)
        hyde_result = hyde_engine.transform(query)

        if hyde_result.fused_embedding is None:
            logger.warning(
                f"[vector_search] HyDE 融合 embedding 失败，降级为 raw embedding"
            )
            query_vector = encode(query)
            if query_vector is None:
                return "向量化服务失败"
        else:
            import numpy as np

            query_vector = np.array(hyde_result.fused_embedding, dtype=np.float32)

        logger.info(
            f"[vector_search] HyDE 完成: hypothetical_doc={hyde_result.hypothetical_docs[0][:50] if hyde_result.hypothetical_docs else 'N/A'}..."
        )

        options = RetrievalOptions(top_k=top_k, use_rerank=False)
        try:
            chunks = self._retrieval._execute_vector_search(
                query_vector, options, top_k
            )
        except Exception as e:
            logger.warning(f"[vector_search] query='{query}' 失败: {e}")
            return f"向量检索失败: {e}"

        if chunks:
            from core.query_engineer.rerank_query import build_rerank_query

            rerank_q = build_rerank_query(query)
            chunks = self._auto_rerank(chunks, rerank_q, top_k)

        # 累积 chunks
        global_seen: Set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id not in global_seen:
                self.accumulated_chunks.append(chunk)
                global_seen.add(chunk.chunk_id)

        return self._format_vector_results(query, chunks)

    def _format_grouped_results(
        self,
        group_results: List[Tuple[str, str, List[RetrievedChunk]]],
        search_type: str = "",
    ) -> str:
        """按 group 分组格式化检索结果，让 LLM 能清晰区分不同子问题的结果。"""
        if not group_results:
            return "未找到匹配结果。"

        # 按 label 分组，保持顺序
        from collections import OrderedDict

        grouped = OrderedDict()
        for label, query, chunks in group_results:
            grouped.setdefault(label, []).append((query, chunks))

        parts = []
        for gi, (label, query_items) in enumerate(grouped.items()):
            type_tag = f" [{search_type}]" if search_type else ""
            parts.append(f"{'━' * 10} 子问题: {label}{type_tag} {'━' * 10}")
            for qi, (query, chunks) in enumerate(query_items):
                parts.append(f'  ═══ 查询 {qi+1}: "{query}" ═══\n')
                if not chunks:
                    parts.append("  未找到匹配结果。\n")
                else:
                    parts.append(f"  找到 {len(chunks)} 条匹配结果：")
                    for j, chunk in enumerate(chunks):
                        doc_name = chunk.doc_title or chunk.doc_id
                        score_str = f"{chunk.score:.4f}" if chunk.score else "N/A"
                        parts.append(
                            f"  [{j+1}] 文档={doc_name} | "
                            f"分数={score_str}\n"
                            f"    内容: {chunk.content}"
                        )
                    parts.append("")  # 空行分隔

        return "\n".join(parts)

    def _format_vector_results(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> str:
        """格式化 vector_search 结果（单条 query，无需 group）"""
        if not chunks:
            return "未找到匹配结果。"

        parts = [
            f"{'━' * 10} Vector 检索 [HyDE] {'━' * 10}",
            f'  查询: "{query}"\n',
            f"  找到 {len(chunks)} 条匹配结果：",
        ]
        for j, chunk in enumerate(chunks):
            doc_name = chunk.doc_title or chunk.doc_id
            score_str = f"{chunk.score:.4f}" if chunk.score else "N/A"
            parts.append(
                f"  [{j+1}] 文档={doc_name} | 分数={score_str}\n"
                f"    内容: {chunk.content}"
            )
        return "\n".join(parts)

    def _auto_rerank(
        self,
        chunks: List[RetrievedChunk],
        rerank_query: str,
        top_k: int,
    ) -> List[RetrievedChunk]:
        """单 query 自动 rerank + score 阈值过滤

        Args:
            chunks: 待 rerank 的候选结果
            rerank_query: 增强后的 rerank 查询（由调用方通过 build_rerank_query 构建）
            top_k: 返回数量
        """
        if not chunks:
            return chunks

        try:
            from core.client.rerank_client import rerank_documents

            documents = [c.content for c in chunks]
            rerank_results = rerank_documents(
                query=rerank_query,
                documents=documents,
                top_k=top_k,
            )
            reranked_map = {doc: score for doc, score in rerank_results}
            for chunk in chunks:
                if chunk.content in reranked_map:
                    chunk.score = reranked_map[chunk.content]

            chunks.sort(key=lambda c: c.score, reverse=True)

            from config import settings

            threshold = settings.rerank_score_threshold
            if threshold > 0:
                before_count = len(chunks)
                chunks = [c for c in chunks if c.score >= threshold]
                logger.info(
                    f"[search_knowledge] score 阈值过滤: {before_count} → {len(chunks)} "
                    f"(threshold={threshold})"
                )

            return chunks[:top_k]

        except Exception as e:
            logger.warning(f"[search_knowledge] 自动 rerank 失败: {e}")
            return chunks[:top_k]

    # ── 自动 parent 展开 ──────────────────────────────────────────────────

    def _expand_to_parent_chunks(
        self,
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        将检索到的 child/summary chunks 做智能展开。

        逻辑（同 rag_pipeline）：
        - summary chunk → 展开为 parent chunk
        - 同一个 parent 有 >= 1 个 child chunk → 展开为 parent chunk
        - 无 parent_id 的 chunk → 保持原样
        """
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
        parent_ids_to_expand = list(parent_children.keys())

        # 3. 批量拉取 parent chunks
        from config import settings

        parent_map: Dict[str, Dict] = {}
        if parent_ids_to_expand:
            try:
                resp = self._store.es.mget(
                    index=settings.es_index_chunks,
                    body={"ids": parent_ids_to_expand},
                )
                for doc in resp.get("docs", []):
                    if doc.get("found") and doc.get("_source"):
                        parent_map[doc["_id"]] = doc["_source"]
            except Exception as e:
                logger.warning(f"[ToolExecutor] 批量获取 parent chunk 失败: {e}")

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

            # 展开：summary 或 >=1 child from same parent → parent chunk
            if chunk.parent_id in parent_map:
                if chunk.parent_id not in seen_ids:
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
                        parent_id=None,
                    )
                    result.append(parent_chunk)
                    seen_ids.add(chunk.parent_id)
                # 同 parent 的其他 child 跳过（已展开为 parent）
                continue
            else:
                # parent 未找到，保留原 child
                if chunk.chunk_id not in seen_ids:
                    result.append(chunk)
                    seen_ids.add(chunk.chunk_id)

        return result

    # ── spec_query ──────────────────────────────────────────────────

    def _exec_spec_query(self, args: Dict[str, Any]) -> str:
        entities = args.get("entities", [])
        fields = args.get("fields", [])
        constraints = args.get("constraints", {})

        if not entities:
            return "错误：entities 不能为空"

        results = query_products(
            target_models=entities,
            required_fields=fields,
            numerical_constraints=constraints,
        )

        if not results:
            return f"未找到匹配 '{', '.join(entities)}' 的产品参数。"

        formatted = format_spec_context(results, "recommend")
        return formatted

    # ── 内部辅助 ────────────────────────────────────────────────────

    def _accumulate_chunks(self, new_chunks: List[RetrievedChunk]) -> None:
        """将新检索结果累积到状态中（去重）"""
        seen_ids = {c.chunk_id for c in self.accumulated_chunks}
        for chunk in new_chunks:
            if chunk.chunk_id not in seen_ids:
                self.accumulated_chunks.append(chunk)
                seen_ids.add(chunk.chunk_id)

    def expand_accumulated_chunks(self) -> None:
        """将累积的 child/summary chunks 统一展开为 parent chunks（finish 时调用）"""
        if not self.accumulated_chunks:
            return
        self.accumulated_chunks = self._expand_to_parent_chunks(self.accumulated_chunks)
        logger.info(
            f"[ToolExecutor] parent 展开完成，当前累积 chunks: {len(self.accumulated_chunks)}"
        )

    def set_current_query(self, query: str) -> None:
        """设置当前查询（保留接口兼容，search_knowledge 自动处理）"""
        pass

    def reset(self) -> None:
        """重置累积状态（新查询时调用）"""
        self.accumulated_chunks = []
