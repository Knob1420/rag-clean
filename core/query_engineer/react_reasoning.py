"""
多跳 ReAct 推理服务

基于 Think→Act→Observe 迭代式推理，处理需要"先找A，根据A的结果再找B"的多跳问题。

核心设计：
- Hop 0 (seed): 用 RewrittenQuery 字段直接检索（无需 LLM）
- Hop 1..N: LLM 驱动，决定下一步检索或结束
"""

import time
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from config import settings
from core.generation.llm import LLMClient, get_llm_client, parse_json_response
from core.retrieve.retrieval import RetrievalService
from core.retrieve.retrieval_models import (
    HighlightOptions,
    RetrievedChunk,
    RetrievalOptions,
    RetrievalResult,
)
from core.query_engineer.query_rewrite import RewrittenQuery
from prompt import REACT_SYSTEM_PROMPT, build_react_step_prompt


# 累积摘要的最大字数限制
MAX_ACCUMULATED_SUMMARY_CHARS = 1500


class ReActReasoningService:
    """多跳 ReAct 推理服务"""

    def __init__(self, retrieval_svc: Optional[RetrievalService] = None):
        self.retrieval_svc = retrieval_svc
        self.llm = get_llm_client()
        self.max_hops = settings.max_react_hops
        self.seed_top_k = settings.react_seed_top_k
        self.step_top_k = settings.react_step_top_k
        self.consecutive_no_new_limit = settings.react_consecutive_no_new

    # ============================================================
    # 主入口
    # ============================================================

    def reason(
        self,
        original_query: str,
        rewritten: RewrittenQuery,
        options: RetrievalOptions,
    ) -> RetrievalResult:
        """
        多跳推理检索主入口

        Args:
            original_query: 原始用户问题
            rewritten: Query Rewrite 结果
            options: 检索选项

        Returns:
            RetrievalResult: 最终检索结果（统一 rerank 后）
        """
        if self.retrieval_svc is None:
            from core.retrieve.retrieval import get_retrieval_service
            self.retrieval_svc = get_retrieval_service()

        timing = {}
        t0 = time.time()

        # 初始化
        all_chunks: List[RetrievedChunk] = []
        seen_ids: Set[str] = set()
        accumulated_summary = ""
        hop_count = 0
        consecutive_no_new = 0

        logger.info(
            f"[ReAct] 启动多跳推理: query='{original_query}', "
            f"max_hops={self.max_hops}"
        )

        # ── Hop 0: 种子检索（无需 LLM）─────────────────────────
        hop_count += 1
        t_hop = time.time()

        seed_chunks = self._seed_first_hop(rewritten, options)
        for c in seed_chunks:
            if c.chunk_id not in seen_ids:
                all_chunks.append(c)
                seen_ids.add(c.chunk_id)

        seed_summary = self._summarize_chunks(seed_chunks)
        accumulated_summary = seed_summary

        timing[f"hop_{hop_count}"] = time.time() - t_hop
        logger.info(
            f"[ReAct] Hop {hop_count} (seed): "
            f"检索到 {len(seed_chunks)} 条, 累计 {len(all_chunks)} 条"
        )

        # ── Hop 1..N: LLM 驱动循环 ───────────────────────────
        while hop_count < self.max_hops:
            hop_count += 1
            t_hop = time.time()

            # 构建 LLM prompt
            step_prompt = build_react_step_prompt(
                original_query=original_query,
                accumulated_summary=self._truncate_summary(accumulated_summary),
                summarized_results=self._format_chunks_for_llm(seed_chunks),
                hop_number=hop_count,
                max_hops=self.max_hops,
            )

            # 调用 LLM 决定下一步
            try:
                response = self.llm.call(
                    [
                        {"role": "system", "content": REACT_SYSTEM_PROMPT},
                        {"role": "user", "content": step_prompt},
                    ],
                    temperature=0.3,
                )

                step_info = self._parse_step_response(response)
                thought = step_info.get("thought", "")
                action = step_info.get("action", "finish")

                logger.info(
                    f"[ReAct] Hop {hop_count} LLM: action={action}, "
                    f"thought={thought[:100]}..."
                )

            except Exception as e:
                logger.warning(f"[ReAct] Hop {hop_count} LLM 调用失败: {e}")
                action = "finish"

            # 检查是否结束
            if action == "finish":
                logger.info(f"[ReAct] LLM 决定结束推理（Hop {hop_count}）")
                timing[f"hop_{hop_count}"] = time.time() - t_hop
                break

            # 执行检索
            search_query = step_info.get("search_query", rewritten.rewritten_query)
            search_keywords = step_info.get("search_keywords", rewritten.keywords)
            search_entities = step_info.get("search_entities", rewritten.target_entities)
            summary_of_findings = step_info.get("summary_of_findings", "")

            # 构建检索选项
            step_options = options.model_copy(update={
                "top_k": self.step_top_k,
                "target_models": search_entities if search_entities else None,
                "keywords": search_keywords if search_keywords else None,
                "use_rerank": False,  # 每步不 rerank，最后统一 rerank
            })

            # 执行混合检索
            step_chunks = self._execute_search(search_query, step_options)

            # 检查是否有新结果
            new_count = 0
            for c in step_chunks:
                if c.chunk_id not in seen_ids:
                    all_chunks.append(c)
                    seen_ids.add(c.chunk_id)
                    new_count += 1

            # 更新累积摘要
            if summary_of_findings:
                accumulated_summary += f"\n{summary_of_findings}"

            timing[f"hop_{hop_count}"] = time.time() - t_hop
            logger.info(
                f"[ReAct] Hop {hop_count}: "
                f"检索 '{search_query}' → "
                f"新结果 {new_count}/{len(step_chunks)} 条"
            )

            # 连续两跳无新结果，提前终止
            if new_count == 0:
                consecutive_no_new += 1
                if consecutive_no_new >= self.consecutive_no_new_limit:
                    logger.info(f"[ReAct] 连续{consecutive_no_new}跳无新结果，提前终止")
                    break
            else:
                consecutive_no_new = 0

            # 保存本轮检索结果用于下次 LLM 上下文
            seed_chunks = step_chunks

        # ── 统一 Rerank ─────────────────────────────────────
        t_rerank = time.time()
        final_chunks = all_chunks

        if options.use_rerank and all_chunks:
            final_chunks = self._rerank(rewritten.rewritten_query, all_chunks, options)

        # 截断到 top_k
        final_chunks = final_chunks[: options.top_k]
        timing["rerank"] = time.time() - t_rerank
        timing["total"] = time.time() - t0

        logger.info(
            f"[ReAct] 完成: 总计 {len(all_chunks)} 条去重, "
            f"最终 {len(final_chunks)} 条, "
            f"hops={hop_count}, "
            f"time={timing['total']:.2f}s"
        )

        return RetrievalResult(
            query=rewritten.rewritten_query,
            total=len(final_chunks),
            chunks=final_chunks,
            timing=timing,
        )

    # ============================================================
    # Hop 0: 种子检索
    # ============================================================

    def _seed_first_hop(
        self,
        rewritten: RewrittenQuery,
        options: RetrievalOptions,
    ) -> List[RetrievedChunk]:
        """
        Hop 0: 用 RewrittenQuery 字段直接检索（无需 LLM）
        直接复用 rewritten_query + target_entities + keywords
        """
        seed_options = options.model_copy(update={
            "top_k": self.seed_top_k,
            "target_models": (
                rewritten.target_entities if rewritten.target_entities else None
            ),
            "keywords": rewritten.keywords if rewritten.keywords else None,
            "use_rerank": False,
        })

        logger.info(
            f"[ReAct] Seed 检索: query='{rewritten.rewritten_query}', "
            f"entities={rewritten.target_entities}, "
            f"keywords={rewritten.keywords}"
        )

        return self._execute_search(rewritten.rewritten_query, seed_options)

    # ============================================================
    # 检索执行
    # ============================================================

    def _execute_search(
        self,
        query: str,
        options: RetrievalOptions,
    ) -> List[RetrievedChunk]:
        """执行混合检索"""
        try:
            result = self.retrieval_svc.search(
                query=query,
                options=options,
                highlight=HighlightOptions(),
                use_hybrid=True,
            )
            return result.chunks
        except Exception as e:
            logger.warning(f"[ReAct] 检索失败: {e}")
            return []

    def _rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        options: RetrievalOptions,
    ) -> List[RetrievedChunk]:
        """统一 Rerank"""
        try:
            return self.retrieval_svc._rerank(query, chunks, options)
        except Exception as e:
            logger.warning(f"[ReAct] Rerank 失败: {e}")
            return chunks

    # ============================================================
    # 结果格式化
    # ============================================================

    def _summarize_chunks(self, chunks: List[RetrievedChunk]) -> str:
        """
        紧凑摘要：每条 [chunk_type] section_title | context_summary
        最多 15 条，每条约 200 字
        """
        if not chunks:
            return "（暂无检索结果）"

        lines = []
        for i, chunk in enumerate(chunks[:15]):
            chunk_type = chunk.chunk_type or "unknown"
            title = chunk.section_title or "（无标题）"
            summary = chunk.context_summary or chunk.content[:200]

            # 截断过长的 summary
            if len(summary) > 200:
                summary = summary[:200] + "..."

            lines.append(f"[{chunk_type}] {title} | {summary}")

        if len(chunks) > 15:
            lines.append(f"...（还有 {len(chunks) - 15} 条结果）")

        return "\n\n".join(lines)

    def _format_chunks_for_llm(self, chunks: List[RetrievedChunk]) -> str:
        """
        格式化检索结果供 LLM 阅读
        显示：类型、标题、内容摘要
        """
        if not chunks:
            return "（本次检索无结果）"

        lines = []
        for i, chunk in enumerate(chunks[:10], 1):
            chunk_type = chunk.chunk_type or "unknown"
            title = chunk.section_title or "（无标题）"
            content = chunk.content[:300]

            lines.append(
                f"【结果 {i}】\n"
                f"类型: {chunk_type}\n"
                f"标题: {title}\n"
                f"内容: {content}..."
            )

        if len(chunks) > 10:
            lines.append(f"（还有 {len(chunks) - 10} 条结果）")

        return "\n\n".join(lines)

    def _truncate_summary(self, summary: str) -> str:
        """限制累积摘要字数，防止 token 溢出"""
        if len(summary) <= MAX_ACCUMULATED_SUMMARY_CHARS:
            return summary
        return summary[:MAX_ACCUMULATED_SUMMARY_CHARS] + "\n...（已截断）"

    # ============================================================
    # LLM 响应解析
    # ============================================================

    def _parse_step_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 返回的 JSON action

        Returns:
            dict with keys: thought, action, search_query, search_keywords,
                          search_entities, summary_of_findings
        """
        try:
            result = parse_json_response(response)

            return {
                "thought": str(result.get("thought", "")),
                "action": str(result.get("action", "finish")).lower().strip(),
                "search_query": str(result.get("search_query", "")),
                "search_keywords": result.get("search_keywords", []),
                "search_entities": result.get("search_entities", []),
                "summary_of_findings": str(result.get("summary_of_findings", "")),
            }

        except Exception as e:
            logger.warning(f"[ReAct] JSON 解析失败: {e}，视为 finish")
            return {
                "thought": f"解析失败: {e}",
                "action": "finish",
                "search_query": "",
                "search_keywords": [],
                "search_entities": [],
                "summary_of_findings": "",
            }


# ── 全局实例 ──────────────────────────────────────────

_react_reasoning_service: Optional[ReActReasoningService] = None


def get_react_reasoning_service() -> ReActReasoningService:
    """获取 ReAct 推理服务单例"""
    global _react_reasoning_service
    if _react_reasoning_service is None:
        _react_reasoning_service = ReActReasoningService()
    return _react_reasoning_service
