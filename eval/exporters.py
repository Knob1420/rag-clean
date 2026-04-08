"""
导出逻辑 — 跑检索管线，收集各阶段结果导出为 JSON

调用现有 service 内部方法，不修改管线代码。
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from config import settings
from core.retrieve.embedder import encode
from core.retrieve.retrieval_models import (
    HighlightOptions,
    RetrievedChunk,
    RetrievalOptions,
)


# ============================================================
# 工具函数
# ============================================================


# doc_id → doc_name 缓存（从 documents 索引批量查询）
_doc_name_cache: Dict[str, str] = {}


def _ensure_doc_name_cache(doc_ids: set):
    """批量从 documents 索引补齐 doc_name 缓存"""
    missing = doc_ids - _doc_name_cache.keys()
    if not missing:
        return
    try:
        from store import get_store

        store = get_store()
        resp = store.es.search(
            index=settings.es_index_documents,
            body={
                "query": {"terms": {"doc_id": list(missing)}},
                "size": len(missing),
                "_source": ["doc_id", "title"],
            },
        )
        for hit in resp.get("hits", {}).get("hits", []):
            src = hit["_source"]
            _doc_name_cache[src["doc_id"]] = src.get("title", "")
        logger.info(f"[export] doc_name 缓存已更新: +{len(missing)} doc_ids")
    except Exception as e:
        logger.warning(f"[export] 获取 doc_name 失败: {e}")


def _chunk_to_dict(chunk: RetrievedChunk, rank: int) -> dict:
    """RetrievedChunk → JSON-friendly dict"""
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_name": _doc_name_cache.get(chunk.doc_id, ""),
        "content": chunk.content,
        "context_summary": chunk.context_summary,
        "section_title": chunk.section_title,
        "chunk_type": chunk.chunk_type,
        "filter_terms": chunk.filter_terms,
        "keywords": chunk.keywords,
        "score": round(chunk.score, 4),
    }


def _chunks_to_list(chunks: List[RetrievedChunk]) -> List[dict]:
    """List[RetrievedChunk] → ranked list of dicts"""
    return [_chunk_to_dict(c, i + 1) for i, c in enumerate(chunks)]


# ============================================================
# Pipeline Exporter
# ============================================================


class PipelineExporter:
    """跑管线各阶段，导出结果"""

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

        # 延迟初始化 service（避免 import 时就连接 ES）
        self._retrieval_svc = None
        self._rewrite_svc = None
        self._generation_svc = None

    def _get_retrieval_service(self):
        if self._retrieval_svc is None:
            from core.retrieve.retrieval import get_retrieval_service

            self._retrieval_svc = get_retrieval_service()
        return self._retrieval_svc

    def _get_rewrite_service(self):
        if self._rewrite_svc is None:
            from core.query_engineer.query_rewrite import get_query_rewrite_service

            self._rewrite_svc = get_query_rewrite_service()
        return self._rewrite_svc

    def _get_generation_service(self):
        if self._generation_svc is None:
            from core.generation.generation import get_generation_service

            self._generation_svc = get_generation_service()
        return self._generation_svc

    # ── 单阶段导出 ──────────────────────────────────────

    def _export_rewrite(self, query: str) -> Optional[dict]:
        """Query Rewrite 阶段"""
        try:
            svc = self._get_rewrite_service()
            rw = svc.rewrite(query)
            return {
                "original_query": rw.original_query,
                "rewritten_query": rw.rewritten_query,
                "intent_type": rw.intent_type,
                "target_entities": rw.target_entities,
                "keywords": rw.keywords,
                "strategy": rw.strategy,
                "sub_queries": rw.sub_queries,
            }
        except Exception as e:
            logger.error(f"[export] rewrite 失败: {e}")
            return None

    def _export_bm25(
        self,
        query: str,
        options: RetrievalOptions,
        highlight: HighlightOptions,
    ) -> dict:
        """BM25 阶段"""
        svc = self._get_retrieval_service()
        bm25_options = options.model_copy(update={"use_rerank": False})
        chunks = svc._execute_bm25(query, bm25_options, highlight, self.top_k)
        return {
            "query_used": query,
            "filters_applied": self._options_to_filters(bm25_options),
            "results": _chunks_to_list(chunks),
        }

    def _export_vector(
        self,
        query: str,
        options: RetrievalOptions,
    ) -> dict:
        """向量检索阶段"""
        query_vector = encode(query)
        if query_vector is None:
            return {"query_used": query, "filters_applied": {}, "results": []}

        svc = self._get_retrieval_service()
        vec_options = options.model_copy(update={"use_rerank": False})
        chunks = svc._execute_vector_search(query_vector, vec_options, self.top_k)
        return {
            "query_used": query,
            "filters_applied": self._options_to_filters(vec_options),
            "results": _chunks_to_list(chunks),
        }

    def _export_hybrid_rrf(
        self,
        query: str,
        options: RetrievalOptions,
        highlight: HighlightOptions,
    ) -> dict:
        """混合检索 RRF 融合阶段"""
        svc = self._get_retrieval_service()
        hyb_options = options.model_copy(update={"use_rerank": False})
        chunks = svc._hybrid_search(query, hyb_options, highlight)
        chunks = chunks[: self.top_k]
        return {
            "query_used": query,
            "filters_applied": self._options_to_filters(hyb_options),
            "results": _chunks_to_list(chunks),
        }

    def _export_reranked(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        options: RetrievalOptions,
    ) -> dict:
        """Rerank 阶段（基于 hybrid 结果）"""
        svc = self._get_retrieval_service()
        rerank_options = options.model_copy(update={"use_rerank": True})
        reranked = svc._rerank(query, chunks, rerank_options)
        reranked = reranked[: self.top_k]
        return {"results": _chunks_to_list(reranked)}

    def _export_generation(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        rewrite_data: Optional[dict],
    ) -> Optional[dict]:
        """Generation 阶段"""
        try:
            svc = self._get_generation_service()
            # 取 top-5 chunks 作为上下文
            context_chunks = chunks[:5]
            answer, usage = svc.generate(
                query=query,
                chunks=context_chunks,
                query_intent=rewrite_data.get("intent_type") if rewrite_data else None,
                query_entities=None,
            )
            return {
                "answer": answer,
                "token_usage": {
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "total": usage.total_tokens,
                },
            }
        except Exception as e:
            logger.error(f"[export] generation 失败: {e}")
            return None

    # ── Filter 消融实验 ──────────────────────────────────

    def _export_filter_breakdown(
        self,
        query: str,
        rewrite_data: Optional[dict],
        highlight: HighlightOptions,
    ) -> dict:
        """
        对每个 query 跑 5 种 BM25 + 5 种 Vector 变体：
        1. 全量 filter（baseline）
        2. 去掉 chunk_types
        3. 去掉 keywords
        4. 去掉 target_model
        5. 裸查询（无 filter）
        """
        svc = self._get_retrieval_service()

        # 从 rewrite 结果构建 baseline options
        baseline = self._build_options_from_rewrite(rewrite_data)

        variants: Dict[str, RetrievalOptions] = {
            "full_filter": baseline,
            "no_chunk_types": baseline.model_copy(update={"chunk_types": None}),
            "no_keywords": baseline.model_copy(update={"keywords": None}),
            "no_target_models": baseline.model_copy(
                update={"target_models": None}
            ),
            "no_filter": RetrievalOptions(top_k=self.top_k, use_rerank=False),
        }

        breakdown = {}
        for name, opt in variants.items():
            # BM25 变体
            bm25_chunks = svc._execute_bm25(query, opt, highlight, 10)
            # Vector 变体
            query_vector = encode(query)
            vec_chunks = []
            if query_vector is not None:
                vec_chunks = svc._execute_vector_search(query_vector, opt, 10)

            breakdown[name] = {
                "bm25": _chunks_to_list(bm25_chunks),
                "vector": _chunks_to_list(vec_chunks),
            }

        return breakdown

    # ── 完整导出 ────────────────────────────────────────

    def export_query(
        self,
        query_id: str,
        query: str,
        question_type: str,
        stages: Optional[Set[str]] = None,
        no_rewrite: bool = False,
    ) -> dict:
        """导出单条 query 的全链路结果"""
        entry: Dict[str, Any] = {
            "id": query_id,
            "query": query,
            "question_type": question_type,
        }

        # "all" 默认不含 generation（耗时且需 LLM）和 filter_ablation
        all_stages = stages is None or "all" in stages

        # 1. Query Rewrite
        rewrite_data = None
        if all_stages or "rewrite" in stages:
            if no_rewrite:
                rewrite_data = {
                    "original_query": query,
                    "rewritten_query": query,
                    "intent_type": "other",
                    "target_entities": [],
                    "keywords": [],
                    "strategy": "direct",
                    "sub_queries": [],
                }
            else:
                rewrite_data = self._export_rewrite(query)
            if rewrite_data:
                entry["rewrite"] = rewrite_data

        # 构建检索 query 和 options
        search_query = rewrite_data["rewritten_query"] if rewrite_data else query
        options = self._build_options_from_rewrite(rewrite_data)
        highlight = HighlightOptions()

        # 判断检索路由策略（与 api/main.py 逻辑一致）
        strategy = rewrite_data.get("strategy", "direct") if rewrite_data else "direct"
        is_parallel = strategy == "parallel" and rewrite_data and rewrite_data.get("sub_queries")
        is_sequential = strategy == "sequential"

        # 保存 hybrid 原始 chunks 用于 rerank 和 generation
        hybrid_chunks_raw: List[RetrievedChunk] = []

        # 2. BM25（用 rewritten_query 做单次检索，作为 baseline）
        if all_stages or "bm25" in stages:
            logger.info(f"[export] q={query_id} BM25...")
            entry["bm25"] = self._export_bm25(search_query, options, highlight)

        # 3. Vector（用 rewritten_query 做单次检索，作为 baseline）
        if all_stages or "vector" in stages:
            logger.info(f"[export] q={query_id} Vector...")
            entry["vector"] = self._export_vector(search_query, options)

        # 4. Hybrid RRF / Routed Search
        if all_stages or "hybrid" in stages:
            svc = self._get_retrieval_service()
            hyb_opt = options.model_copy(update={"use_rerank": False})

            if is_parallel:
                # Parallel: 多子查询路由检索（与 api/main.py 一致）
                logger.info(f"[export] q={query_id} Routed Hybrid ({len(rewrite_data['sub_queries'])} sub_queries)...")
                routed_result = svc.search_routed(
                    rewritten_query=search_query,
                    sub_queries=rewrite_data["sub_queries"],
                    options=hyb_opt,
                    highlight=highlight,
                )
                hybrid_chunks_raw = routed_result.chunks[: self.top_k]
                entry["hybrid_rrf"] = {
                    "query_used": search_query,
                    "sub_queries": rewrite_data["sub_queries"],
                    "filters_applied": self._options_to_filters(hyb_opt),
                    "results": _chunks_to_list(hybrid_chunks_raw),
                }
            else:
                # Simple: 普通混合检索
                logger.info(f"[export] q={query_id} Hybrid RRF...")
                hybrid_chunks_raw = svc._hybrid_search(search_query, hyb_opt, highlight)[
                    : self.top_k
                ]
                entry["hybrid_rrf"] = {
                    "query_used": search_query,
                    "filters_applied": self._options_to_filters(hyb_opt),
                    "results": _chunks_to_list(hybrid_chunks_raw),
                }

        # 5. Rerank
        if all_stages or "rerank" in stages:
            logger.info(f"[export] q={query_id} Rerank...")
            if not hybrid_chunks_raw:
                # 如果没跑 hybrid，先跑一次
                svc = self._get_retrieval_service()
                hyb_opt = options.model_copy(update={"use_rerank": False})
                if is_parallel:
                    routed_result = svc.search_routed(
                        rewritten_query=search_query,
                        sub_queries=rewrite_data["sub_queries"],
                        options=hyb_opt,
                        highlight=highlight,
                    )
                    hybrid_chunks_raw = routed_result.chunks[: self.top_k]
                else:
                    hybrid_chunks_raw = svc._hybrid_search(
                        search_query, hyb_opt, highlight
                    )[: self.top_k]
            entry["reranked"] = self._export_reranked(
                search_query, hybrid_chunks_raw, options
            )

        # 6. Generation（需显式指定 "generation"，all 不含）
        if "generation" in (stages or set()):
            logger.info(f"[export] q={query_id} Generation...")
            gen_chunks = hybrid_chunks_raw
            if not gen_chunks:
                # fallback：用 bm25 结果
                svc = self._get_retrieval_service()
                bm25_opt = options.model_copy(update={"use_rerank": False})
                gen_chunks = svc._execute_bm25(
                    search_query, bm25_opt, highlight, self.top_k
                )
            entry["generation"] = self._export_generation(
                search_query, gen_chunks, rewrite_data
            )

        # 7. Filter Breakdown
        if all_stages or "filter_ablation" in stages:
            logger.info(f"[export] q={query_id} Filter Breakdown...")
            entry["filter_breakdown"] = self._export_filter_breakdown(
                search_query, rewrite_data, highlight
            )

        # 占位：待用户标注
        entry["relevant_chunk_ids"] = []

        return entry

    def export_dataset(
        self,
        dataset: dict,
        output_path: str,
        stages: Optional[Set[str]] = None,
        no_rewrite: bool = False,
    ) -> str:
        """导出整个查询集"""
        queries = dataset.get("queries", [])
        meta = dataset.get("metadata", {})

        export_data = {
            "export_metadata": {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "dataset_name": meta.get("name", ""),
                "dataset_version": meta.get("version", "1.0"),
                "top_k": self.top_k,
                "stages": list(stages) if stages else ["all"],
                "no_rewrite": no_rewrite,
                "num_queries": len(queries),
            },
            "queries": [],
        }

        total = len(queries)
        for i, q in enumerate(queries, 1):
            qid = q["id"]
            query_text = q["query"]
            qtype = q.get("question_type", "其他复杂")

            logger.info(f"[export] ({i}/{total}) qid={qid}: {query_text}")
            entry = self.export_query(qid, query_text, qtype, stages, no_rewrite)
            export_data["queries"].append(entry)

            # 每条 query 导出后，收集 doc_ids 刷新 doc_name 缓存
            all_doc_ids = set()
            for stage_key in ("bm25", "vector", "hybrid_rrf"):
                stage_data = entry.get(stage_key, {})
                for r in stage_data.get("results", []):
                    if r.get("doc_id"):
                        all_doc_ids.add(r["doc_id"])
            if all_doc_ids:
                _ensure_doc_name_cache(all_doc_ids)
                # 用缓存回填 doc_name
                for stage_key in ("bm25", "vector", "hybrid_rrf", "reranked"):
                    stage_data = entry.get(stage_key, {})
                    for r in stage_data.get("results", []):
                        if r.get("doc_id") and not r.get("doc_name"):
                            r["doc_name"] = _doc_name_cache.get(r["doc_id"], "")

        # 写入文件
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"[export] 导出完成: {output} ({total} queries)")
        return str(output)

    # ── 辅助 ────────────────────────────────────────────

    @staticmethod
    def _options_to_filters(options: RetrievalOptions) -> dict:
        """RetrievalOptions → filters_applied dict（含说明）"""
        filters: Dict[str, Any] = {}
        if options.chunk_types:
            filters["chunk_types"] = {
                "value": options.chunk_types,
                "effect_bm25": f"should 条件: term/terms 匹配 chunk_type={options.chunk_types}, boost=2.0 → 含该 chunk_type 的文档得分偏好提升",
                "effect_vector": f"knn should 条件: term 匹配 chunk_type boost=2.0 → 候选池中含该 chunk_type 的文档相似度加权",
            }
        if options.keywords:
            filters["keywords"] = {
                "value": options.keywords,
                "effect_bm25": f"should 条件: multi_match 用 '{' '.join(options.keywords)}' 检索 content+entities_text, cross_fields, boost=2.0 → 含关键词的文档得分 ×2 提升",
                "effect_vector": f"should 条件: multi_match 用 '{' '.join(options.keywords)}' 检索 content+entities_text, best_fields, boost=1.5 → 候选池中含关键词的文档相似度加权",
            }
        if options.target_models:
            filters["target_models"] = {
                "value": options.target_models,
                "effect_bm25": f"should 条件: multi_match 在 keywords+context_summary+content 中匹配任一实体, best_fields, boost=3.0 → 含目标实体的文档得分偏好提升",
                "effect_vector": f"knn should 条件: multi_match 在 keywords+context_summary+content 中匹配任一实体, best_fields, boost=3.0 → 候选池中含目标实体的文档相似度加权",
            }

        return filters

    @staticmethod
    def _build_options_from_rewrite(rewrite_data: Optional[dict]) -> RetrievalOptions:
        """从 rewrite 结果构建 RetrievalOptions"""
        if not rewrite_data:
            return RetrievalOptions(use_rerank=False)

        entities = rewrite_data.get("target_entities")

        return RetrievalOptions(
            top_k=20,
            target_models=entities if entities else None,
            keywords=rewrite_data.get("keywords"),
            chunk_types=(
                [rewrite_data["intent_type"]]
                if rewrite_data.get("intent_type")
                and rewrite_data["intent_type"] != "other"
                else None
            ),
            use_rerank=False,
        )
