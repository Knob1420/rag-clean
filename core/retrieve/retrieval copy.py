"""
混合检索服务 — BM25 + 向量检索 + RRF 融合

适配 rag-clean 实际 ES 字段：
- chunk_id, doc_id, doc_hash, doc_title, dataset_id, chunk_type
- content, parent_id, embedding_vector, is_latest, created_at

两次 ES 请求（BM25 + vector）→ Python 端 RRF 融合：
- BM25 和 vector 各自返回 top_k*2 个候选，避免截断
- RRF 融合保留双方排名信息，分数可解释

高层方法（_bm25_search, _vector_search, _hybrid_search, search）：
- 接受原始 query，内部负责 keyword extraction + synonym + DSL 构建 + encode + HyDE
底层方法（_execute_bm25, _execute_vector_search）：
- 接受预处理后的 query_string / query_vector，纯执行 ES 查询
Rerank：
- _rerank 接受原始 query，内部构建增强 rerank_query（keyword extraction + 权重重复）
"""

import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config import settings
from core.ingestion.cleaner import is_low_quality_content
from core.query_engineer.keyword_extractor import get_keyword_extractor
from core.query_engineer.synonym import get_synonym_lookup
from core.retrieve.retrieval_models import (
    RetrievedChunk,
    RetrievalOptions,
    RetrievalResult,
)
from store import DocumentStore, get_store

# ============================================================
# RRF 融合
# ============================================================


class RRFFusion:
    """Reciprocal Rank Fusion 融合算法"""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        bm25_results: List[Tuple[RetrievedChunk, int]],
        vector_results: List[Tuple[RetrievedChunk, int]],
        bm25_weight: float = 0.05,
        vector_weight: float = 0.95,
    ) -> List[Tuple[RetrievedChunk, float]]:
        """融合两个检索结果，按融合得分降序排列"""
        scores: Dict[str, float] = defaultdict(float)
        chunks: Dict[str, RetrievedChunk] = {}

        for chunk, rank in bm25_results:
            scores[chunk.chunk_id] += bm25_weight * (1.0 / (self.k + rank))
            chunks[chunk.chunk_id] = chunk

        for chunk, rank in vector_results:
            if chunk.chunk_id not in chunks:
                chunks[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] += vector_weight * (1.0 / (self.k + rank))

        ranked = [(chunks[cid], score) for cid, score in scores.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


# ============================================================
# Lucene query_string 特殊字符转义
# ============================================================

# Lucene query_string 语法中的特殊字符
_LUCENE_SPECIAL_CHARS = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\\/])')


def _sub_special_char(text: str) -> str:
    """转义 Lucene query_string 中的特殊字符（参照 RAGFlow QueryBase.sub_special_char）"""
    return _LUCENE_SPECIAL_CHARS.sub(r"\\\1", text)


# ============================================================
# 检索服务
# ============================================================


class RetrievalService:
    """混合检索服务类"""

    def __init__(self, store: Optional[DocumentStore] = None):
        self._store = store or get_store()
        self.chunks_index = settings.es_index_chunks
        self.default_top_k = settings.default_top_k
        self.rrf = RRFFusion(k=60)

    @property
    def es(self):
        return self._store.es

    # ============================================================
    # 主入口
    # ============================================================

    def search(
        self,
        query: str,
        options: Optional[RetrievalOptions] = None,
        use_hybrid: bool = True,
    ) -> RetrievalResult:
        """执行混合检索（BM25 + 向量检索 + RRF 融合）

        Args:
            query: 用户原始查询（内部负责 keyword extraction + synonym + encode + HyDE）
            options: 检索选项
            use_hybrid: 是否使用混合检索（否则仅 BM25）
        """
        timing = {}

        if options is None:
            options = RetrievalOptions()

        # 预处理：构建 query_string
        t0 = time.time()
        query_string = self._build_bm25_query(query, options)
        timing["bm25_dsl"] = time.time() - t0

        # 预处理：构建 query_vector（含 HyDE）
        t0 = time.time()
        query_vector = self._build_query_vector(query, options)
        timing["embedding"] = time.time() - t0

        # 检索
        if use_hybrid:
            t0 = time.time()
            chunks = self._hybrid_search(query_string, query_vector, options)
            timing["retrieve"] = time.time() - t0
        else:
            t0 = time.time()
            chunks = self._execute_bm25(query_string, options, options.top_k)
            timing["retrieve"] = time.time() - t0

        # 过滤低质量 chunk（HTML 碎片等）
        before_filter = len(chunks)
        chunks = [c for c in chunks if not is_low_quality_content(c.content)]
        if len(chunks) < before_filter:
            logger.info(
                f"质量过滤: 移除 {before_filter - len(chunks)} 个低质量 chunk "
                f"({before_filter} → {len(chunks)})"
            )

        # 过滤低分
        if options.min_score > 0:
            chunks = [c for c in chunks if c.score >= options.min_score]

        # 截断
        chunks = chunks[: options.top_k]

        timing_str = ", ".join([f"{k}={v:.0f}s" for k, v in timing.items()])
        logger.info(
            f"检索完成: query_string='{query_string[:50]}...', "
            f"results={len(chunks)}, "
            f"hybrid={use_hybrid}, "
            f"({timing_str})"
        )

        return RetrievalResult(
            query=query_string,
            total=len(chunks),
            chunks=chunks,
            timing=timing,
        )

    # ============================================================
    # 混合检索
    # ============================================================

    def _hybrid_search(
        self,
        query: str,
        options: RetrievalOptions,
        return_intermediates: bool = False,
    ):
        """混合检索（BM25 + 向量 + RRF 融合）

        Args:
            query: 用户原始查询（内部负责 keyword extraction + synonym + encode + HyDE）
            return_intermediates: 若 True，返回 (hybrid_chunks, bm25_chunks, vector_chunks, hybrid_timing) 四元组；
                                  若 False（默认），仅返回 hybrid_chunks（向后兼容）
        """
        # 预处理
        query_string = self._build_bm25_query(query, options)
        query_vector = self._build_query_vector(query, options)

        # 候选扩大到 top_k*2，避免截断
        candidate_k = options.top_k * 2

        # 1. BM25
        t_bm25 = time.time()
        bm25_results = self._execute_bm25(query_string, options, candidate_k)
        t_bm25_total = time.time() - t_bm25

        # 2. 向量
        t_vec = time.time()
        vector_results = []
        if query_vector is not None:
            vector_options = options.model_copy(update={"top_k": candidate_k})
            vector_results = self._execute_vector_search(
                query_vector, vector_options, candidate_k
            )
        t_vec_total = time.time() - t_vec

        # 3. RRF 融合
        t_rrf = time.time()
        vector_w = options.vector_weight if options.vector_weight is not None else 0.95
        bm25_w = 1.0 - vector_w

        rrf_results = self.rrf.fuse(
            bm25_results=[(c, i) for i, c in enumerate(bm25_results)],
            vector_results=[(c, i) for i, c in enumerate(vector_results)],
            bm25_weight=bm25_w,
            vector_weight=vector_w,
        )
        t_rrf_total = time.time() - t_rrf

        final_chunks = []
        for chunk, fused_score in rrf_results:
            chunk.score = fused_score
            final_chunks.append(chunk)

        # 细粒度 timing
        hybrid_timing = {
            "bm25": round(t_bm25_total, 3),
            "vector": round(t_vec_total, 3),
            "rrf": round(t_rrf_total, 3),
        }

        if return_intermediates:
            return final_chunks, bm25_results, vector_results, hybrid_timing
        return final_chunks

    def _bm25_search(
        self,
        query: str,
        options: RetrievalOptions,
    ) -> List[RetrievedChunk]:
        """仅 BM25 检索

        Args:
            query: 用户原始查询（内部负责 keyword extraction + synonym + DSL 构建）
        """
        query_string = self._build_bm25_query(query, options)
        return self._execute_bm25(query_string, options, options.top_k)

    def _vector_search(
        self,
        query: str,
        options: RetrievalOptions,
    ) -> List[RetrievedChunk]:
        """仅向量检索

        Args:
            query: 用户原始查询（内部负责 encode + HyDE）
        """
        query_vector = self._build_query_vector(query, options)
        if query_vector is None:
            return []
        return self._execute_vector_search(query_vector, options, options.top_k)

    # ============================================================
    # ES 查询执行
    # ============================================================

    def _build_filter_conditions(self, options: RetrievalOptions) -> list:
        """构建 filter 条件（BM25 和 kNN 共用）"""
        filters = [
            {"term": {"is_latest": True}},
            {"terms": {"chunk_type": ["child", "summary"]}},
        ]
        if options.doc_ids:
            filters.append({"terms": {"doc_id": options.doc_ids}})
        if options.dataset_ids:
            filters.append({"terms": {"dataset_id": options.dataset_ids}})
        return filters

    def _execute_bm25(
        self,
        query_string: str,
        options: RetrievalOptions,
        top_k: int,
    ) -> List[RetrievedChunk]:
        """执行 BM25 检索（query_string 格式）

        Args:
            query_string: 最终的 Lucene query_string（由调用方通过 _build_bm25_query 构建）
        """
        filter_conditions = self._build_filter_conditions(options)

        dsl: Dict[str, Any] = {
            "query": {
                "bool": {
                    "filter": filter_conditions,
                    "must": [
                        {
                            "query_string": {
                                "query": query_string,
                                "fields": ["doc_title^2", "content"],
                                "type": "best_fields",
                                "minimum_should_match": 1,
                            }
                        }
                    ],
                }
            },
            "size": top_k,
        }

        try:
            response = self.es.search(index=self.chunks_index, body=dsl)
        except Exception as e:
            logger.error(f"BM25 检索失败: {e}")
            return []

        return self._parse_results_with_score(response)

    def _execute_vector_search(
        self,
        query_vector: np.ndarray,
        options: RetrievalOptions,
        top_k: int,
    ) -> List[RetrievedChunk]:
        """执行向量检索"""
        vector_query = self._build_vector_query(query_vector, options, top_k)

        dsl = {**vector_query, "size": top_k}

        try:
            response = self.es.search(index=self.chunks_index, body=dsl)
        except Exception as e:
            if "different number of dimensions" in str(e):
                logger.warning(f"向量维度不匹配，向量检索降级: {e}")
            else:
                logger.error(f"向量检索失败: {e}")
            return []

        return self._parse_results_with_score(response)

    # ============================================================
    # BM25 查询构建
    # ============================================================

    def _build_bm25_query(
        self,
        query: str,
        _options: RetrievalOptions,
    ) -> str:
        """
        构建 BM25 query_string（参照 RAGFlow FulltextQueryer.question 中文分支）

        返回 Lucene query_string 语法字符串，用于 ES query_string 查询。

        结构（5 层）：
        1. 每个原词 + 同义词 OR：(原词 OR (syn1 syn2 ...)^0.2)^w
        2. 相邻原词 bigram 近邻匹配："left right"~2^max(w1,w2)*2
        3. 整句加权：(所有原词拼接)^5
        4. 整句同义词 OR：("syn1" OR "syn2" OR ...)^0.7
        5. 原始 query 兜底

        权重归一化为 sum=1.0（参照 RAGFlow term_weight.weights 归一化逻辑）
        """
        # 提取关键词（带权重）
        raw_keywords = get_keyword_extractor().extract(query)  # list[tuple[str, float]]

        # 同义词扩展：别名替换为标准名 + 同义词降权扩展
        synonym_lookup = get_synonym_lookup()

        # 对每个原词：标准化 + 查同义词
        norm_keywords = []
        word_synonyms: Dict[str, list[str]] = {}
        existing_norm = set()

        for kw, weight in raw_keywords:
            std = synonym_lookup.normalize(kw)
            if std not in existing_norm:
                norm_keywords.append((std, weight))
                existing_norm.add(std)
                syns = synonym_lookup.get_synonyms(std)
                word_synonyms[std] = syns

        # ── 权重归一化（sum=1.0） ──
        total_weight = sum(w for _, w in norm_keywords)
        if total_weight > 0:
            norm_keywords = [(kw, w / total_weight) for kw, w in norm_keywords]

        tms_parts = []

        # ── 1. 每个原词 + 同义词 OR ──
        for kw, w in norm_keywords:
            tk = _sub_special_char(kw)
            # 含空格的词加引号
            if tk.find(" ") > 0:
                tk = f'"{tk}"'

            syns = word_synonyms.get(kw, [])
            if syns:
                # 同义词转义 + 含空格加引号
                syn_escaped = []
                for s in syns:
                    s = _sub_special_char(s)
                    if s.find(" ") > 0:
                        s = f'"{s}"'
                    syn_escaped.append(s)
                tk = f"({tk} OR ({' '.join(syn_escaped)})^0.2)"

            if tk.strip():
                tms_parts.append(f"({tk})^{w:.4f}")

        # ── 2. bigram 近邻匹配（~2 允许中间插入 2 个词） ──
        for i in range(1, len(norm_keywords)):
            left, left_w = norm_keywords[i - 1]
            right, right_w = norm_keywords[i]
            if not left.strip() or not right.strip():
                continue
            bigram = f'"{_sub_special_char(left)} {_sub_special_char(right)}"~2'
            tms_parts.append(f"{bigram}^{max(left_w, right_w) * 2:.4f}")

        # ── 3. 整句加权 + 4. 整句同义词 OR ──
        if len(norm_keywords) > 1:
            tms_core = f"({' '.join(tms_parts)})^5"

            # ── 4. 整句同义词 OR ──
            all_syns = []
            for kw, _ in norm_keywords:
                syns = word_synonyms.get(kw, [])
                all_syns.extend(syns)

            if all_syns:
                syns_str = " OR ".join([f'"{_sub_special_char(s)}"' for s in all_syns])
                tms_core = f"{tms_core} OR ({syns_str})^0.7"

            query_string = tms_core
        else:
            query_string = " ".join(tms_parts)

        # ── 5. 原始 query 兜底 ──
        escaped_query = _sub_special_char(query)
        query_string = f"{query_string} OR {escaped_query}"

        return query_string

    # ============================================================
    # 向量查询构建
    # ============================================================

    def _build_vector_query(
        self,
        query_vector: np.ndarray,
        options: RetrievalOptions,
        top_k: int,
    ) -> Dict[str, Any]:
        """
        构建向量 kNN 查询

        filter: is_latest=True, chunk_type=child, doc_ids, dataset_ids
        """
        filter_conditions = self._build_filter_conditions(options)

        num_candidates = max(top_k * 8, 200)

        return {
            "knn": {
                "field": "embedding_vector",
                "query_vector": query_vector.tolist(),
                "k": top_k,
                "num_candidates": num_candidates,
                "filter": {"bool": {"filter": filter_conditions}},
                "similarity": 0.0,
            }
        }

    # ============================================================
    # 结果解析
    # ============================================================

    def _parse_results_with_score(
        self, response: Dict[str, Any]
    ) -> List[RetrievedChunk]:
        """解析 ES 结果并保留原始得分"""
        hits = response.get("hits", {}).get("hits", [])

        chunks = []
        for hit in hits:
            source = hit.get("_source", {})

            chunk = RetrievedChunk(
                chunk_id=source.get("chunk_id", ""),
                doc_id=source.get("doc_id", ""),
                content=source.get("content", ""),
                score=hit.get("_score", 0.0),
                # 文档级字段
                doc_title=source.get("doc_title"),
                dataset_id=source.get("dataset_id"),
                # chunk 级字段
                chunk_type=source.get("chunk_type"),
                doc_hash=source.get("doc_hash"),
                # 父子导航
                parent_id=source.get("parent_id"),
            )
            chunks.append(chunk)

        return chunks


# ── 全局实例 ──────────────────────────────────────────

_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """获取检索服务单例"""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
