"""
查询理解与重写服务

基于 rag-knowledge-base query_rewrite.py，适配 rag-clean：
- ES 聚合用 doc_type（扁平字段）而非 metadata.doc_type.keyword
- 复用 llm.py 的 LLMClient.call() + parse_json_response()
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from config import settings
from core.generation.llm import LLMClient, get_llm_client, parse_json_response
from prompt import QUERY_REWRITE_SYSTEM_PROMPT, build_rewrite_prompt
from store import get_store


class RewrittenQuery(BaseModel):
    """重写后的查询"""

    original_query: str = ""
    rewritten_query: str
    intent_type: str = "other"          # → ES chunk_type should boost + generation intent context
    target_entities: List[str] = Field(default_factory=list)  # → ES entity boost
    keywords: List[str] = Field(default_factory=list)         # → ES keywords boost
    strategy: str = "direct"            # direct | parallel | sequential → 路由决策
    sub_queries: List[dict] = Field(default_factory=list)     # → parallel path 输入

    @staticmethod
    def _validate_strategy(v: str) -> str:
        if v not in ("direct", "parallel", "sequential"):
            return "direct"
        return v

    def model_post_init(self, __context):
        self.strategy = self._validate_strategy(self.strategy)

    @property
    def intent(self) -> str:
        """兼容旧代码：返回 intent_type"""
        return self.intent_type

    @property
    def entities(self) -> Dict[str, str]:
        """兼容旧代码：将 target_entities 转换为字典"""
        if not self.target_entities:
            return {}
        return {"实体": ",".join(self.target_entities)}

    @property
    def target_models(self) -> List[str]:
        """兼容旧代码"""
        return self.target_entities


class QueryRewriteService:
    """查询理解与重写服务"""

    def __init__(self):
        self.llm = get_llm_client()
        self._store = get_store()
        self.chunks_index = settings.es_index_chunks
        self._chunk_types_cache: Optional[List[str]] = None

    @property
    def es(self):
        return self._store.es

    # ============================================================
    # ES chunk_type 聚合
    # ============================================================

    def refresh_type_cache(self):
        """从 ES 聚合获取 chunk 类型，刷新缓存"""
        try:
            agg = self.es.search(
                index=self.chunks_index,
                body={
                    "size": 0,
                    "aggs": {
                        "chunk_types": {"terms": {"field": "chunk_type", "size": 50}},
                    },
                },
            )

            chunk_buckets = agg.get("aggregations", {}).get("chunk_types", {}).get("buckets", [])
            self._chunk_types_cache = [b["key"] for b in chunk_buckets]

            logger.info(
                f"[Query Rewrite] chunk_type 缓存已刷新: {len(chunk_buckets)} 种"
            )

        except Exception as e:
            logger.warning(f"[Query Rewrite] ES 聚合失败，使用默认缓存: {e}")
            self._chunk_types_cache = []

    def _get_chunk_types_list(self) -> str:
        """获取 chunk_type 列表文本（懒加载）"""
        if self._chunk_types_cache is None:
            self.refresh_type_cache()
        if self._chunk_types_cache:
            return ", ".join(self._chunk_types_cache)
        return "intro, spec_data, feature, procedure, faq, profile, other"

    # ============================================================
    # 核心入口
    # ============================================================

    def rewrite(self, query: str) -> RewrittenQuery:
        """重写查询"""
        logger.info(f"[Query Rewrite] 原始查询: {query}")

        prompt = self._build_rewrite_prompt(query)

        try:
            response = self.llm.call(
                [
                    {
                        "role": "system",
                        "content": QUERY_REWRITE_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            result = parse_json_response(response)
            result["original_query"] = query

            # 兼容旧字段名 → 新字段名
            if "target_models" in result and "target_entities" not in result:
                models = result.pop("target_models", [])
                others = result.pop("other_entities", [])
                seen = set()
                merged = []
                for e in models + others:
                    if e and e not in seen:
                        merged.append(e)
                        seen.add(e)
                result["target_entities"] = merged

            rewritten = RewrittenQuery(**result)

            logger.info(f"[Query Rewrite] 完成:")
            logger.info(f"  strategy={rewritten.strategy}, intent={rewritten.intent_type}")
            logger.info(f"  entities={rewritten.target_entities}")
            logger.info(f"  rewritten: {rewritten.rewritten_query}")

            return rewritten

        except Exception as e:
            logger.error(f"[Query Rewrite] 失败: {e}")
            return RewrittenQuery(
                original_query=query,
                rewritten_query=query,
            )

    # ============================================================
    # Prompt 构建
    # ============================================================

    def _build_rewrite_prompt(self, query: str) -> str:
        """构建重写提示词（动态注入 ES chunk_type 列表）"""
        chunk_types_list = self._get_chunk_types_list()
        return build_rewrite_prompt(query, chunk_types_list)


# ── 全局实例 ──────────────────────────────────────────

_query_rewrite_service: Optional[QueryRewriteService] = None


def get_query_rewrite_service() -> QueryRewriteService:
    """获取查询重写服务单例"""
    global _query_rewrite_service
    if _query_rewrite_service is None:
        _query_rewrite_service = QueryRewriteService()
    return _query_rewrite_service
