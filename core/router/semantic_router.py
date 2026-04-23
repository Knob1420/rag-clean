"""
语义路由器 — embedding 余弦相似度意图分类

启动时用 bge-m3 编码锚点样本取均值作为原型向量，
查询时编码 query 与各原型计算余弦相似度，选最高分类。

confidence >= 0.7 → 高置信度，使用意图路由
confidence <  0.7 → 降级到默认 pipeline
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from core.router.models import RoutingResult
from core.router.intent_prototypes import INTENT_PROTOTYPES
from core.client.embedder import encode, encode_batch

# 置信度阈值
CONFIDENCE_THRESHOLD = 0.7


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticRouter:
    """语义路由器"""

    def __init__(self):
        self._prototypes: Dict[str, np.ndarray] = {}
        self._initialized = False

    # ============================================================
    # 初始化（懒加载原型向量）
    # ============================================================

    def initialize(self) -> None:
        """编码所有锚点样本，计算原型向量"""
        if self._initialized:
            return

        logger.info("[SemanticRouter] 初始化意图原型向量...")

        for intent, samples in INTENT_PROTOTYPES.items():
            # 批量编码锚点样本
            vectors = encode_batch(samples)
            valid_vecs = [v for v in vectors if v is not None]

            if valid_vecs:
                # 取均值作为原型向量
                self._prototypes[intent] = np.mean(valid_vecs, axis=0)
                logger.info(
                    f"  {intent}: {len(valid_vecs)}/{len(samples)} 样本编码成功"
                )
            else:
                logger.warning(f"  {intent}: 编码全部失败")

        self._initialized = True
        logger.info(
            f"[SemanticRouter] 初始化完成: "
            f"{len(self._prototypes)}/{len(INTENT_PROTOTYPES)} 个意图就绪"
        )

    # ============================================================
    # 路由入口
    # ============================================================

    def route(self, query: str) -> RoutingResult:
        """
        路由查询到意图

        Args:
            query: 用户查询文本

        Returns:
            RoutingResult with intent and confidence
        """
        if not self._initialized:
            self.initialize()

        # 意图分类
        intent, confidence = self._classify(query)

        result = RoutingResult(
            intent=intent,
            confidence=confidence,
            original_query=query,
        )

        logger.info(
            f"[SemanticRouter] 路由结果: "
            f"intent={intent}, conf={confidence:.3f}, "
            f"is_high_confidence={result.is_high_confidence}"
        )

        return result

    # ============================================================
    # 意图分类
    # ============================================================

    def _classify(self, query: str) -> Tuple[str, float]:
        """
        编码 query，与各原型计算余弦相似度

        Returns:
            (best_intent, confidence)
        """
        query_vec = encode(query)
        if query_vec is None:
            logger.warning("[SemanticRouter] query 编码失败，降级 simple_lookup")
            return "simple_lookup", 0.0

        scores = {}
        for intent, proto_vec in self._prototypes.items():
            scores[intent] = _cosine_similarity(query_vec, proto_vec)

        if not scores:
            return "simple_lookup", 0.0

        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]

        # Debug: 打印 top-3
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top3 = ", ".join(f"{k}={v:.3f}" for k, v in sorted_scores[:3])
        logger.debug(f"[SemanticRouter] scores: {top3}")

        return best_intent, confidence

    def classify_intent(self, query: str) -> Tuple[str, float]:
        """
        轻量意图分类，只算 embedding 相似度，无 LLM 调用。

        Returns:
            (best_intent, confidence)
        """
        if not self._initialized:
            self.initialize()
        return self._classify(query)

    @property
    def is_ready(self) -> bool:
        return self._initialized and len(self._prototypes) > 0


# ── 全局实例 ──────────────────────────────────────────

_semantic_router: Optional[SemanticRouter] = None


def get_semantic_router() -> SemanticRouter:
    global _semantic_router
    if _semantic_router is None:
        _semantic_router = SemanticRouter()
    return _semantic_router
