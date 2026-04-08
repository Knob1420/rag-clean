"""
评估指标 — 纯函数 + 数据模型

不依赖 service，可独立测试。
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set


# ============================================================
# 数据模型
# ============================================================


@dataclass
class QueryItem:
    """单条查询"""

    id: str
    query: str
    question_type: str


@dataclass
class QueryDataset:
    """完整查询集"""

    metadata: Dict[str, str] = field(default_factory=dict)
    queries: List[QueryItem] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "QueryDataset":
        meta = data.get("metadata", {})
        queries = [QueryItem(**q) for q in data.get("queries", [])]
        return cls(metadata=meta, queries=queries)


@dataclass
class RetrievalMetrics:
    """单 query 单阶段的检索指标"""

    query_id: str
    stage: str
    question_type: str
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr_value: float = 0.0
    hit_at_5: bool = False
    hit_at_10: bool = False
    num_retrieved: int = 0
    num_relevant: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FilterContributionResult:
    """单个 filter 变体的贡献分析"""

    query_id: str
    variant_name: str
    stage: str  # "bm25" | "vector"
    precision_at_10: float = 0.0
    recall_at_10: float = 0.0
    mrr_value: float = 0.0
    hit_at_10: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# 核心指标函数
# ============================================================


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """P@K：前 K 个结果中正确的比例"""
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """R@K：前 K 个结果中覆盖了多少 relevant"""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """MRR：第一个正确结果的倒数排名"""
    if not relevant_ids:
        return 0.0
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> bool:
    """Hit@K：前 K 个结果中是否至少有一个 correct"""
    top_k = retrieved_ids[:k]
    return any(cid in relevant_ids for cid in top_k)


def compute_retrieval_metrics(
    query_id: str,
    stage: str,
    question_type: str,
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> RetrievalMetrics:
    """计算单 query 单阶段的全部指标"""
    return RetrievalMetrics(
        query_id=query_id,
        stage=stage,
        question_type=question_type,
        precision_at_5=precision_at_k(retrieved_ids, relevant_ids, 5),
        precision_at_10=precision_at_k(retrieved_ids, relevant_ids, 10),
        recall_at_5=recall_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_10=recall_at_k(retrieved_ids, relevant_ids, 10),
        mrr_value=mrr(retrieved_ids, relevant_ids),
        hit_at_5=hit_at_k(retrieved_ids, relevant_ids, 5),
        hit_at_10=hit_at_k(retrieved_ids, relevant_ids, 10),
        num_retrieved=len(retrieved_ids),
        num_relevant=len(relevant_ids),
    )


def aggregate_metrics(
    per_query_metrics: List[RetrievalMetrics],
    type_map: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """
    聚合指标：总体 + 按 question_type 分组

    Returns:
        {
            "overall": {"P@5": 0.42, ...},
            "事实": {"P@5": 0.55, ...},
            ...
        }
    """
    if not per_query_metrics:
        return {}

    metric_keys = [
        "precision_at_5",
        "precision_at_10",
        "recall_at_5",
        "recall_at_10",
        "mrr_value",
        "hit_at_5",
        "hit_at_10",
    ]
    display_names = {
        "precision_at_5": "P@5",
        "precision_at_10": "P@10",
        "recall_at_5": "R@5",
        "recall_at_10": "R@10",
        "mrr_value": "MRR",
        "hit_at_5": "Hit@5",
        "hit_at_10": "Hit@10",
    }

    # 按 type 分桶
    buckets: Dict[str, List[RetrievalMetrics]] = {"overall": list(per_query_metrics)}
    for m in per_query_metrics:
        qt = m.question_type
        buckets.setdefault(qt, []).append(m)

    result: Dict[str, Dict[str, float]] = {}
    for group_name, metrics_list in buckets.items():
        n = len(metrics_list)
        group: Dict[str, float] = {}
        for key in metric_keys:
            val = sum(getattr(m, key) for m in metrics_list) / n
            # hit 指标用 bool → float 均值就是命中率
            group[display_names[key]] = round(val, 4)
        result[group_name] = group

    return result
