"""
评估逻辑 — 加载标注数据，计算各阶段指标
"""

from typing import Any, Dict, List, Optional, Set

from eval.metrics import (
    FilterContributionResult,
    RetrievalMetrics,
    aggregate_metrics,
    compute_retrieval_metrics,
    precision_at_k,
    recall_at_k,
    mrr,
)


# ============================================================
# 辅助
# ============================================================


def _extract_ids(results: List[dict]) -> List[str]:
    """从 results list 中提取 chunk_id"""
    return [r["chunk_id"] for r in results if "chunk_id" in r]


def _get_relevant_ids(query_data: dict) -> Set[str]:
    """获取标注的 relevant chunk ids"""
    ids = query_data.get("relevant_chunk_ids", [])
    return set(ids)


# ============================================================
# Rewrite 评估（仅展示，不评分）
# ============================================================


def evaluate_rewrite(labeled_data: dict) -> List[dict]:
    """收集 rewrite 输出供展示，不做自动评分"""
    rewrites = []
    for q in labeled_data.get("queries", []):
        rw = q.get("rewrite")
        if rw:
            rewrites.append(
                {
                    "query_id": q["id"],
                    "query": q["query"],
                    "question_type": q.get("question_type", ""),
                    "original_query": rw.get("original_query", ""),
                    "rewritten_query": rw.get("rewritten_query", ""),
                    "intent_type": rw.get("intent_type", ""),
                    "target_entities": rw.get("target_entities", []),
                    "keywords": rw.get("keywords", []),
                    "strategy": rw.get("strategy", "direct"),
                }
            )
    return rewrites


# ============================================================
# 检索阶段评估
# ============================================================


def evaluate_retrieval_stage(
    labeled_data: dict,
    stage_name: str,
) -> tuple[list[RetrievalMetrics], dict[str, dict[str, float]]]:
    """
    评估单个检索阶段

    Returns:
        (per_query_metrics, aggregated)
    """
    per_query: List[RetrievalMetrics] = []

    for q in labeled_data.get("queries", []):
        relevant = _get_relevant_ids(q)
        if not relevant:
            continue

        stage_data = q.get(stage_name)
        if not stage_data:
            continue

        retrieved = _extract_ids(stage_data.get("results", []))
        metrics = compute_retrieval_metrics(
            query_id=q["id"],
            stage=stage_name,
            question_type=q.get("question_type", "其他复杂"),
            retrieved_ids=retrieved,
            relevant_ids=relevant,
        )
        per_query.append(metrics)

    type_map = {q["id"]: q.get("question_type", "其他复杂") for q in labeled_data.get("queries", [])}
    aggregated = aggregate_metrics(per_query, type_map)
    return per_query, aggregated


# ============================================================
# Rerank 增量评估
# ============================================================


def evaluate_rerank_delta(
    labeled_data: dict,
) -> dict:
    """
    对比 hybrid_rrf → reranked 的指标变化
    """
    deltas = []

    for q in labeled_data.get("queries", []):
        relevant = _get_relevant_ids(q)
        if not relevant:
            continue

        hybrid = q.get("hybrid_rrf", {})
        reranked = q.get("reranked", {})

        hyb_ids = _extract_ids(hybrid.get("results", []))
        rerank_ids = _extract_ids(reranked.get("results", []))

        if not hyb_ids or not rerank_ids:
            continue

        qtype = q.get("question_type", "其他复杂")
        before = compute_retrieval_metrics(q["id"], "hybrid_rrf", qtype, hyb_ids, relevant)
        after = compute_retrieval_metrics(q["id"], "reranked", qtype, rerank_ids, relevant)

        deltas.append(
            {
                "query_id": q["id"],
                "question_type": qtype,
                "query": q["query"],
                "before": before.to_dict(),
                "after": after.to_dict(),
                "delta_p10": round(after.precision_at_10 - before.precision_at_10, 4),
                "delta_r10": round(after.recall_at_10 - before.recall_at_10, 4),
                "delta_mrr": round(after.mrr_value - before.mrr_value, 4),
            }
        )

    # 汇总平均 delta
    n = len(deltas)
    if n == 0:
        return {"per_query": [], "summary": {}}

    summary = {
        "avg_delta_p10": round(sum(d["delta_p10"] for d in deltas) / n, 4),
        "avg_delta_r10": round(sum(d["delta_r10"] for d in deltas) / n, 4),
        "avg_delta_mrr": round(sum(d["delta_mrr"] for d in deltas) / n, 4),
        "improved": sum(1 for d in deltas if d["delta_mrr"] > 0),
        "degraded": sum(1 for d in deltas if d["delta_mrr"] < 0),
        "unchanged": sum(1 for d in deltas if d["delta_mrr"] == 0),
    }

    return {"per_query": deltas, "summary": summary}


# ============================================================
# Generation 评估
# ============================================================


def evaluate_generation(labeled_data: dict) -> dict:
    """按问题类型分组统计 generation 指标"""
    groups: Dict[str, List[dict]] = {}
    total_usage = {"prompt": 0, "completion": 0, "total": 0}

    for q in labeled_data.get("queries", []):
        gen = q.get("generation")
        if not gen:
            continue

        qtype = q.get("question_type", "其他复杂")
        groups.setdefault(qtype, []).append(
            {
                "query_id": q["id"],
                "query": q["query"],
                "answer_length": len(gen.get("answer", "")),
                "token_usage": gen.get("token_usage", {}),
            }
        )

        usage = gen.get("token_usage", {})
        total_usage["prompt"] += usage.get("prompt", 0)
        total_usage["completion"] += usage.get("completion", 0)
        total_usage["total"] += usage.get("total", 0)

    # 按类型统计
    type_stats = {}
    for qtype, items in groups.items():
        n = len(items)
        type_stats[qtype] = {
            "count": n,
            "avg_answer_length": round(sum(it["answer_length"] for it in items) / n, 0),
            "avg_total_tokens": round(sum(it["token_usage"].get("total", 0) for it in items) / n, 0),
        }

    return {
        "total_queries": sum(len(v) for v in groups.values()),
        "total_usage": total_usage,
        "by_type": type_stats,
    }


# ============================================================
# Filter 消融评估
# ============================================================


def evaluate_filter_breakdown(labeled_data: dict) -> dict:
    """各 filter 变体的检索指标对比"""
    variant_names = [
        "full_filter",
        "no_chunk_types",
        "no_keywords",
        "no_target_models",
        "no_filter",
    ]

    results: Dict[str, Dict[str, List[FilterContributionResult]]] = {}

    for q in labeled_data.get("queries", []):
        relevant = _get_relevant_ids(q)
        if not relevant:
            continue

        fb = q.get("filter_breakdown")
        if not fb:
            continue

        qid = q["id"]
        qtype = q.get("question_type", "其他复杂")

        for vname in variant_names:
            variant_data = fb.get(vname, {})
            for stage in ("bm25", "vector"):
                stage_results = variant_data.get(stage, [])
                if not stage_results:
                    continue

                retrieved = _extract_ids(stage_results)
                metrics = compute_retrieval_metrics(qid, stage, qtype, retrieved, relevant)

                key = f"{vname}"
                results.setdefault(key, {}).setdefault(stage, []).append(
                    FilterContributionResult(
                        query_id=qid,
                        variant_name=vname,
                        stage=stage,
                        precision_at_10=metrics.precision_at_10,
                        recall_at_10=metrics.recall_at_10,
                        mrr_value=metrics.mrr_value,
                        hit_at_10=metrics.hit_at_10,
                    )
                )

    # 聚合
    aggregated: Dict[str, Dict[str, dict]] = {}
    for vname, stages in results.items():
        aggregated[vname] = {}
        for stage, items in stages.items():
            n = len(items)
            aggregated[vname][stage] = {
                "P@10": round(sum(i.precision_at_10 for i in items) / n, 4),
                "R@10": round(sum(i.recall_at_10 for i in items) / n, 4),
                "MRR": round(sum(i.mrr_value for i in items) / n, 4),
                "Hit@10": round(sum(i.hit_at_10 for i in items) / n, 4),
            }

    return {"per_variant_per_stage": aggregated}
