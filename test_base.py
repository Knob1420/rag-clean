#!/usr/bin/env python3
"""
检索测试 - 支持多阶段对照

  Stage 1 (--no-rewrite): raw query → hybrid_search → rerank → generation
  Stage 2 (默认):         raw query → rewrite → 路由(simple→hybrid / complex→react) → generation

使用方式:
  python test_base.py                          # Stage 2 快速测试 (6条)
  python test_base.py --no-rewrite             # Stage 1 基线（无 rewrite）
  python test_base.py --mode full              # 全量测试 (35条)
  python test_base.py --no-generation          # 只测试检索，不调用 LLM
  python test_base.py --no-rerank              # 不使用 rerank
  python test_base.py --top-k 20               # 指定召回数量
  python test_base.py --mode custom --queries "G1多重" "G2算力"
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from core.generation.generation import get_generation_service
from core.query_engineer.query_rewrite import RewrittenQuery, get_query_rewrite_service
from core.query_engineer.react_reasoning import get_react_reasoning_service
from core.retrieve.retrieval import get_retrieval_service
from core.retrieve.retrieval_models import (
    HighlightOptions,
    RetrievedChunk,
    RetrievalOptions,
    RetrievalResult,
)


# ============================================================
# 测试配置
# ============================================================

TOP_K = 25
USE_RERANK = True
RERANK_TOP_K = 12
USE_GENERATION = False

# 快速测试 query 列表（覆盖各类问题）
QUICK_TEST_QUERIES = [
    "智加G1的重量是多少？",
    "三体计算星座项目是做什么的",
    "智加G3和智加NX3有什么区别？",
    "推荐哪款国产智算机？",
    "天基分布式操作系统的特点",
    "智算机采用的GPU是进口的还是国产的？国产的主要是哪几家的，什么架构的？",
    "3kg以内的星载智算机，可以帮我推荐一个吗？",
    "NX系列和G系列有什么区别",
    "智加G2和智加G3的算力差多少？",
    "智加全系列产品的尺寸和重量对比",
    "之江实验室首发星座一共发射了多少颗卫星",
    "智加G1、G2、G3三代产品分别适合什么场景？",
]


# ============================================================
# 辅助函数
# ============================================================


def _chunk_to_dict(c: RetrievedChunk, rank: int = 0) -> Dict[str, Any]:
    """将 RetrievedChunk 序列化为可 JSON 化的字典"""
    return {
        "rank": rank,
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "score": round(c.score, 4),
        "chunk_type": c.chunk_type,
        "section_title": c.section_title,
        "doc_type": c.doc_type,
        "keywords": c.keywords,
        "context_summary": c.context_summary,
        "content_preview": c.content[:200] if c.content else "",
    }


# ============================================================
# 核心测试逻辑
# ============================================================


def run_single_query(
    query: str,
    use_rewrite: bool = True,
    top_k: int = TOP_K,
    use_rerank: bool = USE_RERANK,
    rerank_top_k: int = RERANK_TOP_K,
    use_generation: bool = USE_GENERATION,
) -> Dict[str, Any]:
    """
    统一检索入口

    use_rewrite=False → Stage 1: raw query 直接检索（基线）
    use_rewrite=True  → Stage 2: rewrite → 路由检索(simple→hybrid / complex→react)
    """
    try:
        retrieval_svc = get_retrieval_service()
        rewrite_svc = get_query_rewrite_service()
        generation_svc = get_generation_service() if use_generation else None

        rewritten: Optional[RewrittenQuery] = None
        route_method: Optional[str] = None  # "hybrid" or "routed"
        retrieval_result: Optional[RetrievalResult] = None

        highlight = HighlightOptions()

        if use_rewrite:
            # Stage 2: Query Rewrite
            rewritten = rewrite_svc.rewrite(query)

            # 解析 intent_type: 支持逗号分隔多值
            intent_types = None
            if rewritten.intent_type and rewritten.intent_type != "other":
                intent_types = [
                    t.strip() for t in rewritten.intent_type.split(",") if t.strip()
                ]

            options = RetrievalOptions(
                top_k=top_k,
                target_models=(
                    rewritten.target_entities if rewritten.target_entities else None
                ),
                keywords=rewritten.keywords if rewritten.keywords else None,
                chunk_types=intent_types,
                use_rerank=use_rerank,
                rerank_top_k=rerank_top_k,
            )

            search_query = rewritten.rewritten_query

            # 三路分流: direct / parallel / sequential
            strategy = getattr(rewritten, "strategy", "direct")

            if strategy == "sequential":
                # 多步推理 → ReAct
                react_svc = get_react_reasoning_service()
                retrieval_result = react_svc.reason(
                    original_query=query,
                    rewritten=rewritten,
                    options=options,
                )
                route_method = "react"
            elif strategy == "parallel" and rewritten.sub_queries:
                # 独立子查询 → search_routed 并行检索
                retrieval_result = retrieval_svc.search_routed(
                    rewritten_query=rewritten.rewritten_query,
                    sub_queries=rewritten.sub_queries,
                    options=options,
                    highlight=highlight,
                )
                route_method = "parallel"
            else:
                # 直接检索
                retrieval_result = retrieval_svc.search(
                    query=search_query,
                    options=options,
                    highlight=highlight,
                    use_hybrid=True,
                )
                route_method = "hybrid"
        else:
            # Stage 1: 基线
            options = RetrievalOptions(
                top_k=top_k,
                use_rerank=use_rerank,
                rerank_top_k=rerank_top_k,
            )
            search_query = query

            retrieval_result = retrieval_svc.search(
                query=search_query,
                options=options,
                highlight=highlight,
                use_hybrid=True,
            )
            route_method = "hybrid"

        response: Dict[str, Any] = {
            "query": query,
            "rewritten": rewritten,
            "route_method": route_method,
            "retrieval_result": retrieval_result,
            "success": True,
            "error": None,
        }

        # LLM 生成
        if (
            use_generation
            and generation_svc
            and retrieval_result
            and retrieval_result.chunks
        ):
            answer, usage = generation_svc.generate(
                query=query,
                chunks=retrieval_result.chunks,
                chat_history=None,
                query_intent=rewritten.intent_type if rewritten else None,
                query_entities=rewritten.entities if rewritten else None,
            )
            response["generation_answer"] = answer
            response["generation_usage"] = usage
        else:
            response["generation_answer"] = None
            response["generation_usage"] = None

        return response

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {
            "query": query,
            "rewritten": None,
            "route_method": None,
            "retrieval_result": None,
            "generation_answer": None,
            "generation_usage": None,
            "success": False,
            "error": str(e),
        }


# ============================================================
# 输出格式化
# ============================================================


def print_single_result(
    result: Dict[str, Any],
    index: int,
    show_rewrite: bool = True,
):
    """打印单条结果"""
    query = result["query"]
    rewritten = result.get("rewritten")
    retrieval_result = result.get("retrieval_result")
    error = result.get("error")

    print(f"\n{'='*60}")
    print(f"[{index}] 查询: {query}")
    print(f"{'='*60}")

    if error:
        print(f"  失败: {error}")
        return

    # Query Rewrite 结果
    if show_rewrite and rewritten:
        print(f"\n  [Query Rewrite]")
        print(f"    策略: {rewritten.strategy}")
        print(f"    意图: {rewritten.intent_type}")
        print(f"    实体: {rewritten.target_entities}")
        print(f"    关键词: {rewritten.keywords}")
        print(f"    重写: {rewritten.rewritten_query}")
        if rewritten.sub_queries:
            for sq in rewritten.sub_queries:
                print(f"    子查询: {sq}")

    # 路由方式
    route = result.get("route_method", "N/A")
    if route != "N/A":
        route_name = {
            "hybrid": "普通混合检索",
            "react": "ReAct 多跳推理",
            "routed": "多子查询路由",
        }
        print(f"\n  [路由] {route_name.get(route, route)}")

    # ReAct 推理步骤详情（从 timing 中提取）
    if retrieval_result:
        timing = retrieval_result.timing
        hop_timings = {k: v for k, v in timing.items() if k.startswith("hop_")}
        if hop_timings:
            print(f"\n  [ReAct 推理步骤]")
            for hop, dur in sorted(hop_timings.items()):
                print(f"    {hop}: {dur*1000:.0f}ms")

    # 最终检索结果
    if retrieval_result:
        chunks = retrieval_result.chunks
        timing = retrieval_result.timing
        timing_str = ", ".join([f"{k}={v*1000:.0f}ms" for k, v in timing.items()])
        print(f"\n  [检索结果] 召回 {len(chunks)} 条 | 耗时 {timing_str}")

        for i, chunk in enumerate(chunks[:5]):
            print(
                f"\n  [{i+1}] score={chunk.score:.4f} | "
                f"chunk_type={chunk.chunk_type or 'N/A'} | "
                f"doc_type={chunk.doc_type or 'N/A'}"
            )
            print(f"      keywords={chunk.keywords[:5] if chunk.keywords else []}")
            content_preview = chunk.content.replace("\n", " ")[:150]
            print(f"      {content_preview}...")

    answer = result.get("generation_answer")
    if answer:
        print(f"\n  [LLM 回答]")
        for line in answer.split("\n"):
            print(f"      {line}")
        usage = result.get("generation_usage")
        if usage:
            print(f"\n  [Token] total={usage.total_tokens}")


def print_summary(results: List[Dict[str, Any]], stage_name: str):
    """打印总结"""
    total = len(results)
    success = sum(1 for r in results if r.get("success"))

    retrieval_times = []
    chunk_counts = []
    for r in results:
        if r.get("retrieval_result"):
            timing = r["retrieval_result"].timing
            t = sum(timing.values()) * 1000
            retrieval_times.append(int(t))
            chunk_counts.append(len(r["retrieval_result"].chunks))

    print(f"\n\n{'='*60}")
    print(f"{stage_name} 测试报告")
    print(f"{'='*60}")
    print(f"\n  成功率: {success}/{total} ({success/total*100:.0f}%)")

    if retrieval_times:
        print(f"\n  检索耗时:")
        print(f"    平均: {sum(retrieval_times)/len(retrieval_times):.0f}ms")
        print(f"    最快: {min(retrieval_times)}ms")
        print(f"    最慢: {max(retrieval_times)}ms")

    if chunk_counts:
        print(f"\n  召回数量:")
        print(f"    平均: {sum(chunk_counts)/len(chunk_counts):.1f}")
        print(f"    最多: {max(chunk_counts)}")
        print(f"    最少: {min(chunk_counts)}")

    # 意图分布
    rewrite_results = [r for r in results if r.get("rewritten")]
    if rewrite_results:
        intent_dist: Dict[str, int] = {}
        for r in rewrite_results:
            intent = r["rewritten"].intent_type
            intent_dist[intent] = intent_dist.get(intent, 0) + 1
        print(f"\n  意图分布:")
        for intent, count in sorted(intent_dist.items(), key=lambda x: -x[1]):
            print(f"    {intent}: {count}")

        strategy_dist: Dict[str, int] = {}
        for r in rewrite_results:
            strategy = r["rewritten"].strategy
            strategy_dist[strategy] = strategy_dist.get(strategy, 0) + 1
        print(f"\n  策略分布:")
        for strategy, count in sorted(strategy_dist.items(), key=lambda x: -x[1]):
            print(f"    {strategy}: {count}")

    # 路由分布
    route_dist: Dict[str, int] = {}
    for r in results:
        route = r.get("route_method")
        if route:
            route_dist[route] = route_dist.get(route, 0) + 1
    if route_dist:
        route_name = {
            "hybrid": "普通混合检索",
            "react": "ReAct 多跳推理",
            "routed": "多子查询路由",
        }
        print(f"\n  路由分布:")
        for route, count in sorted(route_dist.items(), key=lambda x: -x[1]):
            print(f"    {route_name.get(route, route)}: {count}")

    # 生成统计
    gen_count = sum(1 for r in results if r.get("generation_answer"))
    if gen_count:
        tokens = [
            r["generation_usage"].total_tokens
            for r in results
            if r.get("generation_usage")
        ]
        print(f"\n  LLM 生成:")
        print(f"    生成数: {gen_count}/{total}")
        if tokens:
            print(f"    平均 Token: {sum(tokens)/len(tokens):.0f}")
            print(f"    总 Token: {sum(tokens)}")


def save_report(results: List[Dict[str, Any]], stage_name: str):
    """保存 JSON 报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"test_reports/{stage_name}_{timestamp}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "stage": stage_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "top_k": TOP_K,
            "use_rerank": USE_RERANK,
            "rerank_top_k": RERANK_TOP_K,
            "use_generation": USE_GENERATION,
            "use_rewrite": "rewrite" in stage_name,
        },
        "results": [],
    }

    for r in results:
        item: Dict[str, Any] = {
            "query": r["query"],
            "success": r["success"],
            "error": r.get("error"),
        }

        if r.get("rewritten"):
            rw = r["rewritten"]
            item["rewrite"] = {
                "strategy": rw.strategy,
                "intent": rw.intent_type,
                "entities": rw.target_entities,
                "keywords": rw.keywords,
                "rewritten_query": rw.rewritten_query,
            }
            if r.get("route_method"):
                item["rewrite"]["route"] = r["route_method"]
            if rw.sub_queries:
                item["rewrite"]["sub_queries"] = rw.sub_queries

        if r.get("retrieval_result"):
            ret = r["retrieval_result"]
            item["retrieval"] = {
                "timing_ms": {k: int(v * 1000) for k, v in ret.timing.items()},
                "total": ret.total,
                "chunks": [_chunk_to_dict(c, i + 1) for i, c in enumerate(ret.chunks)],
            }

        if r.get("generation_answer"):
            item["generation"] = {"answer": r["generation_answer"]}
            if r.get("generation_usage"):
                item["generation"]["total_tokens"] = r["generation_usage"].total_tokens

        report["results"].append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  报告已保存: {output_path}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    # 加载全量查询
    _queries_path = Path(__file__).parent / "eval" / "datasets" / "queries.json"
    if _queries_path.exists():
        with open(_queries_path, encoding="utf-8") as _f:
            _dataset = json.load(_f)
        FULL_QUERIES = [q["query"] for q in _dataset["queries"]]
    else:
        FULL_QUERIES = QUICK_TEST_QUERIES

    parser = argparse.ArgumentParser(
        description="检索测试 (支持 Stage 1 基线 / Stage 2 rewrite)"
    )
    parser.add_argument("--mode", choices=["quick", "full", "custom"], default="quick")
    parser.add_argument("--queries", nargs="+", help="自定义查询 (mode=custom)")
    parser.add_argument("--no-details", action="store_true", help="不显示详细结果")
    parser.add_argument("--no-report", action="store_true", help="不保存报告")
    parser.add_argument("--no-generation", action="store_true", help="不调用 LLM 生成")
    parser.add_argument("--no-rerank", action="store_true", help="不使用 rerank")
    parser.add_argument(
        "--no-rewrite", action="store_true", help="Stage 1 基线（不使用 query rewrite）"
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)

    args = parser.parse_args()

    # 查询集选择
    if args.mode == "quick":
        queries = QUICK_TEST_QUERIES
    elif args.mode == "full":
        queries = FULL_QUERIES
    else:
        if not args.queries:
            print("--queries is required for mode=custom")
            sys.exit(1)
        queries = args.queries

    use_rewrite = not args.no_rewrite

    # Stage 判定
    if not use_rewrite:
        stage_name = "baseline"
        stage_label = "1 (baseline)"
    else:
        stage_name = "rewrite"
        stage_label = "2 (rewrite)"

    # 运行
    print(
        f"Stage {stage_label} "
        f"| queries={len(queries)} "
        f"| top_k={args.top_k} "
        f"| rerank={not args.no_rerank} "
        f"| generation={not args.no_generation}"
    )
    print()

    results = []
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q}")
        r = run_single_query(
            query=q,
            use_rewrite=use_rewrite,
            top_k=args.top_k,
            use_rerank=not args.no_rerank,
            use_generation=not args.no_generation,
        )
        results.append(r)
        if not args.no_details:
            print_single_result(r, i, show_rewrite=use_rewrite)

    print_summary(results, stage_name)

    if not args.no_report:
        save_report(results, stage_name)
