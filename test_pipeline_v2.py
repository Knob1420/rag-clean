#!/usr/bin/env python3
"""
Pipeline V2 测试 - 基于 QueryRewriteRetrievalPipeline

使用新的 V2 pipeline (QueryRewriteServiceV2 + RetrievalService)

使用方式:
  python test_pipeline_v2.py                                    # 快速测试
  python test_pipeline_v2.py --mode full                       # 全量测试
  python test_pipeline_v2.py --queries "G1重量" "NX3算力"        # 自定义查询
  python test_pipeline_v2.py --use-generation                    # 启用 LLM 生成
  python test_pipeline_v2.py --no-rewrite                        # 跳过 query rewrite
  python test_pipeline_v2.py --query-rewrite-only               # 只运行意图识别+query transform
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.retrieve.retrieval_models import RetrievedChunk

sys.path.insert(0, str(Path(__file__).parent))

from core.generation.generation import get_generation_service
from core.pipeline import QueryRewriteRetrievalPipeline


# ============================================================
# 测试配置
# ============================================================

TOP_K = 25
USE_RERANK = True
RERANK_TOP_K = 12
USE_GENERATION = True

# 快速测试 query 列表
QUICK_TEST_QUERIES = [
    "三体计算星座的定义",
    "介绍一下三体计算星座，帮我翻译成英文",
    "三体计算星座建设规划",
    "地卫二项目代号",
    "具身智能卫星模型介绍，一段话介绍一下它的能力，不超过50字",
    "宇宙X射线偏振探测器原理",
    "简单介绍橄榄叶计划",
    "3kg以内的星载智算机，可以帮我推荐一个吗？",
    "我们首发之前的智算机的在轨验证，有哪几次啊？",
    "3618号新型胞元的来历写一下",
    "介绍一下3D打印卫星",
    "nx1 gpu板的数据盘落盘速度是多少",
    "上海的合作单位有哪些？",
    "长三角地区合作单位及合作形式、预期成果",
    "我们和蓝箭鸿擎的合作有什么，写一段话即可",
    "我们与国星宇航的合作有什么",
    "之江实验室发射了多少颗卫星",
    "智加G3支持什么接口",
    "天基分布式操作系统的特点",
    "NX系列和G系列有什么区别 ",
    "G1、G2、G3分别适合什么场景？",
    "智加全系列产品的尺寸和重量对比",
    "推荐一款轻量级星载智算机",
    "推荐一款2kg以内，算力大于250TFlops的智算机",
]


# ============================================================
# 核心测试逻辑
# ============================================================


def run_single_query(
    query: str,
    pipeline: QueryRewriteRetrievalPipeline,
    top_k: int = TOP_K,
    use_rewrite: bool = True,
    use_rerank: bool = USE_RERANK,
    rerank_top_k: int = RERANK_TOP_K,
    use_generation: bool = USE_GENERATION,
    query_rewrite_only: bool = False,
    dataset_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    使用 Pipeline V2 执行单条查询

    Args:
        query_rewrite_only: 如果为 True，只运行意图识别+query transform，不执行检索
        dataset_ids: 按数据集 ID 列表筛选
    """
    try:
        generation_svc = get_generation_service() if use_generation else None

        # 调用 pipeline
        result = pipeline.run(
            query=query,
            top_k=top_k,
            use_rewrite=use_rewrite,
            use_rerank=use_rerank,
            rerank_top_k=rerank_top_k,
            query_rewrite_only=query_rewrite_only,
            dataset_ids=dataset_ids,
        )

        response: Dict[str, Any] = {
            "query": query,
            "pipeline_result": result,
            "success": True,
            "error": None,
        }

        # 只运行 query rewrite 阶段（不执行检索和生成）
        if query_rewrite_only:
            return response

        # LLM 生成 — 多子问句合并逻辑
        if use_generation and generation_svc and result.chunks:
            understanding = result.understanding_result
            sub_queries = understanding.sub_queries if understanding else []

            if len(sub_queries) == 1:
                # 单子问句：直接生成
                sq = sub_queries[0]
                sq_chunks = result.per_sub_question_chunks.get(sq.query, result.chunks)
                sq_spec = result.per_sub_question_spec_context.get(sq.query, "")
                sq_constraints = result.per_sub_question_generation_constraints.get(sq.query, [])
                answer, usage = generation_svc.generate(
                    query=request.query,
                    chunks=sq_chunks,
                    query_intent=sq.intent,
                    spec_context=sq_spec,
                    generation_constraints=sq_constraints,
                )
                response["generation_answer"] = answer
                response["generation_usage"] = usage
            else:
                # 多子问句：逐条生成 → 合并 → 整合生成
                merged_answers = []
                all_chunks = []
                total_usage = None

                for sq in sub_queries:
                    sq_chunks = result.per_sub_question_chunks.get(sq.query, [])
                    if not sq_chunks:
                        continue
                    all_chunks.extend(sq_chunks)
                    sq_spec = result.per_sub_question_spec_context.get(sq.query, "")
                    sq_constraints = result.per_sub_question_generation_constraints.get(sq.query, [])
                    sub_answer, sub_usage = generation_svc.generate(
                        query=sq.query,
                        chunks=sq_chunks,
                        query_intent=sq.intent,
                        spec_context=sq_spec,
                        generation_constraints=sq_constraints,
                    )
                    merged_answers.append(f"【{sq.query}】\n{sub_answer}")
                    total_usage = sub_usage

                if merged_answers:
                    # 整合生成
                    integrated = "\n\n---\n\n".join(merged_answers)
                    system_prompt, user_prompt = generation_svc._build_integration_prompt(
                        original_query=query,
                        merged_answers=integrated,
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    import openai
                    from config import settings
                    if settings.deepseek_api_key:
                        client = openai.OpenAI(
                            api_key=settings.deepseek_api_key,
                            base_url=settings.deepseek_base_url,
                        )
                        resp = client.chat.completions.create(
                            model=settings.deepseek_model,
                            messages=messages,
                            temperature=0.3,
                            max_tokens=2000,
                        )
                        answer = resp.choices[0].message.content
                        usage = total_usage
                    else:
                        answer = integrated

                    response["generation_answer"] = answer
                    response["generation_usage"] = usage
                else:
                    response["generation_answer"] = None
                    response["generation_usage"] = None
        else:
            response["generation_answer"] = None
            response["generation_usage"] = None

        return response

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {
            "query": query,
            "pipeline_result": None,
            "success": False,
            "error": str(e),
        }


# ============================================================
# 输出格式化
# ============================================================


def print_single_result(result: Dict[str, Any], index: int):
    """打印单条结果"""
    query = result["query"]
    error = result.get("error")

    print(f"\n{'='*60}")
    print(f"[{index}] 查询: {query}")
    print(f"{'='*60}")

    if error:
        print(f"  失败: {error}")
        return

    pipeline_result = result.get("pipeline_result")
    if not pipeline_result:
        return

    # Query Rewrite 结果
    print(f"\n  [Query Rewrite V2]")
    print(f"  intent: {pipeline_result.intent}")
    print(f"  rewritten_queries: {pipeline_result.rewritten_queries}")

    # Timing
    timing = pipeline_result.timing
    if timing:
        timing_str = ", ".join([f"{k}={v*1000:.0f}ms" for k, v in timing.items()])
        print(f"\n  [Timing] {timing_str}")

    # 检索结果
    chunks = pipeline_result.chunks
    print(f"\n  [检索结果] 共召回 {len(chunks)} 条")
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(
                f"\n  [{i+1}] score={chunk.score:.4f} | "
                f"chunk_type={chunk.chunk_type or 'N/A'} | "
                f"doc_title={chunk.doc_title or 'N/A'}"
            )
            content_preview = (chunk.content or "").replace("\n", " ")
            print(f"      {content_preview}...")

    # Generation 结果
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

    print(f"\n\n{'='*60}")
    print(f"{stage_name} 测试报告")
    print(f"{'='*60}")
    print(f"\n  成功率: {success}/{total} ({success/total*100:.0f}%)")

    # 统计 rewritten_queries 数量分布
    multi_query_count = 0
    total_rewrite_time = 0
    total_retrieve_time = 0

    for r in results:
        pr = r.get("pipeline_result")
        if pr and pr.rewritten_queries:
            if len(pr.rewritten_queries) > 1:
                multi_query_count += 1
            total_rewrite_time += pr.timing.get("rewrite", 0) * 1000
            total_retrieve_time += pr.timing.get("retrieve", 0) * 1000

    if results:
        avg_rewrite = total_rewrite_time / success if success else 0
        avg_retrieve = total_retrieve_time / success if success else 0
        print(f"\n  平均耗时:")
        print(f"    rewrite: {avg_rewrite:.0f}ms")
        print(f"    retrieve: {avg_retrieve:.0f}ms")

    print(f"\n  多子查询拆分: {multi_query_count}/{total}")

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
    output_path = f"test_reports/pipeline_v2_{stage_name}_{timestamp}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "stage": stage_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "top_k": TOP_K,
            "use_rerank": USE_RERANK,
            "rerank_top_k": RERANK_TOP_K,
            "use_generation": USE_GENERATION,
        },
        "results": [],
    }

    for r in results:
        item: Dict[str, Any] = {
            "query": r["query"],
            "success": r["success"],
            "error": r.get("error"),
        }

        pr = r.get("pipeline_result")
        if pr:
            item["rewrite"] = {
                "intent": pr.intent,
                "rewritten_queries": pr.rewritten_queries,
            }
            item["timing_ms"] = {k: int(v * 1000) for k, v in pr.timing.items()}

            # 每个 sub_question 的检索结果（合并去重 + 独立 rerank 后）
            item["per_sub_question_chunks"] = {}
            for sq_query, chunks in pr.per_sub_question_chunks.items():
                item["per_sub_question_chunks"][sq_query] = [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "doc_title": c.doc_title,
                        "score": round(c.score, 4),
                        "chunk_type": c.chunk_type,
                        "content_preview": (c.content or ""),
                    }
                    for c in chunks
                ]

            # 每个 sub_question 的结构化上下文
            item["per_sub_question_spec_context"] = pr.per_sub_question_spec_context

            # rq → intent 映射
            item["rq_intent_map"] = pr.rq_intent_map

            # 合并后的检索结果（hybrid RRF + rerank）
            item["retrieval"] = {
                "total": pr.total,
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "doc_title": c.doc_title,
                        "score": round(c.score, 4),
                        "chunk_type": c.chunk_type,
                        "content_preview": (c.content or ""),
                    }
                    for c in pr.chunks
                ],
            }

        if r.get("generation_answer"):
            item["generation"] = {"answer": r["generation_answer"]}
            if r.get("generation_usage"):
                item["generation"]["total_tokens"] = r["generation_usage"].total_tokens

        report["results"].append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  报告已保存: {output_path}")


def save_csv(results: List[Dict[str, Any]], stage_name: str):
    """保存 CSV 报告 — 每条路径各自排序，每行一个 chunk"""
    import csv

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"test_reports/pipeline_v2_{stage_name}_{timestamp}.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # CSV 列定义
    fieldnames = [
        "query",
        "sub_question",
        "rank",
        "chunk_id",
        "chunk_content",
        "score",
        "doc_title",
        "chunk_type",
        "answer",
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            query = r["query"]
            answer = r.get("generation_answer") or ""
            pr = r.get("pipeline_result")

            if not pr:
                writer.writerow(
                    {
                        "query": query,
                        "sub_question": "none",
                        "rank": 1,
                        "answer": answer,
                    }
                )
                continue

            # 按每个 sub_question 输出其检索结果
            for sq_query, chunks in pr.per_sub_question_chunks.items():
                for rank, chunk in enumerate(chunks, 1):
                    writer.writerow(
                        {
                            "query": query,
                            "sub_question": sq_query,
                            "rank": rank,
                            "chunk_id": chunk.chunk_id,
                            "chunk_content": (chunk.content or ""),
                            "score": round(chunk.score, 4),
                            "doc_title": chunk.doc_title or "",
                            "chunk_type": chunk.chunk_type or "",
                            "answer": answer if rank == 1 else "",
                        }
                    )

    print(f"\n  CSV 已保存: {output_path}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline V2 测试")
    parser.add_argument("--queries", nargs="+", help="自定义查询 (mode=custom)")
    parser.add_argument("--no-details", action="store_true", help="不显示详细结果")
    parser.add_argument("--no-report", action="store_true", help="不保存报告")
    parser.add_argument("--use-generation", action="store_true", help="启用 LLM 生成")
    parser.add_argument("--no-rerank", action="store_true", help="不使用 rerank")
    parser.add_argument("--no-rewrite", action="store_true", help="跳过 query rewrite")
    parser.add_argument(
        "--query-rewrite-only",
        action="store_true",
        help="只运行意图识别+query transform，不执行检索",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        default=["产品", "合同", "测试"],
        help="按数据集 ID 筛选（如 products contracts）",
    )

    args = parser.parse_args()

    # 查询集选择
    queries = QUICK_TEST_QUERIES

    use_rewrite = not args.no_rewrite
    stage_name = "rewrite_v2" if use_rewrite else "baseline_v2"

    # 初始化 pipeline
    pipeline = QueryRewriteRetrievalPipeline()

    print(
        f"Pipeline V2 | queries={len(queries)} | top_k={args.top_k} | "
        f"rerank={not args.no_rerank} | rewrite={use_rewrite} | "
        f"generation={args.use_generation} | dataset_ids={args.dataset_ids}"
    )
    print()

    results = []
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q}")
        r = run_single_query(
            query=q,
            pipeline=pipeline,
            top_k=args.top_k,
            use_rewrite=use_rewrite,
            use_rerank=not args.no_rerank,
            use_generation=args.use_generation,
            query_rewrite_only=args.query_rewrite_only,
            dataset_ids=args.dataset_ids,
        )
        results.append(r)
        if not args.no_details:
            print_single_result(r, i)

    print_summary(results, stage_name)

    if not args.no_report:
        save_report(results, stage_name)
        save_csv(results, stage_name)
