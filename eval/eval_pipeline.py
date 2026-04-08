"""
RAG 评估管线 CLI 主入口

Usage:
    # Step 1: 导出
    python -m eval.eval_pipeline export \\
        --queries eval/datasets/queries.json \\
        --output eval/exports/run_001.json

    # Step 2: 标注（手动编辑 JSON 中的 relevant_chunk_ids）

    # Step 3: 评估
    python -m eval.eval_pipeline evaluate \\
        --labeled eval/exports/run_001_labeled.json \\
        --report eval/results/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval.eval_pipeline",
        description="RAG 评估管线 — 导出检索结果 & 评估各阶段指标",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── export ────────────────────────────────────────────
    exp = subparsers.add_parser("export", help="导出检索结果为 JSON")
    exp.add_argument(
        "--queries",
        required=True,
        help="查询集 JSON 路径 (eval/datasets/queries.json)",
    )
    exp.add_argument(
        "--output",
        required=True,
        help="导出输出路径 (eval/exports/run_001.json)",
    )
    exp.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="每阶段返回的结果数 (default: 20)",
    )
    exp.add_argument(
        "--no-rewrite",
        action="store_true",
        help="跳过 Query Rewrite 阶段",
    )
    exp.add_argument(
        "--stages",
        default="all",
        help="要导出的阶段，逗号分隔 (all|bm25,vector,hybrid,rerank,generation,filter_ablation)。默认 all 不含 generation",
    )

    # ── evaluate ──────────────────────────────────────────
    evl = subparsers.add_parser("evaluate", help="评估标注后的数据")
    evl.add_argument(
        "--labeled",
        required=True,
        help="标注后的 JSON 路径",
    )
    evl.add_argument(
        "--report",
        default="eval/results",
        help="报告输出目录 (default: eval/results)",
    )
    evl.add_argument(
        "--verbose",
        action="store_true",
        help="输出逐 query 详细指标",
    )

    return parser.parse_args()


# ============================================================
# export 子命令
# ============================================================


def cmd_export(args: argparse.Namespace):
    from eval.exporters import PipelineExporter

    # 加载查询集
    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"错误: 查询集文件不存在: {queries_path}", file=sys.stderr)
        sys.exit(1)

    with open(queries_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 解析 stages
    stages_str = args.stages.strip()
    if stages_str == "all":
        stages: Set[str] = set()
    else:
        stages = {s.strip() for s in stages_str.split(",") if s.strip()}

    # 导出
    exporter = PipelineExporter(top_k=args.top_k)
    output_path = exporter.export_dataset(
        dataset=dataset,
        output_path=args.output,
        stages=stages if stages else None,
        no_rewrite=args.no_rewrite,
    )

    print(f"\n导出完成: {output_path}")
    print("下一步：在 JSON 中为每条 query 填写 relevant_chunk_ids，然后运行 evaluate。")


# ============================================================
# evaluate 子命令
# ============================================================


def cmd_evaluate(args: argparse.Namespace):
    from eval.evaluators import (
        evaluate_filter_breakdown,
        evaluate_generation,
        evaluate_rerank_delta,
        evaluate_rewrite,
        evaluate_retrieval_stage,
    )
    from eval.reporters import CLIReporter, JSONReporter

    labeled_path = Path(args.labeled)
    if not labeled_path.exists():
        print(f"错误: 标注文件不存在: {labeled_path}", file=sys.stderr)
        sys.exit(1)

    with open(labeled_path, "r", encoding="utf-8") as f:
        labeled_data = json.load(f)

    queries = labeled_data.get("queries", [])
    labeled_count = sum(1 for q in queries if q.get("relevant_chunk_ids"))
    print(f"加载标注数据: {len(queries)} queries, {labeled_count} 条已标注")

    if labeled_count == 0:
        print("警告: 没有任何 query 标注了 relevant_chunk_ids，指标将全部为 0。")

    reporter = CLIReporter(verbose=args.verbose)
    stage_results = {}

    # ── Rewrite ──────────────────────────────────────────
    rewrites = evaluate_rewrite(labeled_data)
    reporter.print_rewrite_summary(rewrites)

    # ── 各检索阶段 ──────────────────────────────────────
    eval_stages = ["bm25", "vector", "hybrid_rrf", "reranked"]
    for stage in eval_stages:
        # 检查数据中是否存在该阶段
        has_stage = any(q.get(stage) for q in queries)
        if not has_stage:
            continue

        per_query, aggregated = evaluate_retrieval_stage(labeled_data, stage)
        reporter.print_stage_report(stage, aggregated)
        stage_results[stage] = (per_query, aggregated)

        if args.verbose:
            reporter.print_per_query_detail(per_query, stage)

    # ── Rerank Delta ─────────────────────────────────────
    rerank_delta = evaluate_rerank_delta(labeled_data)
    reporter.print_rerank_delta(rerank_delta)

    # ── Generation ───────────────────────────────────────
    gen_data = evaluate_generation(labeled_data)
    reporter.print_generation_summary(gen_data)

    # ── Filter Breakdown ─────────────────────────────────
    fb_data = evaluate_filter_breakdown(labeled_data)
    reporter.print_filter_breakdown(fb_data)

    # ── JSON 报告 ────────────────────────────────────────
    json_reporter = JSONReporter(report_dir=args.report)
    report = JSONReporter.build_report(
        rewrites=rewrites,
        stage_results=stage_results,
        rerank_delta=rerank_delta,
        generation_data=gen_data,
        filter_breakdown=fb_data,
    )
    report_path = json_reporter.save_report(report)
    print(f"\nJSON 报告已保存: {report_path}")


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    if args.command == "export":
        cmd_export(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        print("请指定子命令: export 或 evaluate", file=sys.stderr)
        print("使用 --help 查看帮助")
        sys.exit(1)


if __name__ == "__main__":
    main()
