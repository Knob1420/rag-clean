"""
输出格式化 — CLI 表格 + JSON 文件输出
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.metrics import RetrievalMetrics


# ============================================================
# CLI Reporter（tabulate 表格）
# ============================================================


class CLIReporter:
    """终端表格输出"""

    # 指标显示顺序和列名
    METRIC_ORDER = ["P@5", "P@10", "R@5", "R@10", "MRR", "Hit@5", "Hit@10"]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def print_stage_report(
        self,
        stage_name: str,
        aggregated: Dict[str, Dict[str, float]],
    ):
        """打印单阶段汇总表格"""
        from tabulate import tabulate

        overall = aggregated.get("overall", {})
        if not overall:
            print(f"\n--- Stage: {stage_name} --- (无数据)")
            return

        # 收集出现过的 type
        type_names = [k for k in aggregated if k != "overall"]

        headers = ["Metric", "Overall"] + type_names
        rows = []
        for metric in self.METRIC_ORDER:
            row = [metric]
            row.append(self._fmt(overall.get(metric)))
            for t in type_names:
                row.append(self._fmt(aggregated[t].get(metric)))
            rows.append(row)

        print(f"\n--- Stage: {stage_name} ---")
        print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".4f"))

    def print_rerank_delta(self, delta_data: dict):
        """打印 rerank 增量对比"""
        from tabulate import tabulate

        summary = delta_data.get("summary", {})
        if not summary:
            print("\n--- Rerank Delta --- (无数据)")
            return

        print(f"\n--- Rerank Delta (hybrid → reranked) ---")
        print(f"  Avg ΔP@10: {self._fmt(summary.get('avg_delta_p10', 0))}")
        print(f"  Avg ΔR@10: {self._fmt(summary.get('avg_delta_r10', 0))}")
        print(f"  Avg ΔMRR:  {self._fmt(summary.get('avg_delta_mrr', 0))}")
        print(
            f"  Improved: {summary.get('improved', 0)}, "
            f"Degraded: {summary.get('degraded', 0)}, "
            f"Unchanged: {summary.get('unchanged', 0)}"
        )

        if self.verbose and delta_data.get("per_query"):
            headers = ["Query", "Type", "ΔP@10", "ΔR@10", "ΔMRR"]
            rows = []
            for d in delta_data["per_query"]:
                rows.append(
                    [
                        d["query"][:30],
                        d["question_type"],
                        self._fmt(d["delta_p10"]),
                        self._fmt(d["delta_r10"]),
                        self._fmt(d["delta_mrr"]),
                    ]
                )
            print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".4f"))

    def print_rewrite_summary(self, rewrites: List[dict]):
        """打印 rewrite 摘要"""
        if not rewrites:
            print("\n--- Query Rewrite --- (无数据)")
            return

        print(f"\n--- Query Rewrite ({len(rewrites)} queries) ---")
        for rw in rewrites:
            print(f"  [{rw['query_id']}] {rw['query']}")
            print(f"    → {rw['rewritten_query']}")
            print(
                f"    strategy={rw.get('strategy', 'direct')}, intent={rw['intent_type']}, "
                f"entities={rw['target_entities']}, keywords={rw['keywords']}"
            )

    def print_generation_summary(self, gen_data: dict):
        """打印 generation 摘要"""
        if not gen_data or not gen_data.get("total_queries"):
            print("\n--- Generation --- (无数据)")
            return

        print(f"\n--- Generation ({gen_data['total_queries']} queries) ---")
        usage = gen_data.get("total_usage", {})
        print(
            f"  Total tokens: {usage.get('total', 0)} "
            f"(prompt: {usage.get('prompt', 0)}, completion: {usage.get('completion', 0)})"
        )

        by_type = gen_data.get("by_type", {})
        if by_type:
            from tabulate import tabulate

            headers = ["Type", "Count", "Avg Answer Len", "Avg Tokens"]
            rows = []
            for qtype, stats in by_type.items():
                rows.append(
                    [
                        qtype,
                        stats["count"],
                        int(stats["avg_answer_length"]),
                        int(stats["avg_total_tokens"]),
                    ]
                )
            print(tabulate(rows, headers=headers, tablefmt="github"))

    def print_filter_breakdown(self, fb_data: dict):
        """打印 filter 消融结果"""
        from tabulate import tabulate

        per_variant = fb_data.get("per_variant_per_stage", {})
        if not per_variant:
            print("\n--- Filter Breakdown --- (无数据)")
            return

        print(f"\n--- Filter Breakdown ---")
        for vname, stages in per_variant.items():
            for stage, metrics in stages.items():
                print(f"\n  Variant: {vname} | Stage: {stage}")
                headers = ["Metric", "Value"]
                rows = [[k, self._fmt(v)] for k, v in metrics.items()]
                print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".4f"))

    def print_per_query_detail(
        self,
        per_query: List[RetrievalMetrics],
        stage_name: str,
    ):
        """打印逐 query 详细指标"""
        from tabulate import tabulate

        if not per_query:
            return

        headers = ["QID", "Type", "P@5", "P@10", "R@5", "R@10", "MRR", "Hit@5", "Hit@10"]
        rows = []
        for m in per_query:
            rows.append(
                [
                    m.query_id,
                    m.question_type,
                    f"{m.precision_at_5:.4f}",
                    f"{m.precision_at_10:.4f}",
                    f"{m.recall_at_5:.4f}",
                    f"{m.recall_at_10:.4f}",
                    f"{m.mrr_value:.4f}",
                    m.hit_at_5,
                    m.hit_at_10,
                ]
            )

        print(f"\n--- {stage_name}: Per-Query Detail ---")
        print(tabulate(rows, headers=headers, tablefmt="github"))

    @staticmethod
    def _fmt(val: Any) -> str:
        if isinstance(val, float):
            return f"{val:.4f}"
        if isinstance(val, bool):
            return "Y" if val else "N"
        return str(val)


# ============================================================
# JSON Reporter
# ============================================================


class JSONReporter:
    """JSON 文件输出"""

    def __init__(self, report_dir: str = "eval/results"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save_report(self, report_data: dict) -> str:
        """保存完整评估报告为 JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.report_dir / f"eval_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        return str(filename)

    @staticmethod
    def build_report(
        rewrites: List[dict],
        stage_results: Dict[str, tuple],
        rerank_delta: dict,
        generation_data: dict,
        filter_breakdown: dict,
    ) -> dict:
        """组装完整报告"""
        report: Dict[str, Any] = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            "rewrite": rewrites,
            "stages": {},
            "rerank_delta": rerank_delta,
            "generation": generation_data,
            "filter_breakdown": filter_breakdown,
        }

        for stage_name, (per_query, aggregated) in stage_results.items():
            report["stages"][stage_name] = {
                "per_query": [m.to_dict() for m in per_query],
                "aggregated": aggregated,
            }

        return report
