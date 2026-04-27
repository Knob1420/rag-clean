# core/query_engineer/operators.py
"""
DAG Executor 的操作符定义

每个 Operator 接收 context（执行状态），返回更新后的 context
所有操作均为确定性规则操作，不依赖 LLM
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from loguru import logger

# ── Context ──────────────────────────────────────────────────────────────────

@dataclass
class ExecutionContext:
    """
    DAG 执行状态容器
    """
    chunks: List[Any] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    filtered: List[Any] = field(default_factory=list)
    structured: Dict[str, List[Any]] = field(default_factory=dict)
    aggregated: Any = None
    retrieved: Dict[str, List[Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_input_chunks(self) -> List[Any]:
        return self.filtered if self.filtered else self.chunks

# ── Operator Protocol ────────────────────────────────────────────────────────

class Operator(ABC):
    name: str = ""

    @abstractmethod
    def execute(self, ctx: ExecutionContext, params: Dict[str, Any]) -> ExecutionContext:
        ...

# ── Operator Implementations ─────────────────────────────────────────────────

class RetrieveOperator(Operator):
    """检索操作 — 调用 RetrievalService 获取文档"""
    name = "retrieve"

    def __init__(self, retrieval_service):
        self.retrieval_service = retrieval_service

    def execute(self, ctx: ExecutionContext, params: Dict[str, Any]) -> ExecutionContext:
        from core.retrieve.retrieval import RetrievalOptions

        query = params.get("query", "")
        top_k = params.get("top_k", 20)
        dataset_ids = params.get("dataset_ids")

        logger.info(f"[RetrieveOperator] query='{query}', top_k={top_k}")

        try:
            options = RetrievalOptions(
                top_k=top_k,
                use_rerank=False,
                dataset_ids=dataset_ids,
            )
            chunks = self.retrieval_service._hybrid_search(query, options)
            ctx.chunks = chunks
            ctx.retrieved[params.get("step_id", "retrieve")] = chunks
            logger.info(f"[RetrieveOperator] retrieved {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"[RetrieveOperator] failed: {e}")
            ctx.errors.append(f"retrieve failed: {e}")

        return ctx


class FilterOperator(Operator):
    """过滤操作 — 基于 constraints 过滤 chunks"""
    name = "filter"

    def execute(self, ctx: ExecutionContext, params: Dict[str, Any]) -> ExecutionContext:
        constraints_raw = params.get("constraints", [])
        if not constraints_raw:
            return ctx

        from core.query_engineer.query_ir import Constraint

        constraints: List[Constraint] = []
        for c in constraints_raw:
            if isinstance(c, Constraint):
                constraints.append(c)
            elif isinstance(c, dict):
                constraints.append(Constraint(
                    type=c.get("type", "numeric"),
                    field=c.get("field", ""),
                    op=c.get("op", "="),
                    value=c.get("value", ""),
                    unit=c.get("unit"),
                ))

        if not constraints:
            return ctx

        logger.info(f"[FilterOperator] applying {len(constraints)} constraints")

        input_chunks = ctx.get_input_chunks()
        filtered_chunks = []

        for chunk in input_chunks:
            content_lower = chunk.content.lower()
            all_match = True
            for constraint in constraints:
                matched = self._check_constraint(chunk, constraint, content_lower)
                if not matched:
                    all_match = False
                    break
            if all_match:
                filtered_chunks.append(chunk)

        ctx.filtered = filtered_chunks
        logger.info(f"[FilterOperator] filtered {len(input_chunks)} → {len(filtered_chunks)} chunks")
        return ctx

    def _check_constraint(self, chunk, constraint, content_lower: str) -> bool:
        import re
        field = constraint.field.lower()
        patterns = [
            rf"{field}[：:]\s*([^\n，,]+)",
            rf"{field}\s*=\s*([^\n，,]+)",
            rf"([^\n，,]+)\s*{field}",
        ]
        value_str = None
        for pattern in patterns:
            match = re.search(pattern, content_lower)
            if match:
                value_str = match.group(1).strip()
                break
        if value_str is None:
            return True

        if constraint.type == "numeric":
            try:
                numbers = re.findall(r"[\d.]+", value_str)
                if not numbers:
                    return True
                value = float(numbers[0])
                if constraint.op == ">":
                    return value > float(constraint.value)
                elif constraint.op == "<":
                    return value < float(constraint.value)
                elif constraint.op == ">=":
                    return value >= float(constraint.value)
                elif constraint.op == "<=":
                    return value <= float(constraint.value)
                elif constraint.op == "=":
                    return abs(value - float(constraint.value)) < 0.01
            except (ValueError, TypeError):
                pass
        elif constraint.type in ("region", "category"):
            if isinstance(constraint.value, list):
                return any(v in content_lower for v in constraint.value)
            return str(constraint.value) in content_lower
        return True


class ExtractOperator(Operator):
    """提取操作 — 从 chunks 中提取结构化字段"""
    name = "extract"

    def execute(self, ctx: ExecutionContext, params: Dict[str, Any]) -> ExecutionContext:
        fields = params.get("fields", [])
        if not fields:
            return ctx

        logger.info(f"[ExtractOperator] extracting fields: {fields}")

        input_chunks = ctx.get_input_chunks()
        structured: Dict[str, List[Any]] = {f: [] for f in fields}

        import re
        for chunk in input_chunks:
            content = chunk.content
            for field_name in fields:
                field_lower = field_name.lower()
                pattern = rf"{field_lower}[：:]\s*([^\n，,]+)"
                match = re.search(pattern, content)
                if match:
                    structured[field_name].append(match.group(1).strip())
                else:
                    structured[field_name].append(None)

        ctx.structured = structured
        logger.info(f"[ExtractOperator] extracted: { {k: len(v) for k, v in structured.items()} }")
        return ctx


class ForeachOperator(Operator):
    """遍历操作 — 对每个实体独立执行子操作"""
    name = "foreach"

    def __init__(self, operators):
        self.operators = operators

    def execute(self, ctx: ExecutionContext, params: Dict[str, Any]) -> ExecutionContext:
        entities = params.get("entities", ctx.entities)
        sub_operator_name = params.get("sub_operator", "retrieve")
        sub_params = params.get("sub_params", {})

        logger.info(f"[ForeachOperator] iterating over {len(entities)} entities")

        sub_results: Dict[str, List[Any]] = {}
        for entity in entities:
            logger.info(f"[ForeachOperator] processing entity: {entity}")
            resolved_params = {}
            for k, v in sub_params.items():
                if isinstance(v, str):
                    resolved_params[k] = v.replace("{entity}", entity)
                else:
                    resolved_params[k] = v

            sub_op = self.operators.get(sub_operator_name)
            if sub_op:
                sub_ctx = ExecutionContext()
                sub_ctx.entities = [entity]
                result_ctx = sub_op.execute(sub_ctx, resolved_params)
                sub_results[entity] = result_ctx.chunks

        all_chunks = []
        for entity, chunks in sub_results.items():
            all_chunks.extend(chunks)

        ctx.chunks = all_chunks
        ctx.retrieved["foreach"] = all_chunks
        logger.info(f"[ForeachOperator] total chunks from all entities: {len(all_chunks)}")
        return ctx


class AggregateOperator(Operator):
    """聚合操作 — 按维度分组归纳结果"""
    name = "aggregate"

    def execute(self, ctx: ExecutionContext, params: Dict[str, Any]) -> ExecutionContext:
        dimensions = params.get("dimensions", [])
        if not dimensions:
            return ctx

        logger.info(f"[AggregateOperator] aggregating by dimensions: {dimensions}")

        structured = ctx.structured
        if not structured:
            logger.warning("[AggregateOperator] no structured data to aggregate")
            return ctx

        aggregated: Dict[str, Any] = {}
        for dim in dimensions:
            if dim in structured:
                aggregated[dim] = structured[dim]

        ctx.aggregated = aggregated
        logger.info(f"[AggregateOperator] aggregated result keys: {list(aggregated.keys())}")
        return ctx

# ── Operator Registry ─────────────────────────────────────────────────────

class OperatorRegistry:
    def __init__(self):
        self._operators: Dict[str, Operator] = {}

    def register(self, name: str, operator: Operator):
        self._operators[name] = operator

    def get(self, name: str) -> Optional[Operator]:
        return self._operators.get(name)

    @property
    def operators(self) -> Dict[str, Operator]:
        return self._operators

    def names(self) -> List[str]:
        return list(self._operators.keys())