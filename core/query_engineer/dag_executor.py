"""
DAG Executor — 执行 QueryIR 生成的执行计划

支持的操作符：retrieve / extract / filter / foreach / aggregate
维护执行状态（ExecutionContext），按依赖顺序执行各步骤
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from core.query_engineer.operators import (
    ExecutionContext,
    OperatorRegistry,
    RetrieveOperator,
    FilterOperator,
    ExtractOperator,
    ForeachOperator,
    AggregateOperator,
)
from core.query_engineer.query_ir import QueryIR

# ── Plan / Step ─────────────────────────────────────────────────────────────

@dataclass
class Step:
    """DAG 中的单个步骤"""
    id: str
    op: str
    params: Dict[str, Any] = field(default_factory=dict)
    input_from: List[str] = field(default_factory=list)

@dataclass
class Plan:
    """执行计划 — 一组有序的步骤"""
    steps: List[Step] = field(default_factory=list)

# ── DAGExecutor ─────────────────────────────────────────────────────────────

class DAGExecutor:
    """
    DAG 执行器

    使用方法：
        executor = DAGExecutor(retrieval_service)
        result = executor.execute(plan)
    """

    def __init__(self, retrieval_service=None):
        self.retrieval_service = retrieval_service
        self.operators = OperatorRegistry()
        self._register_operators()

    def _register_operators(self):
        if self.retrieval_service:
            self.operators.register("retrieve", RetrieveOperator(self.retrieval_service))
        self.operators.register("filter", FilterOperator())
        self.operators.register("extract", ExtractOperator())
        self.operators.register("aggregate", AggregateOperator())
        self.operators.register("foreach", ForeachOperator(self.operators))

    def execute(self, plan: Plan, initial_ctx: Optional[ExecutionContext] = None) -> ExecutionContext:
        ctx = initial_ctx or ExecutionContext()

        for step in plan.steps:
            logger.info(f"[DAGExecutor] executing step: {step.id} ({step.op})")
            ctx = self._execute_step(step, ctx, plan)

        return ctx

    def _execute_step(self, step: Step, ctx: ExecutionContext, plan: Plan) -> ExecutionContext:
        operator = self.operators.get(step.op)

        if operator is None:
            logger.error(f"[DAGExecutor] unknown operator: {step.op}")
            ctx.errors.append(f"unknown operator: {step.op}")
            return ctx

        params = self._resolve_params(step.params, ctx)

        try:
            ctx = operator.execute(ctx, params)
            logger.info(f"[DAGExecutor] step {step.id} done, errors={len(ctx.errors)}")
        except Exception as e:
            logger.error(f"[DAGExecutor] step {step.id} failed: {e}")
            ctx.errors.append(str(e))

        return ctx

    def _resolve_params(self, params: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str) and "{entities}" in v:
                resolved[k] = v.replace("{entities}", ",".join(ctx.entities))
            else:
                resolved[k] = v
        return resolved

    def build_plan_from_query_ir(self, ir: QueryIR, top_k: int = 20) -> Plan:
        """
        从 QueryIR 构建执行计划
        """
        steps: List[Step] = []
        step_counter = 0

        def add_step(op: str, params: Dict[str, Any], input_from: List[str] = None) -> str:
            nonlocal step_counter
            step_id = f"s{step_counter}"
            step_counter += 1
            steps.append(Step(
                id=step_id,
                op=op,
                params=params,
                input_from=input_from or [],
            ))
            return step_id

        # 1. 如果需要 expand（多实体扩展）
        if ir.needs_expand() and ir.expand_map:
            for parent, children in ir.expand_map.items():
                for child in children:
                    add_step("retrieve", {
                        "query": child,
                        "top_k": top_k,
                        "step_id": f"retrieve_{child}",
                    })
        else:
            # 2. 直接 retrieve
            target_query = ir.target
            if ir.entities:
                target_query = " ".join(ir.entities)
            add_step("retrieve", {
                "query": target_query,
                "top_k": top_k,
                "step_id": "main_retrieve",
            })

        # 3. 如果需要 filter
        if ir.needs_filter() and ir.constraints:
            last_step_id = steps[-1].id if steps else None
            add_step("filter", {
                "constraints": [
                    {
                        "type": c.type,
                        "field": c.field,
                        "op": c.op,
                        "value": c.value,
                        "unit": c.unit,
                    }
                    for c in ir.constraints
                ],
            }, input_from=[last_step_id] if last_step_id else [])

        # 4. 如果需要 extract（提取结构化字段）
        if ir.required_fields:
            last_step_id = steps[-1].id if steps else None
            add_step("extract", {
                "fields": ir.required_fields,
            }, input_from=[last_step_id] if last_step_id else [])

        # 5. 如果需要 aggregate
        if ir.needs_aggregate() and ir.dimensions:
            last_step_id = steps[-1].id if steps else None
            add_step("aggregate", {
                "dimensions": ir.dimensions,
            }, input_from=[last_step_id] if last_step_id else [])

        # 6. 如果需要 foreach（多实体分别检索）
        if ir.needs_foreach() and ir.entities:
            last_step_id = steps[-1].id if steps else None
            add_step("foreach", {
                "entities": ir.entities,
                "sub_operator": "retrieve",
                "sub_params": {
                    "query": "{entity}",
                    "top_k": top_k,
                },
            }, input_from=[last_step_id] if last_step_id else [])

        logger.info(f"[DAGExecutor] built plan with {len(steps)} steps")
        return Plan(steps=steps)

    def execute_query_ir(self, ir: QueryIR, top_k: int = 20) -> ExecutionContext:
        """
        便捷方法：直接接收 QueryIR，返回执行结果
        """
        plan = self.build_plan_from_query_ir(ir, top_k)
        initial_ctx = ExecutionContext()
        initial_ctx.entities = ir.entities
        return self.execute(plan, initial_ctx)