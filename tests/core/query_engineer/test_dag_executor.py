from unittest.mock import MagicMock
from core.query_engineer.dag_executor import DAGExecutor, Step, Plan
from core.query_engineer.query_ir import QueryIR, Constraint
from core.query_engineer.operators import ExecutionContext

def test_plan_creation():
    plan = Plan(steps=[
        Step(id="s0", op="retrieve", params={"query": "智算机"}, input_from=[]),
        Step(id="s1", op="filter", params={}, input_from=["s0"]),
    ])
    assert len(plan.steps) == 2
    assert plan.steps[1].input_from == ["s0"]

def test_dag_executor_unknown_operator():
    """DAGExecutor should handle unknown operators gracefully"""
    executor = DAGExecutor(retrieval_service=None)
    # No operators registered since retrieval_service is None

    # Create a plan with an unknown operator
    plan = Plan(steps=[
        Step(id="s0", op="unknown_op", params={}, input_from=[]),
    ])

    ctx = executor.execute(plan)
    assert len(ctx.errors) > 0
    assert "unknown operator" in ctx.errors[0]

def test_build_plan_from_query_ir_basic():
    """DAGExecutor should build correct plan for basic lookup"""
    executor = DAGExecutor(retrieval_service=None)

    ir = QueryIR(
        original_query="G1重量是多少",
        intent="lookup",
        target="G1",
        constraints=[],
        operations=["retrieve"],
        expand_map={},
        need_split=False,
        need_aggregate=False,
        dimensions=[],
        entities=["G1"],
        required_fields=[],
    )

    plan = executor.build_plan_from_query_ir(ir, top_k=20)

    assert len(plan.steps) >= 1
    assert plan.steps[0].op == "retrieve"
    assert plan.steps[0].params["query"] == "G1"

def test_build_plan_from_query_ir_with_constraints():
    """DAGExecutor should add filter step when constraints present"""
    executor = DAGExecutor(retrieval_service=None)

    ir = QueryIR(
        original_query="2kg以内的智算机",
        intent="lookup",
        target="智算机",
        constraints=[
            Constraint(type="numeric", field="weight", op="<=", value=3, unit="kg"),
        ],
        operations=["filter"],
        expand_map={},
        need_split=False,
        need_aggregate=False,
        dimensions=[],
        entities=["智算机"],
        required_fields=[],
    )

    plan = executor.build_plan_from_query_ir(ir, top_k=20)

    ops = [s.op for s in plan.steps]
    assert "retrieve" in ops
    assert "filter" in ops

def test_build_plan_from_query_ir_with_dimensions():
    """DAGExecutor should add aggregate step for analysis queries"""
    executor = DAGExecutor(retrieval_service=None)

    ir = QueryIR(
        original_query="按合作形式分析合作单位",
        intent="analysis",
        target="合作单位",
        constraints=[],
        operations=["aggregate"],
        expand_map={},
        need_split=False,
        need_aggregate=True,
        dimensions=["合作形式"],
        entities=[],
        required_fields=["合作形式"],
    )

    plan = executor.build_plan_from_query_ir(ir, top_k=20)

    ops = [s.op for s in plan.steps]
    assert "retrieve" in ops
    assert "extract" in ops
    assert "aggregate" in ops

def test_execute_query_ir_basic():
    """execute_query_ir should work with basic QueryIR"""
    # Mock retrieval service
    mock_retrieval = MagicMock()
    mock_chunks = [MagicMock(), MagicMock()]
    mock_retrieval._hybrid_search.return_value = mock_chunks

    executor = DAGExecutor(retrieval_service=mock_retrieval)

    ir = QueryIR(
        original_query="G1重量",
        intent="lookup",
        target="G1",
        constraints=[],
        operations=[],
        expand_map={},
        need_split=False,
        need_aggregate=False,
        dimensions=[],
        entities=["G1"],
        required_fields=[],
    )

    ctx = executor.execute_query_ir(ir, top_k=5)

    # Should have called hybrid_search
    mock_retrieval._hybrid_search.assert_called()
    assert ctx.chunks == mock_chunks

def test_resolve_params():
    """DAGExecutor should resolve {entities} placeholder in params"""
    executor = DAGExecutor()

    params = {"query": "find {entities}", "top_k": 20}
    ctx = ExecutionContext()
    ctx.entities = ["智算机", "激光通信机"]

    resolved = executor._resolve_params(params, ctx)

    assert resolved["query"] == "find 智算机,激光通信机"