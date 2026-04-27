from core.query_engineer.query_ir import Constraint, QueryIR

def test_constraint_numeric():
    c = Constraint(type="numeric", field="weight", op="<=", value=3, unit="kg")
    assert c.type == "numeric"
    assert c.field == "weight"
    assert c.op == "<="
    assert c.value == 3

def test_constraint_region():
    c = Constraint(type="region", field="region", op="contains", value=["上海", "江苏"])
    assert c.type == "region"

def test_constraint_matches_numeric():
    c = Constraint(type="numeric", field="weight", op="<=", value=3, unit="kg")
    assert c.matches(2.5)
    assert c.matches(3.0)
    assert not c.matches(3.1)

def test_query_ir_full():
    ir = QueryIR(
        original_query="推荐2kg以内算力大于250TFlops的智算机",
        intent="recommendation",
        target="智算机",
        constraints=[
            Constraint(type="numeric", field="weight", op="<=", value=3, unit="kg"),
            Constraint(type="numeric", field="compute", op=">", value=250, unit="TFlops"),
        ],
        operations=["filter"],
        expand_map={},
        need_split=False,
        need_aggregate=False,
        dimensions=[],
        entities=["智算机"],
        required_fields=["型号", "重量", "算力"],
    )
    assert ir.intent == "recommendation"
    assert len(ir.constraints) == 2
    assert "filter" in ir.operations
    assert ir.needs_filter()

def test_query_ir_helper_methods():
    ir = QueryIR(
        original_query="test",
        intent="lookup",
        target="test",
        operations=["filter", "expand"],
    )
    assert ir.needs_filter()
    assert ir.needs_expand()
    assert not ir.needs_aggregate()

def test_query_ir_empty_constraints():
    ir = QueryIR(
        original_query="test",
        intent="lookup",
        target="test",
        constraints=[],
    )
    assert not ir.has_constraints()