import pytest
from unittest.mock import MagicMock
from core.query_engineer.operators import (
    ExecutionContext, Operator, OperatorRegistry,
    RetrieveOperator, FilterOperator, ExtractOperator,
    ForeachOperator, AggregateOperator
)
from core.query_engineer.query_ir import Constraint

def test_execution_context_defaults():
    ctx = ExecutionContext()
    assert ctx.chunks == []
    assert ctx.entities == []
    assert ctx.filtered == []
    assert ctx.structured == {}
    assert ctx.errors == []

def test_execution_context_get_input_chunks():
    ctx = ExecutionContext()
    ctx.chunks = ["a", "b"]
    assert ctx.get_input_chunks() == ["a", "b"]
    ctx.filtered = ["a"]
    assert ctx.get_input_chunks() == ["a"]

def test_operator_registry_basic():
    registry = OperatorRegistry()
    assert len(registry.names()) == 0
    assert registry.get("nonexistent") is None

def test_operator_registry_register():
    registry = OperatorRegistry()
    mock_op = MagicMock(spec=Operator)
    mock_op.name = "test_op"
    registry.register("test_op", mock_op)
    assert "test_op" in registry.names()
    assert registry.get("test_op") is mock_op

def test_filter_operator_no_constraints():
    """FilterOperator with no constraints should return unchanged chunks"""
    ctx = ExecutionContext()
    ctx.chunks = ["chunk1", "chunk2"]
    op = FilterOperator()
    result_ctx = op.execute(ctx, {"constraints": []})
    assert result_ctx.filtered == []

def test_filter_operator_with_numeric_constraint():
    """FilterOperator should filter chunks by numeric constraint"""
    mock_chunk = MagicMock()
    mock_chunk.content = "重量: 2.5kg"

    ctx = ExecutionContext()
    ctx.chunks = [mock_chunk]

    op = FilterOperator()
    constraint = Constraint(type="numeric", field="重量", op="<=", value=3, unit="kg")
    result_ctx = op.execute(ctx, {"constraints": [constraint]})

    # The chunk with 2.5kg should pass the <=3 constraint
    assert len(result_ctx.filtered) == 1

def test_extract_operator():
    """ExtractOperator should extract field values from chunks"""
    mock_chunk = MagicMock()
    mock_chunk.content = "重量: 2.5kg\n算力: 200TFlops"

    ctx = ExecutionContext()
    ctx.chunks = [mock_chunk]

    op = ExtractOperator()
    result_ctx = op.execute(ctx, {"fields": ["重量", "算力"]})

    assert "重量" in result_ctx.structured
    assert "算力" in result_ctx.structured

def test_aggregate_operator():
    """AggregateOperator should aggregate by dimensions"""
    ctx = ExecutionContext()
    ctx.structured = {
        "合作形式": ["联合研制", "技术合作", "联合研制"],
        "地区": ["上海", "北京", "上海"],
    }

    op = AggregateOperator()
    result_ctx = op.execute(ctx, {"dimensions": ["合作形式", "地区"]})

    assert result_ctx.aggregated is not None
    assert "合作形式" in result_ctx.aggregated
    assert "地区" in result_ctx.aggregated

def test_foreach_operator():
    """ForeachOperator should iterate over entities"""
    ctx = ExecutionContext()
    ctx.entities = ["智算机", "激光通信机"]

    mock_retrieve_op = MagicMock()
    mock_retrieve_op.name = "retrieve"
    mock_retrieve_result_ctx = ExecutionContext()
    mock_retrieve_result_ctx.chunks = ["chunk1"]
    mock_retrieve_op.execute.return_value = mock_retrieve_result_ctx

    registry = OperatorRegistry()
    registry.register("retrieve", mock_retrieve_op)

    op = ForeachOperator(registry)
    result_ctx = op.execute(ctx, {
        "entities": ["智算机", "激光通信机"],
        "sub_operator": "retrieve",
        "sub_params": {"query": "{entity}"},
    })

    # Should have called retrieve twice (once per entity)
    assert mock_retrieve_op.execute.call_count == 2