import pytest
from unittest.mock import MagicMock, patch
from core.query_engineer.query_planner import QueryPlanner, REGION_KEYWORDS, NUMERIC_PATTERNS

def test_planner_extracts_region_constraint():
    """Planner should extract region constraints from query"""
    planner = QueryPlanner()
    with patch.object(planner, '_call_llm', return_value={"intent": "lookup", "target": "合作单位", "entities": [], "required_fields": []}):
        ir = planner.plan("上海合作单位有哪些")
        assert len(ir.constraints) >= 1
        region_constraints = [c for c in ir.constraints if c.type == "region"]
        assert len(region_constraints) >= 1

def test_planner_extracts_numeric_constraints():
    """Planner should extract numeric constraints"""
    planner = QueryPlanner()
    with patch.object(planner, '_call_llm', return_value={"intent": "recommendation", "target": "智算机", "entities": ["智算机"], "required_fields": ["型号", "重量"]}):
        ir = planner.plan("2kg以内算力大于250TFlops的智算机")
        numeric_constraints = [c for c in ir.constraints if c.type == "numeric"]
        assert len(numeric_constraints) >= 2

def test_planner_infers_filter_operation():
    """Planner should infer filter operation when constraints present"""
    planner = QueryPlanner()
    with patch.object(planner, '_call_llm', return_value={"intent": "lookup", "target": "智算机", "entities": [], "required_fields": []}):
        ir = planner.plan("重量小于3kg的智算机")
        assert "filter" in ir.operations

def test_planner_infers_aggregate_for_analysis():
    """Planner should infer aggregate operation for analysis intent"""
    planner = QueryPlanner()
    with patch.object(planner, '_call_llm', return_value={"intent": "analysis", "target": "合作单位", "entities": [], "required_fields": ["合作形式"]}):
        ir = planner.plan("按合作形式分析合作单位")
        assert ir.need_aggregate == True
        assert "aggregate" in ir.operations

def test_planner_empty_query():
    """Planner should handle empty query"""
    planner = QueryPlanner()
    ir = planner.plan("")
    assert ir.intent == "lookup"
    assert ir.target == ""

def test_planner_llm_fallback():
    """Planner should fallback when LLM fails"""
    planner = QueryPlanner()
    with patch.object(planner, '_call_llm', side_effect=Exception("LLM failed")):
        ir = planner.plan("G1重量")
        assert ir.intent == "lookup"
        assert ir.original_query == "G1重量"