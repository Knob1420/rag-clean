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

def test_infer_expand_map_known_category():
    """_infer_expand_map should return correct expand map for known categories"""
    planner = QueryPlanner()
    expand_map = planner._infer_expand_map("智算机有哪些", ["智算机系列"])
    assert "智算机系列" in expand_map
    assert "NX1" in expand_map["智算机系列"]
    assert "G1" in expand_map["智算机系列"]

def test_infer_expand_map_unknown_category():
    """_infer_expand_map should return empty dict for unknown categories"""
    planner = QueryPlanner()
    expand_map = planner._infer_expand_map("查询产品", ["未知类别"])
    assert expand_map == {}

def test_infer_expand_map_multiple_entities():
    """_infer_expand_map should expand multiple known entities"""
    planner = QueryPlanner()
    expand_map = planner._infer_expand_map("产品有哪些", ["产品", "智算机系列"])
    assert "产品" in expand_map
    assert "智算机系列" in expand_map

def test_infer_dimensions_analysis_intent():
    """_infer_dimensions should return correct dimensions for analysis intent"""
    planner = QueryPlanner()
    dims = planner._infer_dimensions("按合作形式分析合作单位", "analysis")
    assert "合作形式" in dims
    assert "合作单位" in dims

def test_infer_dimensions_non_analysis_intent():
    """_infer_dimensions should return empty list for non-analysis intent"""
    planner = QueryPlanner()
    dims = planner._infer_dimensions("查询智算机重量", "lookup")
    assert dims == []

def test_needs_foreach_with_keywords():
    """_needs_foreach should return True when query has foreach keywords"""
    planner = QueryPlanner()
    llm_result = {"entities": ["智算机", "激光通信机"]}
    result = planner._needs_foreach("分别查询智算机和激光通信机的重量", llm_result)
    assert result == True

def test_needs_foreach_without_keywords():
    """_needs_foreach should return False when no foreach keywords"""
    planner = QueryPlanner()
    llm_result = {"entities": ["智算机"]}
    result = planner._needs_foreach("查询智算机重量", llm_result)
    assert result == False

def test_needs_foreach_single_entity():
    """_needs_foreach should return False with single entity even with keywords"""
    planner = QueryPlanner()
    llm_result = {"entities": ["智算机"]}
    result = planner._needs_foreach("分别查询智算机的重量", llm_result)
    assert result == False

def test_region_expansion_yangtze_delta():
    """Planner should expand 长三角 to sub-regions"""
    planner = QueryPlanner()
    with patch.object(planner, '_call_llm', return_value={"intent": "lookup", "target": "合作单位", "entities": [], "required_fields": []}):
        ir = planner.plan("长三角合作单位有哪些")
        region_constraints = [c for c in ir.constraints if c.type == "region"]
        assert len(region_constraints) >= 1
        region_values = region_constraints[0].value
        assert "上海" in region_values
        assert "江苏" in region_values
        assert "浙江" in region_values
        assert "长三角" not in region_values