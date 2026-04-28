"""tests/test_ontology_builder.py"""
import pytest
from pathlib import Path

def test_generate_node_id():
    from core.preprocessing.ontology_builder import generate_node_id
    assert generate_node_id("之江实验室", "ORG") == "org_之江实验室"
    assert generate_node_id("成都国星宇航科技股份有限公司", "ORG") == "org_成都国星宇航科技"
    assert generate_node_id("智加NX1", "MODEL") == "model_智加NX1"

def test_load_entity_raw(tmp_path):
    from core.preprocessing.ontology_builder import load_entity_raw
    test_file = tmp_path / "entity_raw.json"
    test_file.write_text('[{"entity_name":"测试","entity_type":"ORG"}]', encoding="utf-8")
    result = load_entity_raw(str(test_file))
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["entity_name"] == "测试"

def test_load_product_params(tmp_path):
    from core.preprocessing.ontology_builder import load_product_params
    test_file = tmp_path / "product_params.json"
    test_file.write_text('[{"model":"智加NX1","params":{}}]', encoding="utf-8")
    result = load_product_params(str(test_file))
    assert isinstance(result, list)
    assert result[0]["model"] == "智加NX1"

def test_load_cooperation(tmp_path):
    from core.preprocessing.ontology_builder import load_cooperation
    test_file = tmp_path / "cooperation.json"
    test_file.write_text('[{"units":["A","B"],"content":"合作"}]', encoding="utf-8")
    result = load_cooperation(str(test_file))
    assert isinstance(result, list)
    assert result[0]["units"] == ["A","B"]

def test_save_and_load_graph(tmp_path):
    from core.preprocessing.ontology_builder import save_graph, load_graph
    graph = {
        "metadata": {"total_nodes": 1, "total_edges": 0, "org_count": 1, "product_count": 0, "model_count": 0},
        "nodes": [{"id": "org_test", "type": "ORG", "name": "测试"}],
        "edges": []
    }
    out = tmp_path / "ontology.json"
    path = save_graph(graph, str(out))
    assert path == str(out)
    loaded = load_graph(str(out))
    assert loaded["metadata"]["total_nodes"] == 1
    assert len(loaded["nodes"]) == 1