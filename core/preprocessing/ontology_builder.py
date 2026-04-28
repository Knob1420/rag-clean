"""core/preprocessing/ontology_builder.py — 轻量本体构建

从 entity_raw + product_params + cooperation 数据构建轻量本体图。
"""

import json
import re
from pathlib import Path
from typing import Any

def load_entity_raw(path: str) -> list[dict]:
    """加载 entity_raw JSON 文件。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_product_params(path: str) -> list[dict]:
    """加载 product_params JSON 文件。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_cooperation(path: str) -> list[dict]:
    """加载 cooperation JSON 文件。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def generate_node_id(name: str, node_type: str) -> str:
    """
    生成稳定节点 ID。
    规则：保留中文/英文/数字，前8字 + 类型前缀。
    例: 之江实验室, ORG → org_之江实验室
    """
    # 保留中文、英文、数字
    clean = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '', name)
    prefix = node_type.lower()
    return f"{prefix}_{clean[:8]}"

def save_graph(graph: dict, output_path: str) -> str:
    """保存本体图到 JSON 文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    return str(path)

def load_graph(path: str) -> dict:
    """加载本体图 JSON 文件。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)