"""tools/preprocessing/ontology_builder.py — 轻量本体构建

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

def _normalize_for_compare(name: str) -> str:
    """归一化：全角转半角，转小写，移除空格。"""
    result = []
    for ch in name:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        elif code == 0x3000:
            code = 0x0020
        result.append(chr(code))
    text = "".join(result).lower().strip()
    return re.sub(r'\s+', '', text)

def build_org_alias_union(entities: list[dict]) -> dict[str, dict]:
    """
    ORG 别名并查集。

    将同一家公司的不同别名（entity_name + aliases）通过 Union-Find 合并。
    合并规则：
    1. entity_name 和它的 aliases 共享同一个集合
    2. 如果 entity_name A 是 entity_name B 的子串（且长度差 > 2），合并
    3. 如果某个别名是另一个 entity_name 的子串，也合并

    Returns:
        {canonical_name: {
            "names": set of all names in the union,
            "representative": str (highest frequency entity_name in the set),
            "frequency": int (max frequency),
            "source_docs": list,
            "aliases": list (all aliases from all entities in set)
        }}
    """
    # Step 1: build name→entity index
    name_to_entity: dict[str, dict] = {}
    for e in entities:
        if e.get("entity_type") != "ORG":
            continue
        name = e["entity_name"]
        name_to_entity[name] = e
        for alias in e.get("aliases", []):
            if alias and alias not in name_to_entity:
                name_to_entity[alias] = e

    # Step 2: union-find
    n = len(name_to_entity)
    names = list(name_to_entity.keys())
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    name_to_idx = {name: i for i, name in enumerate(names)}

    for i, name in enumerate(names):
        for j in range(i + 1, n):
            other = names[j]
            # Rule 1: exact match after normalization
            if _normalize_for_compare(name) == _normalize_for_compare(other):
                union(i, j)
            # Rule 2: substring
            if len(name) >= 2 and len(other) >= 2:
                if name in other or other in name:
                    union(i, j)

    # Step 3: group by root
    groups: dict[int, list[str]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(names[i])

    # Step 4: build canonical result
    result: dict[str, dict] = {}
    for root, group_names in groups.items():
        # Find representative (highest frequency)
        best_name = max(group_names, key=lambda n: name_to_entity.get(n, {}).get("frequency", 0))
        best_entity = name_to_entity.get(best_name, {})
        all_names = set(group_names)
        all_aliases = []
        all_docs = []
        max_freq = 0
        for n in group_names:
            e = name_to_entity.get(n, {})
            all_aliases.extend(e.get("aliases", []))
            for d in e.get("source_docs", []):
                if d not in all_docs:
                    all_docs.append(d)
            freq = e.get("frequency", 0)
            if freq > max_freq:
                max_freq = freq

        result[best_name] = {
            "names": all_names,
            "representative": best_name,
            "frequency": max_freq,
            "source_docs": all_docs,
            "aliases": all_aliases,
        }

    return result


def _build_org_node(repr_name: str, union_data: dict, product_params: list[dict], cooperation: list[dict]) -> dict:
    """
    从 ORG 并查集数据构建一个 ORG 节点。

    Args:
        repr_name: 并查集代表名（如"之江实验室"）
        union_data: build_org_alias_union 返回的数据
        product_params: 产品参数列表（用于收集合作产品）
        cooperation: 合作关系列表（用于收集合作伙伴）
    """
    all_names = union_data["names"]
    node_id = generate_node_id(repr_name, "ORG")

    # 收集该 ORG 参与的产品型号（从 cooperation 的 products_or_projects 字段）
    products: set[str] = set()
    cooperators: set[str] = set()
    for coop in cooperation:
        units = coop.get("units", [])
        if repr_name in units:
            for proj in coop.get("products_or_projects", []):
                if proj:
                    products.add(proj)
            for u in units:
                if u != repr_name:
                    cooperators.add(u)

    # 也从 product_params 的合作单位字段匹配
    for model in product_params:
        cooperator = model.get("params", {}).get("合作单位", "")
        if cooperator and (cooperator in all_names or cooperator in repr_name or repr_name in cooperator):
            products.add(model.get("model", ""))

    return {
        "id": node_id,
        "type": "ORG",
        "name": repr_name,
        "aliases": list(all_names - {repr_name}),
        "frequency": union_data.get("frequency", 0),
        "source_docs": union_data.get("source_docs", []),
        "products": sorted(list(products)),
        "cooperators": sorted(list(cooperators)),
    }


def _build_model_node(model: dict) -> dict:
    """从 product_params 条目构建 MODEL 节点。"""
    model_name = model.get("model", "")
    node_id = generate_node_id(model_name, "MODEL")

    params = model.get("params", {})
    # filled_fields: 哪些字段有值（非空）
    filled_fields = [k for k, v in params.items() if v and str(v).strip()]

    # 收集 source_chunks（从所有 product_params 条目共享的 chunks，通过 model 名匹配）
    source_chunks: list[str] = []
    # 从 params 里找 source_chunks（如果有的话）
    if params.get("source_chunks"):
        source_chunks = params["source_chunks"]

    return {
        "id": node_id,
        "type": "MODEL",
        "category": model.get("category", ""),
        "series": model.get("series", ""),
        "model": model_name,
        "filled_fields": filled_fields,
        "source_chunks": source_chunks,
        "params": params,  # 保留完整参数
    }


def _build_cooperation_edge(coop: dict, org_node_map: dict[str, dict]) -> dict:
    """
    从 cooperation 条目构建 COOPERATION 边。
    from = 之江实验室（或第一个单位）
    to = 合作单位
    """
    units = coop.get("units", [])
    if len(units) < 2:
        return None

    from_org = units[0]
    to_org = units[1] if len(units) > 1 else units[0]

    from_id = org_node_map.get(from_org, {}).get("id", generate_node_id(from_org, "ORG"))
    to_id = org_node_map.get(to_org, {}).get("id", generate_node_id(to_org, "ORG"))

    return {
        "id": f"coop_{len(units)}_{from_org[:4]}_{to_org[:4]}",
        "type": "COOPERATION",
        "from": from_id,
        "to": to_id,
        "units": units,
        "content": coop.get("content", ""),
        "products_or_projects": coop.get("products_or_projects", []),
        "confidence": coop.get("confidence", 0.0),
        "source_chunks": coop.get("source_chunks", []),
    }


def _build_owned_by_edge(model: dict, org_node_map: dict[str, dict]) -> dict:
    """
    从 product_params 的合作单位字段构建 OWNED_BY 边。
    from = MODEL节点
    to = ORG节点
    """
    cooperator = model.get("params", {}).get("合作单位", "")
    if not cooperator:
        return None

    model_name = model.get("model", "")
    model_id = generate_node_id(model_name, "MODEL")

    # 找匹配的 ORG
    to_id = None
    for repr_name, node_data in org_node_map.items():
        if cooperator in repr_name or repr_name in cooperator or cooperator in node_data.get("names", set()):
            to_id = node_data.get("id")
            break

    if to_id is None:
        to_id = generate_node_id(cooperator, "ORG")

    return {
        "id": f"own_{model_name[:8]}",
        "type": "OWNED_BY",
        "from": model_id,
        "to": to_id,
        "model": model_name,
    }


def build_graph(
    entity_raw: list[dict],
    product_params: list[dict],
    cooperation: list[dict],
) -> dict:
    """
    从三个数据源构建本体图。

    Args:
        entity_raw: entity_raw 列表
        product_params: step3_product_params 列表
        cooperation: step3_cooperation 列表

    Returns:
        {"metadata": {...}, "nodes": [...], "edges": [...]}
    """
    from datetime import date

    nodes: list[dict] = []
    edges: list[dict] = []

    # 1. ORG并查集 + 构建 org_node_map{name→node}
    org_entities = [e for e in entity_raw if e.get("entity_type") == "ORG"]
    union_map = build_org_alias_union(org_entities)  # repr_name → union_data

    # 2. ORG节点
    org_node_map: dict[str, dict] = {}  # repr_name → node
    for repr_name, union_data in union_map.items():
        node = _build_org_node(repr_name, union_data, product_params, cooperation)
        nodes.append(node)
        org_node_map[repr_name] = node

    # 3. MODEL节点
    for model in product_params:
        nodes.append(_build_model_node(model))

    # 4. COOPERATION边
    for coop in cooperation:
        edge = _build_cooperation_edge(coop, org_node_map)
        if edge:
            edges.append(edge)

    # 5. OWNED_BY边
    for model in product_params:
        edge = _build_owned_by_edge(model, org_node_map)
        if edge:
            edges.append(edge)

    # 6. metadata
    org_count = sum(1 for n in nodes if n["type"] == "ORG")
    model_count = sum(1 for n in nodes if n["type"] == "MODEL")

    coop_edges = sum(1 for e in edges if e["type"] == "COOPERATION")
    own_edges = sum(1 for e in edges if e["type"] == "OWNED_BY")

    metadata = {
        "created_at": str(date.today()),
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "org_count": org_count,
        "model_count": model_count,
        "product_count": model_count,
        "coop_edge_count": coop_edges,
        "owned_by_edge_count": own_edges,
    }

    return {"metadata": metadata, "nodes": nodes, "edges": edges}