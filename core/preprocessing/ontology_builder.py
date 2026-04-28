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