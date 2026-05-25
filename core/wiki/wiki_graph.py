"""
Wiki Graph Builder - 从 wiki 页面构建知识图谱

从 wiki/ 目录下的 md 文件提取节点（页面）和边（wikilinks），
并使用 Louvain 算法检测社区。

Ported from LLM Wiki (src/lib/wiki-graph.ts)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from loguru import logger

import networkx as nx


# ══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GraphNode:
    """图谱节点"""
    id: str           # 页面 slug
    label: str        # 显示名称
    type: str         # 页面类型
    path: str         # 文件路径
    link_count: int  # 引用次数
    community: int    # 社区 ID


@dataclass
class GraphEdge:
    """图谱边"""
    source: str
    target: str
    weight: float


@dataclass
class CommunityInfo:
    """社区信息"""
    id: int
    node_count: int
    cohesion: float
    top_nodes: List[str]


# ══════════════════════════════════════════════════════════════════════════════
# 正则和常量
# ══════════════════════════════════════════════════════════════════════════════

WIKILINK_REGEX = re.compile(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]')

# 按类型过滤的节点
HIDDEN_TYPES = {'query'}


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════


def file_name_to_id(file_name: str) -> str:
    """从文件名获取节点 ID（去掉 .md 后缀）"""
    return file_name.replace('.md', '')


def resolve_target(
    raw: str,
    node_map: Dict[str, dict],
) -> Optional[str]:
    """
    解析 wikilink 目标，尝试匹配节点 ID

    Args:
        raw: wikilink 中的原始文本
        node_map: 节点 ID -> 节点信息的映射

    Returns:
        匹配到的节点 ID，或 None
    """
    # 直接匹配
    if raw in node_map:
        return raw

    # 标准化匹配（大小写不敏感，空格/连字符互换）
    normalized = raw.lower().replace(' ', '-')

    for node_id in node_map.keys():
        if node_id.lower() == normalized:
            return node_id
        if node_id.lower() == raw.lower():
            return node_id
        if node_id.lower().replace(' ', '-') == normalized:
            return node_id

    return None


def extract_title(content: str, file_name: str) -> str:
    """从文件内容提取标题"""
    # 尝试 frontmatter title
    match = re.search(r'^title:\s*(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip().strip('"\'')
    # 尝试第一个 # 标题
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # 回退到文件名
    return file_name.replace('.md', '').replace('-', ' ')


def extract_type(content: str) -> str:
    """从文件内容提取类型"""
    match = re.search(r'^type:\s*(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return 'other'


def extract_wikilinks(content: str) -> List[str]:
    """从文件内容提取所有 wikilinks"""
    return [m.group(1).strip() for m in WIKILINK_REGEX.finditer(content)]


# ══════════════════════════════════════════════════════════════════════════════
# 社区检测
# ══════════════════════════════════════════════════════════════════════════════


def detect_communities(
    nodes: List[dict],
    edges: List[GraphEdge],
) -> Tuple[Dict[str, int], List[CommunityInfo]]:
    """
    使用 Louvain 算法检测社区

    Args:
        nodes: 节点列表 [{id, label, link_count}]
        edges: 边列表

    Returns:
        (node_id -> community_id, communities 信息)
    """
    if not nodes:
        return {}, []

    # 构建 NetworkX 图
    G = nx.Graph()
    for node in nodes:
        G.add_node(node['id'])

    for edge in edges:
        if edge.source in G and edge.target in G:
            if G.has_edge(edge.source, edge.target):
                G[edge.source][edge.target]['weight'] += edge.weight
            else:
                G.add_edge(edge.source, edge.target, weight=edge.weight)

    # Louvain 社区检测
    try:
        from networkx.algorithms.community import louvain_communities
        partition_list = louvain_communities(G, weight='weight', resolution=1)
        # partition_list 是 List[Set]，转为 {node_id: community_id}
        partition = {}
        for comm_id, comm_nodes in enumerate(partition_list):
            for node in comm_nodes:
                partition[node] = comm_id
    except Exception as e:
        logger.warning(f"Community detection failed: {e}")
        # 回退：每个节点单独一个社区
        partition = {node['id']: i for i, node in enumerate(nodes)}

    assignments = dict(partition)

    # 按社区分组
    groups: Dict[int, List[str]] = {}
    for node_id, comm_id in assignments.items():
        if comm_id not in groups:
            groups[comm_id] = []
        groups[comm_id].append(node_id)

    # 构建边集合用于内聚度计算
    edge_set = set()
    for edge in edges:
        edge_set.add((edge.source, edge.target))
        edge_set.add((edge.target, edge.source))

    # 节点信息映射
    node_info = {n['id']: n for n in nodes}

    # 计算每个社区的信息
    communities: List[CommunityInfo] = []
    for comm_id, member_ids in groups.items():
        n = len(member_ids)

        # 计算内聚度（实际边数 / 可能边数）
        intra_edges = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (member_ids[i], member_ids[j]) in edge_set or (member_ids[j], member_ids[i]) in edge_set:
                    intra_edges += 1

        possible_edges = n * (n - 1) / 2 if n > 1 else 1
        cohesion = intra_edges / possible_edges if possible_edges > 0 else 0

        # 按 linkCount 排序的 top 节点
        sorted_ids = sorted(member_ids, key=lambda x: node_info[x]['link_count'], reverse=True)
        top_nodes = [node_info[mid]['label'] for mid in sorted_ids[:5]]

        communities.append(CommunityInfo(
            id=comm_id,
            node_count=n,
            cohesion=cohesion,
            top_nodes=top_nodes,
        ))

    # 按 node_count 降序排序
    communities.sort(key=lambda c: c.node_count, reverse=True)

    # 重新编号（从 0 开始）
    id_remap = {c.id: i for i, c in enumerate(communities)}
    communities = [CommunityInfo(
        id=id_remap[c.id],
        node_count=c.node_count,
        cohesion=c.cohesion,
        top_nodes=c.top_nodes,
    ) for c in communities]

    assignments = {node_id: id_remap.get(comm_id, 0) for node_id, comm_id in assignments.items()}

    return assignments, communities


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════


def flatten_md_files(tree: List[dict]) -> List[dict]:
    """递归扁平化文件树，提取所有 .md 文件"""
    files = []
    for node in tree:
        if node.get('is_dir') and node.get('children'):
            files.extend(flatten_md_files(node['children']))
        elif not node.get('is_dir') and node['name'].endswith('.md'):
            files.append(node)
    return files


def build_wiki_graph(project_path: str) -> Dict:
    """
    从 wiki 目录构建知识图谱

    Args:
        project_path: 项目根目录

    Returns:
        {nodes: [], edges: [], communities: []}
    """
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    wiki_root = Path(project_path)

    # 遍历 wiki 目录下的所有 md 文件
    md_files: List[Path] = []
    for md_path in wiki_root.rglob('*.md'):
        # 跳过隐藏文件和非 wiki 页面
        if md_path.stem.startswith('.'):
            continue
        md_files.append(md_path)

    if not md_files:
        logger.info("[wiki-graph] No md files found")
        return {'nodes': [], 'edges': [], 'communities': []}

    logger.info(f"[wiki-graph] Found {len(md_files)} md files")

    # 构建节点映射
    node_map: Dict[str, dict] = {}
    for md_path in md_files:
        node_id = md_path.stem  # 文件名作为 ID

        try:
            content = md_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"[wiki-graph] Failed to read {md_path}: {e}")
            continue

        node_map[node_id] = {
            'id': node_id,
            'label': extract_title(content, md_path.name),
            'type': extract_type(content),
            'path': str(md_path),
            'links': extract_wikilinks(content),
        }

    # 过滤隐藏类型
    for node_id in list(node_map.keys()):
        if node_map[node_id]['type'] in HIDDEN_TYPES:
            del node_map[node_id]

    logger.info(f"[wiki-graph] {len(node_map)} nodes after filtering")

    # 统计引用次数
    link_counts: Dict[str, int] = {nid: 0 for nid in node_map}

    # 构建边
    raw_edges: List[Tuple[str, str]] = []
    for source_id, node_data in node_map.items():
        for target_raw in node_data['links']:
            target_id = resolve_target(target_raw, node_map)
            if target_id is None or target_id == source_id:
                continue

            raw_edges.append((source_id, target_id))
            link_counts[source_id] = link_counts.get(source_id, 0) + 1
            link_counts[target_id] = link_counts.get(target_id, 0) + 1

    # 去重边
    seen_edges: Set[Tuple[str, str]] = set()
    deduped_edges: List[GraphEdge] = []
    for source, target in raw_edges:
        key = (min(source, target), max(source, target))
        if key not in seen_edges:
            seen_edges.add(key)
            deduped_edges.append(GraphEdge(source=source, target=target, weight=1.0))

    logger.info(f"[wiki-graph] {len(deduped_edges)} unique edges")

    # 社区检测
    prelim_nodes = [
        {'id': nid, 'label': ndata['label'], 'link_count': link_counts.get(nid, 0)}
        for nid, ndata in node_map.items()
    ]

    assignments, communities = detect_communities(prelim_nodes, deduped_edges)

    # 构建最终节点列表
    nodes = [
        GraphNode(
            id=nid,
            label=ndata['label'],
            type=ndata['type'],
            path=ndata['path'],
            link_count=link_counts.get(nid, 0),
            community=assignments.get(nid, 0),
        )
        for nid, ndata in node_map.items()
    ]

    return {
        'nodes': [asdict(n) for n in nodes],
        'edges': [asdict(e) for e in deduped_edges],
        'communities': [asdict(c) for c in communities],
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI / 测试
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python wiki_graph.py <project_path>")
        sys.exit(1)

    result = build_wiki_graph(sys.argv[1])

    print(f"\n=== Wiki Graph ===")
    print(f"Nodes: {len(result['nodes'])}")
    print(f"Edges: {len(result['edges'])}")
    print(f"Communities: {len(result['communities'])}")

    print(f"\n=== Top Communities ===")
    for comm in result['communities'][:5]:
        print(f"  Community {comm['id']}: {comm['node_count']} nodes, cohesion={comm['cohesion']:.3f}")
        print(f"    Top nodes: {', '.join(comm['top_nodes'][:3])}")

    print(f"\n=== Sample Nodes ===")
    for node in result['nodes'][:5]:
        print(f"  [{node['type']}] {node['label']} (community={node['community']}, links={node['link_count']})")