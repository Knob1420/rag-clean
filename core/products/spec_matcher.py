"""
结构化产品参数查询服务

支持三层结构：产品大类 → 产品系列 → 具体型号
JSON 结构（products_specs.json）：
  {
    "星载智能计算机": [
      {
        "series": "智加G系列",
        "model_list": [{"model": "智加G1", "params": {...}}, ...]
      },
      ...
    ],
    ...
  }
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


# ============================================================
# 别名映射（从 data/alias_table.json 加载）
# ============================================================

_ALIASES: Optional[Dict[str, Dict[str, str]]] = None


def _load_aliases() -> Dict[str, Dict[str, str]]:
    """从 alias_table.json 加载别名归一化表"""
    path = Path(__file__).parent.parent.parent / "data" / "alias_table.json"
    if not path.exists():
        logger.warning(f"[SpecMatcher] 别名表不存在: {path}")
        return {"category": {}, "series": {}, "model": {}, "field": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_aliases() -> Dict[str, Dict[str, str]]:
    global _ALIASES
    if _ALIASES is None:
        _ALIASES = _load_aliases()
    return _ALIASES


def reload_aliases():
    """重新加载别名表（用于热更新）"""
    global _ALIASES
    _ALIASES = _load_aliases()


# 便捷属性：保持与原代码兼容的访问方式
def _get_category_aliases() -> Dict[str, str]:
    return _get_aliases().get("category", {})


def _get_series_aliases() -> Dict[str, str]:
    return _get_aliases().get("series", {})


def _get_model_aliases() -> Dict[str, str]:
    return _get_aliases().get("model", {})


def _get_field_aliases() -> Dict[str, str]:
    return _get_aliases().get("field", {})


# ============================================================
# 辅助函数
# ============================================================


def _normalize_field(field: str) -> str:
    return _get_field_aliases().get(field, field)


def _get_category_groups() -> Dict[str, List[str]]:
    """返回 {组合大类名: [子类别列表]}（如 '太空计算组件' → ['星载智能计算机', ...]）"""
    return _get_aliases().get("category_groups", {})


def _normalize_category_name(name: str) -> str:
    """将大类别名映射为标准大类名（仅处理单一类别）"""
    return _get_category_aliases().get(name, name)


def _resolve_category(name: str) -> List[str]:
    """
    解析类别名，返回实际存在的子类别列表。

    - 单一类别（如 '星载智能计算机'）→ ['星载智能计算机']
    - 组合类别（如 '太空计算组件'）→ ['星载智能计算机', '星载路由器', '星载激光通信机']
    - 别名先标准化再判断
    """
    normalized = _normalize_category_name(name)
    groups = _get_category_groups()

    # 先检查是否是组合类别
    if normalized in groups:
        return groups[normalized]

    # 不是组合类别，返回单一类别
    return [normalized]


def _normalize_series_name(name: str) -> Optional[str]:
    """将系列别名映射为标准系列名"""
    return _get_series_aliases().get(name)


def _normalize_model_name(name: str) -> str:
    """将型号别名映射为标准型号名"""
    return _get_model_aliases().get(name, name)


def _build_series_alias_map() -> Dict[str, str]:
    """构建别名→标准名的反向索引（用于快速判断某词是否系列别名）"""
    return {
        alias: standard
        for standard, aliases in _get_series_groups().items()
        for alias in aliases
    }


def _get_series_groups() -> Dict[str, List[str]]:
    """返回 {标准系列名: [别名列表（含标准名）]}"""
    out = {}
    for alias, std in _get_series_aliases().items():
        if std not in out:
            out[std] = []
        out[std].append(alias)
    return out


# ============================================================
# 数据加载
# ============================================================


def _load_product_specs() -> Dict[str, Any]:
    """加载三层结构产品参数表"""
    specs_path = Path(__file__).parent.parent.parent / "data" / "products_specs.json"
    with open(specs_path, "r", encoding="utf-8") as f:
        return json.load(f)


_PRODUCT_SPECS: Optional[Dict[str, Any]] = None


def get_product_specs() -> Dict[str, Any]:
    global _PRODUCT_SPECS
    if _PRODUCT_SPECS is None:
        _PRODUCT_SPECS = _load_product_specs()
    return _PRODUCT_SPECS


def reload_product_specs():
    """重新加载（用于切换数据版本）"""
    global _PRODUCT_SPECS
    _PRODUCT_SPECS = _load_product_specs()


def _build_series_lookup(specs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    构建 {系列名: {category, series, model_list}} 的索引
    用于快速判断某型号属于哪个系列
    """
    lookup = {}
    for category, series_list in specs.items():
        if not isinstance(series_list, list):
            continue
        for series_item in series_list:
            if not isinstance(series_item, dict):
                continue
            series_name = series_item.get("series", "")
            if not series_name:
                continue
            lookup[series_name] = {
                "category": category,
                "series": series_name,
                "model_list": series_item.get("model_list", []),
            }
            # 将别名也索引到同一项
            for alias, std in _get_series_aliases().items():
                if std == series_name:
                    lookup[alias] = lookup[series_name]
    return lookup


def _build_model_to_series_lookup(specs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """构建 {型号名: {category, series, params}} 索引"""
    lookup = {}
    for category, series_list in specs.items():
        if not isinstance(series_list, list):
            continue
        for series_item in series_list:
            series_name = series_item.get("series", "")
            for model_entry in series_item.get("model_list", []):
                if not isinstance(model_entry, dict):
                    continue
                model_name = model_entry.get("model", "")
                if model_name:
                    lookup[model_name] = {
                        "category": category,
                        "series": series_name,
                        "params": model_entry.get("params", {}),
                    }
    return lookup


# ============================================================
# 数值解析
# ============================================================


def _parse_weight(weight_str: str) -> Optional[float]:
    match = re.search(r"([\d.]+)\s*kg", weight_str)
    if match:
        return float(match.group(1))
    match = re.search(r"([\d.]+)\s*kg", weight_str.split("（")[0])
    if match:
        return float(match.group(1))
    return None


def _parse_power(power_str: str) -> Optional[float]:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*W", power_str)
    if not matches:
        return None
    return float(matches[0])


def _parse_tops(tops_str: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*TOPS", tops_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


# ============================================================
# 核心查询
# ============================================================


def query_products(
    target_models: List[str],
    required_fields: List[str],
    numerical_constraints: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    根据意图查询参数表（三层匹配规则）

    命中【产品大类】：遍历旗下所有系列 → 所有型号，全部汇总
    命中【产品系列】：只遍历当前系列下全部型号，自动聚合
    命中【具体型号】：精准单条
    """
    specs = get_product_specs()
    if not specs:
        logger.warning("[SpecMatcher] 产品参数表为空或不存在")
        return []

    results = []

    # 标准化字段
    normalized_constraints: Dict[str, str] = {}
    for field, constraint in numerical_constraints.items():
        normalized_constraints[_normalize_field(field)] = constraint

    # ── 预处理：把 target_models 分类 ──
    #   - 大类列表：直接在 specs 顶层 key 中命中
    #   - 系列列表：标准化为 "智加G系列"
    #   - 型号列表：标准化为 "智加G1"
    categories_hit = set()  # 需要全展开的大类
    series_hit = []  # 需要展开的系列
    models_hit = []  # 直接匹配的型号

    # 先把所有输入归一化为可能的名称
    raw_inputs = [_normalize_model_name(m) for m in target_models]

    for raw in raw_inputs:
        # 1. 检查是否是大类名（顶层 key）或组合大类，先用别名标准化
        raw_normalized = _normalize_category_name(raw)
        resolved_cats = _resolve_category(raw)
        # 过滤出 specs 中实际存在的类别
        actual_cats = [c for c in resolved_cats if c in specs]
        if actual_cats:
            categories_hit.update(actual_cats)
            continue

        # 2. 检查是否是系列名（通过 SERIES_ALIASES）
        std_series = _normalize_series_name(raw)
        if std_series:
            series_hit.append(std_series)
            continue

        # 3. 剩下的是具体型号
        models_hit.append(raw)

    # ── 构建索引 ──
    series_lookup = _build_series_lookup(specs)
    model_lookup = _build_model_to_series_lookup(specs)

    # 收集需要匹配的具体型号（展开系列 + 直接型号）
    models_to_find: Dict[str, Dict[str, Any]] = (
        {}
    )  # model_name → {category, series, params}

    # 从命中的大类展开所有系列 → 所有型号
    for cat in categories_hit:
        for series_item in specs.get(cat, []):
            if not isinstance(series_item, dict):
                continue

            # ── 3 层结构：{series: "...", model_list: [{model: "...", params: {...}}]}
            if "model_list" in series_item:
                series_name = series_item.get("series", "")
                for model_entry in series_item.get("model_list", []):
                    if not isinstance(model_entry, dict):
                        continue
                    mn = model_entry.get("model", "")
                    if mn and mn not in models_to_find:
                        models_to_find[mn] = {
                            "category": cat,
                            "series": series_name,
                            "params": model_entry.get("params", {}),
                        }
                continue

            # ── 2 层结构：{name: "合作单位名/卫星类型名", 发射卫星数/发射数量: N, ...}
            item_name = series_item.get("name", "")
            if item_name and item_name not in models_to_find:
                models_to_find[item_name] = {
                    "category": cat,
                    "series": cat,  # 2 层没有系列，用大类名代替
                    "params": series_item,  # 把整条记录作为 params
                }

    # 从命中的系列展开其下所有型号
    for s in set(series_hit):
        entry = series_lookup.get(s)
        if not entry:
            continue
        cat = entry["category"]
        for model_entry in entry["model_list"]:
            if not isinstance(model_entry, dict):
                continue
            mn = model_entry.get("model", "")
            if mn and mn not in models_to_find:
                models_to_find[mn] = {
                    "category": cat,
                    "series": entry["series"],
                    "params": model_entry.get("params", {}),
                }

    # 直接型号
    for mn in set(models_hit):
        entry = model_lookup.get(mn)
        if entry and mn not in models_to_find:
            models_to_find[mn] = entry

    # ── 对每个候选型号检查数值约束 ──
    for model_name, info in models_to_find.items():
        params = info["params"]
        category = info["category"]
        series_name = info["series"]

        violate = False
        for field, constraint in normalized_constraints.items():
            if field not in params:
                continue
            value_str = params[field]
            parsed_value: Optional[float] = None

            if field in ("重量", "weight"):
                parsed_value = _parse_weight(value_str)
            elif field in ("功耗", "power"):
                parsed_value = _parse_power(value_str)
            elif field in ("算力", "compute", "tops"):
                parsed_value = _parse_tops(value_str)

            if parsed_value is None:
                continue

            constraint_match = re.match(r"(<=|>=|<|>|<|==)\s*([\d.]+)", constraint)
            if not constraint_match:
                continue

            op, threshold = constraint_match.groups()
            threshold = float(threshold)

            if op == "<=" and not (parsed_value <= threshold):
                violate = True
            elif op == ">=" and not (parsed_value >= threshold):
                violate = True
            elif op == "<" and not (parsed_value < threshold):
                violate = True
            elif op == ">" and not (parsed_value > threshold):
                violate = True
            elif op == "==" and not (parsed_value == threshold):
                violate = True

            if violate:
                break

        if violate:
            continue

        # 构建结果
        result: Dict[str, Any] = {
            "product_name": model_name,
            "category": category,
            "series": series_name,
        }

        if required_fields:
            for field in required_fields:
                normalized = _normalize_field(field)
                if normalized in params:
                    result[normalized] = params[normalized]
            # 补全其他字段
            for key, value in params.items():
                if key not in result:
                    result[key] = value
        else:
            result.update(params)

        results.append(result)

    return results


# ============================================================
# 合作单位 & 卫星类型 查询
# ============================================================


def query_partners(target_companies: Optional[List[str]] = None) -> Dict[str, int]:
    """
    查询合作单位的卫星数量

    Args:
        target_companies: 目标合作单位列表，为空则返回全部

    Returns:
        {单位名: 卫星数量}
    """
    specs = get_product_specs()
    all_partners = specs.get("合作单位", [])
    if not all_partners:
        return {}
    # 支持旧 dict 格式（{"单位名": 数量}）和 新 list 格式（[{"name": "单位名", "发射卫星数": 数量}]）
    if isinstance(all_partners, dict):
        if not target_companies:
            return all_partners
        return {k: v for k, v in all_partners.items() if k in target_companies}
    # 新 list 格式
    result = {}
    for p in all_partners:
        if isinstance(p, dict):
            name = p.get("name", "")
            count = p.get("发射卫星数", 0)
        else:
            name = p
            count = 0
        if name and (not target_companies or name in target_companies):
            result[name] = count
    return result


def query_sat_types(target_types: Optional[List[str]] = None) -> Dict[str, int]:
    """
    查询卫星类型的发射数量

    Args:
        target_types: 目标卫星类型列表，为空则返回全部

    Returns:
        {类型名: 发射数量}
    """
    specs = get_product_specs()
    all_types = specs.get("卫星类型", [])
    if not all_types:
        return {}
    # 支持旧 dict 格式和 新 list 格式（[{"name": "类型名", "发射数量": N}]）
    if isinstance(all_types, dict):
        if not target_types:
            return all_types
        return {k: v for k, v in all_types.items() if k in target_types}
    # 新 list 格式
    result = {}
    for t in all_types:
        if isinstance(t, dict):
            name = t.get("name", "")
            count = t.get("发射数量", 0)
        else:
            name = t
            count = 0
        if name and (not target_types or name in target_types):
            result[name] = count
    return result


def format_partner_context(partners: Dict[str, int], sat_types: Dict[str, int]) -> str:
    """格式化合作单位和卫星类型的查询结果"""
    parts = ["【项目信息】"]

    if partners:
        parts.append("\n## 各单位合作卫星数")
        for unit, count in sorted(partners.items(), key=lambda x: -x[1]):
            parts.append(f"  {unit}: {count} 颗")

    if sat_types:
        parts.append("\n## 卫星类型分布")
        for stype, count in sorted(sat_types.items(), key=lambda x: -x[1]):
            parts.append(f"  {stype}: {count} 颗")

    if not partners and not sat_types:
        return "（知识库中无匹配的项目信息）"

    return "\n".join(parts)


# ============================================================
# 格式化输出
# ============================================================


def format_spec_context(query_results: List[Dict[str, Any]], intent: str) -> str:
    if not query_results:
        return "（知识库中无匹配的结构化产品参数）"

    parts = ["【产品参数表查询结果】"]

    # 按系列分组展示
    by_series: Dict[str, List[Dict[str, Any]]] = {}
    for p in query_results:
        s = p.get("series", "其他")
        if s not in by_series:
            by_series[s] = []
        by_series[s].append(p)

    for series_name, products in by_series.items():
        parts.append(f"\n## 系列：{series_name}")
        for i, product in enumerate(products, 1):
            parts.append(f"\n### {i}. {product.get('product_name', '未知产品')}")
            parts.append(f"   分类: {product.get('category', '未知')}")

            key_fields = [
                "合作单位",
                "架构",
                "算力",
                "重量",
                "功耗",
                "尺寸",
                "内存",
                "存储",
                "对外接口",
            ]
            for field in key_fields:
                if field in product and product[field]:
                    parts.append(f"   {field}: {product[field]}")

            for field, value in product.items():
                if field in ("product_name", "category", "series") or not value:
                    continue
                if field not in key_fields:
                    parts.append(f"   {field}: {value}")

    return "\n".join(parts)
