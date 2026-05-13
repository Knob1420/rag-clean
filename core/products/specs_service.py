"""
产品参数查询服务

提供结构化产品参数查询接口，支持：
- 按产品类型、型号筛选
- 按字段条件过滤（如重量<3kg）
- 与 RAG 生成流程集成，自动注入结构化数据
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# 产品参数文件路径
SPECS_FILE = Path(__file__).parent.parent.parent / "data" / "products_specs.json"


def _load_specs() -> Dict[str, Any]:
    """加载产品参数 JSON"""
    if not SPECS_FILE.exists():
        logger.warning(f"产品参数文件不存在: {SPECS_FILE}")
        return {}
    try:
        return json.loads(SPECS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"加载产品参数失败: {e}")
        return {}


def _parse_filter(filter_str: str) -> Tuple[str, str, Any]:
    """
    解析过滤条件，返回 (字段名, 操作符, 值)

    支持格式:
    - "重量<3kg" -> ("重量", "<", "3kg")
    - "算力>=10" -> ("算力", ">=", "10")
    - "尺寸==100x100x50" -> ("尺寸", "==", "100x100x50")
    """
    match = re.match(r"(.+?)([<>=]+)(.+)", filter_str)
    if not match:
        return (filter_str, "==", "")
    return (match.group(1), match.group(2), match.group(3))


def _matches_filter(value: str, operator: str, target: str) -> bool:
    """判断值是否满足过滤条件"""
    if not value:
        return False

    # 尝试数值比较（处理 "3kg", "10W", "5TOPS" 等）
    def extract_number(s: str) -> Optional[float]:
        m = re.search(r"[\d.]+", s)
        return float(m.group()) if m else None

    num_value = extract_number(value)
    num_target = extract_number(target)

    if num_value is not None and num_target is not None:
        if operator == "<":
            return num_value < num_target
        elif operator == "<=":
            return num_value <= num_target
        elif operator == ">":
            return num_value > num_target
        elif operator == ">=":
            return num_value >= num_target
        elif operator == "==":
            return num_value == num_target

    # 字符串比较
    if operator == "==":
        return value == target
    elif operator == "!=":
        return value != target

    return False


def query_specs(
    product: Optional[str] = None,
    model: Optional[str] = None,
    filter_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    查询产品参数

    Args:
        product: 产品类型（星载智算机 / 星载路由器 / 星载激光通信机）
        model: 型号（如 NX1, G1, 智加G3）
        filter_field: 过滤条件（如 "重量<3kg"）

    Returns:
        符合条件的参数列表，每项包含 product_type, model, 和参数字典
    """
    specs = _load_specs()
    if not specs:
        return []

    results = []

    # 确定要查询的产品类型
    if product:
        products_to_query = {product: specs.get(product, {})}
    else:
        products_to_query = specs

    # 解析过滤条件
    filter_field_name, filter_op, filter_value = "", "==", ""
    if filter_field:
        filter_field_name, filter_op, filter_value = _parse_filter(filter_field)

    for prod_type, models in products_to_query.items():
        if not isinstance(models, dict):
            continue

        for model_name, params in models.items():
            if not isinstance(params, dict):
                continue

            # 型号过滤
            if model and model != model_name:
                continue

            # 字段过滤
            if filter_field_name:
                field_value = str(params.get(filter_field_name, ""))
                if not _matches_filter(field_value, filter_op, filter_value):
                    continue

            results.append({
                "product_type": prod_type,
                "model": model_name,
                **params,
            })

    return results


def build_specs_context(query: str, max_models: int = 10) -> str:
    """
    根据查询构建结构化参数上下文，用于注入 LLM

    Args:
        query: 用户查询
        max_models: 最多返回几个型号

    Returns:
        格式化的参数字符串，如无法匹配产品则返回空字符串
    """
    # 检测查询涉及的产品类型
    query_lower = query.lower()
    specs = _load_specs()
    if not specs:
        return ""

    # 关键词匹配
    product_keywords = {
        "星载智算机": ["智算机", "算力", "gpu", "nx", "g1", "g2", "g3"],
        "星载路由器": ["路由器", "智加", "接口"],
        "星载激光通信机": ["激光通信", "激光"],
    }

    # 检测是否涉及产品参数查询
    is_spec_query = any(
        kw in query_lower
        for kws in product_keywords.values()
        for kw in kws
    )

    if not is_spec_query:
        return ""

    # 收集匹配的产品类型
    matched_products = []
    for prod_type, kws in product_keywords.items():
        if any(kw in query_lower for kw in kws):
            matched_products.append(prod_type)

    if not matched_products:
        # 尝试从查询中提取型号
        model_pattern = re.compile(r"(NX\d|G\d|智加G\d)", re.IGNORECASE)
        model_matches = model_pattern.findall(query)
        if model_matches:
            for prod_type, models in specs.items():
                for model_name in models:
                    if any(m.upper() == model_name.upper() for m in model_matches):
                        matched_products.append(prod_type)

    if not matched_products:
        return ""

    # 构建上下文
    context_parts = ["【产品参数参考】"]
    for prod_type in matched_products:
        models = specs.get(prod_type, {})
        if not isinstance(models, dict) or not models:
            continue
        context_parts.append(f"\n## {prod_type}")
        for i, (model_name, params) in enumerate(models.items()):
            if i >= max_models:
                context_parts.append(f"  ...（共 {len(models)} 款）")
                break
            context_parts.append(f"\n### {model_name}")
            for k, v in params.items():
                if v:  # 只显示有值的参数
                    context_parts.append(f"  {k}: {v}")

    return "\n".join(context_parts)


# ── 全局实例 ──────────────────────────────────────────

_specs_cache: Optional[Dict[str, Any]] = None


def get_specs() -> Dict[str, Any]:
    """获取产品参数字典（带缓存）"""
    global _specs_cache
    if _specs_cache is None:
        _specs_cache = _load_specs()
    return _specs_cache
