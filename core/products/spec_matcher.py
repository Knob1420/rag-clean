"""
结构化产品参数查询服务

根据 extracted_params 中的 target_models、required_fields、numerical_constraints
从 products_specs.json 中查询匹配的产品的结构化参数。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


# 字段名别名映射：用户可能使用的字段名 -> products_specs.json 标准字段名
FIELD_ALIASES: Dict[str, str] = {
    # 重量
    "重量": "重量",
    "weight": "重量",
    "质量": "重量",
    # 功耗
    "功耗": "功耗",
    "power": "功耗",
    "用电量": "功耗",
    "power_consumption": "功耗",
    # 算力
    "算力": "算力",
    "compute": "算力",
    "tops": "算力",
    "计算能力": "算力",
    "性能": "算力",
    # 尺寸
    "尺寸": "尺寸",
    "size": "尺寸",
    "大小": "尺寸",
    "dimension": "尺寸",
    # 内存
    "内存": "内存",
    "memory": "内存",
    "ram": "内存",
    # 存储
    "存储": "存储",
    "storage": "存储",
    "硬盘": "存储",
    "disk": "存储",
    # 接口/对外接口
    "接口": "对外接口",
    "interface": "对外接口",
    "interfaces": "对外接口",
    "对外接口": "对外接口",
    "接口类型": "对外接口",
    # 合作单位
    "合作单位": "合作单位",
    "partner": "合作单位",
    "厂商": "合作单位",
    "生产商": "合作单位",
    # 架构
    "架构": "架构",
    "architecture": "架构",
    # 芯片/核心芯片
    "芯片": "核心芯片",
    "核心芯片": "核心芯片",
    "chip": "核心芯片",
    # 操作系统
    "操作系统": "操作系统",
    "os": "操作系统",
    "系统": "操作系统",
    # 设计寿命
    "寿命": "设计寿命",
    "设计寿命": "设计寿命",
    "lifespan": "设计寿命",
    # 功能
    "功能": "功能",
    "function": "功能",
    "functions": "功能",
    "能力": "功能",
    # 在轨情况
    "在轨情况": "在轨情况",
    "在轨": "在轨情况",
    "发射": "在轨情况",
    "launch": "在轨情况",
    "orbit": "在轨情况",
    # 输入电压
    "电压": "输入电压",
    "input_voltage": "输入电压",
    "voltage": "输入电压",
    # 工作温度
    "工作温度": "工作温度",
    "temperature": "工作温度",
    # 重量约束关键词
    "轻": "重量",
    "轻量": "重量",
    "轻量级": "重量",
}


# 产品型号别名映射：用户可能输入的型号 -> specs中的标准型号
MODEL_ALIASES: Dict[str, str] = {
    # 智加系列
    "国产智算机": "星载智能计算机",
    "星载智算机": "星载智能计算机",
    "智算机": "星载智能计算机",
    "路由器": "星载路由器",
    "卫星路由器": "星载路由器",
    "激光通信机": "星载激光通信机",
    "激光通信": "星载激光通信机",
    "智加G1": "智加G1",
    "G1": "智加G1",
    "智加G2": "智加G2",
    "G2": "智加G2",
    "智加G3": "智加G3",
    "G3": "智加G3",
    "智加NX1": "智加NX1",
    "NX1": "智加NX1",
    "智加NX2": "智加NX2",
    "NX2": "智加NX2",
    "智加NX3": "智加NX3",
    "NX3": "智加NX3",
    "智加NX4": "智加NX4",
    "NX4": "智加NX4",
    "智加X1": "智加X1",
    "X1": "智加X1",
    # 智桥系列
    "智桥R1": "智桥R1",
    "R1": "智桥R1",
    "智桥RH1": "智桥RH1",
    "RH1": "智桥RH1",
    # 智光系列
    "智光-100-T2-100G-Z": "智光-100-T2-100G-Z",
    "智光": "智光-100-T2-100G-Z",
}


def _normalize_field(field: str) -> str:
    """将字段别名映射为标准字段名"""
    return FIELD_ALIASES.get(field, field)


def _normalize_model(model: str) -> str:
    """将型号别名映射为标准型号名"""
    return MODEL_ALIASES.get(model, model)


def _normalize_models(models: List[str]) -> List[str]:
    """标准化型号列表，返回唯一的标准型号"""
    normalized = set()
    for m in models:
        # 先检查是否是完整匹配
        if m in MODEL_ALIASES:
            normalized.add(MODEL_ALIASES[m])
        else:
            # 尝试模糊匹配：G1 -> 智加G1
            for alias, std_name in MODEL_ALIASES.items():
                if alias in m or m in alias:
                    normalized.add(std_name)
                    break
            else:
                # 没有匹配到，保留原值
                normalized.add(m)
    return list(normalized)


def _load_product_specs() -> Dict[str, Any]:
    """懒加载产品参数表"""
    specs_path = Path(__file__).parent.parent.parent / "data" / "products_specs.json"
    if not specs_path.exists():
        return {}
    with open(specs_path, "r", encoding="utf-8") as f:
        return json.load(f)


_PRODUCT_SPECS: Optional[Dict[str, Any]] = None


def get_product_specs() -> Dict[str, Any]:
    global _PRODUCT_SPECS
    if _PRODUCT_SPECS is None:
        _PRODUCT_SPECS = _load_product_specs()
    return _PRODUCT_SPECS


def _parse_weight(weight_str: str) -> Optional[float]:
    """从重量字符串中提取数值（kg）"""
    match = re.search(r"([\d.]+)\s*kg", weight_str)
    if match:
        return float(match.group(1))
    # 处理"不含盖"等情况，取主值
    match = re.search(r"([\d.]+)\s*kg", weight_str.split("（")[0])
    if match:
        return float(match.group(1))
    return None


def _parse_power(power_str: str) -> Optional[float]:
    """从功耗字符串中提取典型功耗（W）"""
    # 匹配 "普通业务93.5W" 或 "峰值128W" 等模式
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*W", power_str)
    if not matches:
        return None
    # 取第一个数值作为典型功耗
    return float(matches[0])


def _parse_tops(tops_str: str) -> Optional[float]:
    """从算力字符串中提取 TOPS 数值"""
    match = re.search(r"(\d+(?:\.\d+)?)\s*TOPS", tops_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def query_products(
    target_models: List[str],
    required_fields: List[str],
    numerical_constraints: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    根据意图查询参数表

    Args:
        target_models: 目标产品型号列表（如 ["智加G1", "智加NX2"]）
        required_fields: 需要的参数字段（如 ["重量", "功耗"]）
        numerical_constraints: 数值约束（如 {"重量": "<3.0"}）

    Returns:
        匹配产品的结构化参数列表
    """
    specs = get_product_specs()
    if not specs:
        logger.warning("[SpecMatcher] 产品参数表为空或不存在")
        return []

    results = []

    # 标准化型号和字段名
    target_models = _normalize_models(target_models)
    normalized_constraints: Dict[str, str] = {}
    for field, constraint in numerical_constraints.items():
        normalized_constraints[_normalize_field(field)] = constraint

    for category, products in specs.items():
        if not isinstance(products, dict):
            continue

        for product_name, params in products.items():
            if not isinstance(params, dict):
                continue

            # 1. 如果指定了 target_models，需要匹配
            if target_models:
                matched = False
                for target in target_models:
                    # 1.1 先检查是否是大类名（JSON 顶层 key）
                    if target == category:
                        matched = True
                        break
                    # 1.2 再检查是否匹配具体产品名
                    if target in product_name or product_name in target:
                        matched = True
                        break
                if not matched:
                    continue

            # 2. 检查数值约束
            violate_constraint = False
            for field, constraint in normalized_constraints.items():
                if field not in params:
                    continue
                value_str = params[field]
                parsed_value: Optional[float] = None

                # 根据字段类型解析数值
                if field in ("重量", "weight"):
                    parsed_value = _parse_weight(value_str)
                elif field in ("功耗", "power"):
                    parsed_value = _parse_power(value_str)
                elif field in ("算力", "compute", "tops"):
                    parsed_value = _parse_tops(value_str)

                if parsed_value is None:
                    continue

                # 解析约束条件
                constraint_match = re.match(r"(<=|>=|<|>|==)\s*([\d.]+)", constraint)
                if not constraint_match:
                    continue

                op, threshold = constraint_match.groups()
                threshold = float(threshold)

                if op == "<=" and not (parsed_value <= threshold):
                    violate_constraint = True
                elif op == ">=" and not (parsed_value >= threshold):
                    violate_constraint = True
                elif op == "<" and not (parsed_value < threshold):
                    violate_constraint = True
                elif op == ">" and not (parsed_value > threshold):
                    violate_constraint = True
                elif op == "==" and not (parsed_value == threshold):
                    violate_constraint = True

                if violate_constraint:
                    break

            if violate_constraint:
                continue

            # 3. 构建结果
            result = {
                "product_name": product_name,
                "category": category,
            }
            # 提取需要的字段（使用标准化字段名）
            for field in required_fields:
                normalized = _normalize_field(field)
                if normalized in params:
                    result[normalized] = params[normalized]
            # 提取所有字段（如果没指定 required_fields）
            if not required_fields:
                result.update(params)
            else:
                # 添加其他可能相关的字段
                for key, value in params.items():
                    if key not in result:
                        result[key] = value

            results.append(result)

    return results


def format_spec_context(query_results: List[Dict[str, Any]], intent: str) -> str:
    """
    将查询结果格式化为可读上下文

    Args:
        query_results: query_products() 返回的结果
        intent: 意图类型

    Returns:
        格式化后的字符串
    """
    if not query_results:
        return "（知识库中无匹配的结构化产品参数）"

    parts = ["【产品参数表查询结果】"]

    for i, product in enumerate(query_results, 1):
        parts.append(f"\n## {i}. {product.get('product_name', '未知产品')}")
        parts.append(f"   分类: {product.get('category', '未知')}")

        # 优先显示关键参数
        key_fields = [
            "合作单位",
            "架构",
            "算力",
            "重量",
            "功耗",
            "尺寸",
            "内存",
            "存储",
        ]
        for field in key_fields:
            if field in product and product[field]:
                parts.append(f"   {field}: {product[field]}")

        # 显示其他字段
        for field, value in product.items():
            if field in ("product_name", "category") or not value:
                continue
            if field not in key_fields:
                parts.append(f"   {field}: {value}")

    return "\n".join(parts)
