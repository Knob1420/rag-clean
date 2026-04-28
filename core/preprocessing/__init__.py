"""
core.preprocessing — MD 文档预处理流水线

步骤1：清洗 + 分片（cleaner_ext.py + chunker_ext.py）
步骤2：NER + 共现聚类（entity_extractor.py）
步骤3：四大结构化抽取（struct_extractor.py）
步骤4：轻量本体构建（ontology_builder.py）
步骤5：固化存储（storage.py）
"""

from core.preprocessing.cleaner_ext import TextCleaner, clean_and_normalize
from core.preprocessing.chunker_ext import SmartChunker
from core.preprocessing.entity_extractor import extract_entities, save_results
from core.preprocessing.struct_extractor import (
    extract_product_params,
    extract_cooperation,
    save_results as save_struct_results,
)
from core.preprocessing.ontology_builder import (
    load_entity_raw,
    load_product_params,
    load_cooperation,
    build_graph,
    save_graph,
    load_graph,
    build_org_alias_union,
)

__all__ = [
    "TextCleaner",
    "clean_and_normalize",
    "SmartChunker",
    "extract_entities",
    "save_results",
    "extract_product_params",
    "extract_cooperation",
    "save_struct_results",
    "load_entity_raw",
    "load_product_params",
    "load_cooperation",
    "build_graph",
    "save_graph",
    "load_graph",
    "build_org_alias_union",
]
