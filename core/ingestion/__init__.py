"""
数据处理层 — 文档分块、提取、清洗、Pipeline
"""

from core.ingestion.document_processor import process_document, process_markdown
from core.ingestion.chunker import SmartChunker
from core.ingestion.extractor import (
    detect_format,
    convert_to_markdown,
    extract,
    MinerUPDFProcessor,
    SUPPORTED_FORMATS,
)
from core.ingestion.cleaner import TextCleaner, clean_text

__all__ = [
    # Pipeline
    "process_document",
    "process_markdown",
    # 分块
    "SmartChunker",
    # 提取与转换
    "extract",
    "convert_to_markdown",
    "detect_format",
    "MinerUPDFProcessor",
    "SUPPORTED_FORMATS",
    # 清洗
    "TextCleaner",
    "clean_text",
]
