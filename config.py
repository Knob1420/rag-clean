"""
配置 — 数据层专用，清理重复项
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

_PROJECT_ROOT = Path(__file__).parent


class Settings(BaseSettings):
    # ========== Elasticsearch ==========
    es_url: str = "http://localhost:9200"
    es_index_chunks: str = "rag_chunks"
    es_index_documents: str = "rag_documents"

    # ========== LLM ==========
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # ========== Embedding ==========
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_model_path: str = (
        "/home/zjlab/Documents/build_LLMs/NLP_course_hf/pretrain_model/BAAI/bge-m3"  # 本地模型路径（优先）
    )

    # ========== Rerank ==========
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_model_path: str = (
        "/home/zjlab/Documents/build_LLMs/NLP_course_hf/pretrain_model/BAAI/bge-reranker-v2-m3"  # 本地模型路径（优先）
    )

    # ========== 检索 ==========
    default_top_k: int = 25
    default_rerank_top_k: int = 12

    # ========== ReAct 多跳推理 ==========
    max_react_hops: int = 3  # 最大推理轮次（含 seed hop 0）
    react_seed_top_k: int = 15  # 首轮种子检索数量
    react_step_top_k: int = 15  # 后续轮次检索数量
    react_consecutive_no_new: int = 1  # 连续无新结果立即停止

    # ========== 服务端口 ==========
    api_port: int = 8000  # 主 API 服务
    embedding_port: int = 8001  # Embedding 服务
    rerank_port: int = 8002  # Rerank 服务
    mineru_port: int = 8003  # MinerU 解析服务

    # ========== 文件存储 ==========
    data_root: str = "./data"
    raw_docs_dir: str = "./data/raw/pending"
    parent_store_dir: str = "./data/parents"  # parent chunk 本地存储
    cache_dir: str = "./data/cache"  # enrichment 结果缓存，防重复消耗 token
    parse_backup_dir: str = "./data/parsed_backups"  # MinerU 解析输出

    class Config:
        env_file = str(_PROJECT_ROOT / ".env")
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
