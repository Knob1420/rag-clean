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
    es_index_chunks: str = "rag_chunk_0509"
    # es_index_documents: str = "rag_documents"

    # ========== LLM ==========
    deepseek_api_key: str = ""
    deepseek_base_url: str = "http://i-nhi.zhejianglab.org/maas/v1"
    deepseek_model: str = "deepseek-v3.2-bf16"
    # deepseek_api_key: str = "Qwen3-30B-A3B"
    # deepseek_base_url: str = "http://10.107.207.88:8081/v1"
    # deepseek_model: str = "Qwen3-30B-A3B"
    deepseek_r2_model: str = "deepseek-v3.2-bf16"

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
    rerank_score_threshold: float = 0.0  # rerank 最低分数阈值，低于此值的结果被过滤

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

    # ========== MinerU 3.x ==========
    # 通过 subprocess 调 mineru CLI，独立 conda env 隔离 vllm/torch 依赖
    mineru3_env: str = "memory"              # conda env name
    mineru3_gpu: str = "2"                   # CUDA_VISIBLE_DEVICES（配合 PCI_BUS_ID）
    mineru3_backend: str = "hybrid-engine"   # pipeline / vlm-engine / hybrid-engine
    mineru3_effort: str = "medium"           # medium / high（仅 hybrid-* 生效）
    mineru3_lang: str = "ch"                 # OCR 语言
    # 常驻服务（推荐）：先起 mineru-api 服务，convert 自动复用模型，单文件 ~15s
    # 空 api_url = 走冷启动 CLI（每次重载模型 ~58s）
    mineru3_host: str = "127.0.0.1"
    mineru3_port: int = 8004
    mineru3_api_url: str = "http://127.0.0.1:8004"  # 设空 = 强制冷启动

    class Config:
        env_file = str(_PROJECT_ROOT / ".env")
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
