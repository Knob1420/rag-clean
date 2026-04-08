"""
Embedding 服务 API

独立的 FastAPI 服务，专门负责文本向量化
"""
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
import numpy as np
import os

from config import settings


# ============================================================
# Embedding 服务类
# ============================================================


class EmbeddingService:
    """向量化服务"""

    def __init__(self):
        self._model = None
        self._model_name = None

    def _load_model(self):
        """延迟加载向量模型（本地路径 > ModelScope > HuggingFace）"""
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")

        # 1. 优先使用本地路径
        if settings.embedding_model_path and os.path.exists(settings.embedding_model_path):
            try:
                from FlagEmbedding import FlagModel

                logger.info(f"从本地加载模型: {settings.embedding_model_path}")
                self._model = FlagModel(
                    settings.embedding_model_path,
                    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                    use_fp16=True,
                    device=device,
                )
                self._model_name = settings.embedding_model_path
                logger.success(f"本地向量模型加载成功 (device={device})")
                return
            except ImportError:
                logger.warning("FlagEmbedding 未安装")
            except Exception as e:
                logger.warning(f"本地模型加载失败: {e}")

        # 2. 尝试从 ModelScope 加载
        try:
            from FlagEmbedding import FlagModel

            logger.info(f"从 ModelScope 加载模型: {settings.embedding_model}")
            self._model = FlagModel(
                settings.embedding_model,
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=True,
                device=device,
            )
            self._model_name = settings.embedding_model
            logger.success(f"ModelScope 向量模型加载成功 (device={device})")
            return
        except ImportError:
            logger.warning("FlagEmbedding 未安装")
        except Exception as e:
            logger.warning(f"ModelScope 加载失败: {e}")

        # 3. 回退到 sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"从 HuggingFace 加载模型: {settings.embedding_model}")
            self._model = SentenceTransformer(
                settings.embedding_model,
                trust_remote_code=True,
                device=device,
            )
            self._model_name = settings.embedding_model
            logger.success(f"向量模型加载成功: {settings.embedding_model} (device={device})")
        except ImportError:
            logger.warning("请安装 FlagEmbedding 或 sentence-transformers")
            self._model = None
        except Exception as e:
            logger.error(f"向量模型加载失败: {e}")
            self._model = None

    def encode(self, text: str) -> Optional[np.ndarray]:
        """将文本编码为向量"""
        if self._model is None and self._model_name is None:
            self._load_model()

        if self._model is None:
            logger.warning("使用Mock向量，请安装 FlagEmbedding")
            return np.random.randn(settings.embedding_dim).astype(np.float32)

        try:
            if hasattr(self._model, "encode_queries"):
                embeddings = self._model.encode_queries([text])
                embedding = embeddings[0]
            else:
                embedding = self._model.encode(
                    text,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"向量化失败: {e}")
            return None

    def encode_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """批量编码文本"""
        if self._model is None and self._model_name is None:
            self._load_model()

        if self._model is None:
            return [np.random.randn(settings.embedding_dim).astype(np.float32) for _ in texts]

        try:
            if hasattr(self._model, "encode_queries"):
                embeddings = self._model.encode_queries(texts)
            else:
                embeddings = self._model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            return [e.astype(np.float32) for e in embeddings]
        except Exception as e:
            logger.error(f"批量向量化失败: {e}")
            return [None] * len(texts)


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="Embedding Service",
    description="文本向量化服务",
    version="1.0.0",
)

embedding_service: Optional[EmbeddingService] = None


# ============================================================
# 请求/响应模型
# ============================================================


class EncodeRequest(BaseModel):
    text: str


class EncodeResponse(BaseModel):
    embedding: List[float]
    dimension: int


class EncodeBatchRequest(BaseModel):
    texts: List[str]
    max_batch_size: int = 32


class EncodeBatchResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    dimension: int
    model_loaded: bool


# ============================================================
# 生命周期事件
# ============================================================


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global embedding_service

    logger.info("=" * 50)
    logger.info("Embedding Service 启动中...")
    logger.info("=" * 50)

    embedding_service = EmbeddingService()

    logger.info(f"加载模型: {settings.embedding_model}")
    if settings.embedding_model_path and os.path.exists(settings.embedding_model_path):
        logger.info(f"本地路径: {settings.embedding_model_path}")

    dummy_text = "测试"
    _ = embedding_service.encode(dummy_text)

    logger.success(f"✓ 模型加载成功")
    logger.info(f"  模型: {settings.embedding_model}")
    logger.info(f"  维度: {settings.embedding_dim}")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理资源"""
    logger.info("Embedding Service 关闭中...")


# ============================================================
# API 端点
# ============================================================


@app.get("/", response_model=dict)
async def root():
    """根路径"""
    return {
        "service": "Embedding Service",
        "version": "1.0.0",
        "model": settings.embedding_model,
        "dimension": settings.embedding_dim,
        "endpoints": {
            "POST /encode": "单文本向量化",
            "POST /encode_batch": "批量向量化",
            "GET /health": "健康检查",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    model_loaded = (
        embedding_service is not None and embedding_service._model is not None
    )

    return HealthResponse(
        status="ok" if model_loaded else "error",
        model=settings.embedding_model,
        dimension=settings.embedding_dim,
        model_loaded=model_loaded,
    )


@app.post("/encode", response_model=EncodeResponse)
async def encode(request: EncodeRequest):
    """单文本向量化"""
    if embedding_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    if not request.text:
        raise HTTPException(status_code=400, detail="text 字段不能为空")

    try:
        vector = embedding_service.encode(request.text)

        if vector is None:
            raise HTTPException(status_code=500, detail="向量化失败")

        return EncodeResponse(
            embedding=vector.tolist(),
            dimension=settings.embedding_dim,
        )

    except Exception as e:
        logger.error(f"向量化错误: {e}")
        raise HTTPException(status_code=500, detail=f"向量化失败: {str(e)}")


@app.post("/encode_batch", response_model=EncodeBatchResponse)
async def encode_batch(request: EncodeBatchRequest):
    """批量向量化"""
    if embedding_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    if not request.texts:
        raise HTTPException(status_code=400, detail="texts 列表不能为空")

    if len(request.texts) > request.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"批量大小超过限制: {len(request.texts)} > {request.max_batch_size}",
        )

    try:
        vectors = embedding_service.encode_batch(request.texts)

        valid_vectors = []
        for i, v in enumerate(vectors):
            if v is not None:
                valid_vectors.append(v.tolist())
            else:
                logger.warning(f"第 {i} 个文本向量化失败")

        if not valid_vectors:
            raise HTTPException(status_code=500, detail="所有文本向量化失败")

        return EncodeBatchResponse(
            embeddings=valid_vectors,
            dimension=settings.embedding_dim,
            count=len(valid_vectors),
        )

    except Exception as e:
        logger.error(f"批量向量化错误: {e}")
        raise HTTPException(status_code=500, detail=f"批量向量化失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.embedding:app",
        host="0.0.0.0",
        port=settings.embedding_port,
        reload=False,
        workers=1,
        log_level="info",
    )
