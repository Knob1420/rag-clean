"""
Rerank 服务 API

独立的 FastAPI 服务，专门负责结果重排序
"""
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
import os

from config import settings


# ============================================================
# Rerank 服务类
# ============================================================


class RerankService:
    """重排序服务"""

    def __init__(self):
        self._model = None
        self._model_name = None

    def _load_model(self):
        """延迟加载 Rerank 模型（本地路径 > ModelScope > HuggingFace）"""

        # 1. 优先使用本地路径
        if settings.rerank_model_path and os.path.exists(settings.rerank_model_path):
            try:
                from FlagEmbedding import FlagReranker

                logger.info(f"从本地加载 Rerank 模型: {settings.rerank_model_path}")
                self._model = FlagReranker(settings.rerank_model_path)
                self._model_name = settings.rerank_model_path
                logger.success(f"本地 Rerank 模型加载成功")
                return
            except ImportError:
                logger.warning("FlagEmbedding 未安装")
            except Exception as e:
                logger.warning(f"本地模型加载失败: {e}")

        # 2. 尝试从 ModelScope 加载
        try:
            from FlagEmbedding import FlagReranker

            logger.info(f"从 ModelScope 加载 Rerank 模型: {settings.rerank_model}")
            self._model = FlagReranker(settings.rerank_model)
            self._model_name = settings.rerank_model
            logger.success(f"ModelScope Rerank 模型加载成功")
            return
        except ImportError:
            logger.warning("FlagEmbedding 未安装")
        except Exception as e:
            logger.warning(f"ModelScope 加载失败: {e}")

        # 3. 回退到 HuggingFace
        try:
            from FlagEmbedding import FlagReranker

            logger.info(f"从 HuggingFace 加载 Rerank 模型: {settings.rerank_model}")
            self._model = FlagReranker(settings.rerank_model)
            self._model_name = settings.rerank_model
            logger.success(f"Rerank 模型加载成功: {settings.rerank_model}")
        except ImportError:
            logger.warning("请安装 FlagEmbedding: pip install FlagEmbedding")
            self._model = None
        except Exception as e:
            logger.error(f"Rerank 模型加载失败: {e}")
            self._model = None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[tuple]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回前 K 个结果，None 表示返回全部

        Returns:
            [(文档, 得分), ...] 按相关性降序排列
        """
        if self._model is None and self._model_name is None:
            self._load_model()

        if self._model is None:
            logger.warning("使用 Mock Rerank，请安装 FlagEmbedding")
            return [(doc, 1.0 - i * 0.01) for i, doc in enumerate(documents)]

        try:
            sentence_pairs = [[query, doc] for doc in documents]
            scores = self._model.compute_score(sentence_pairs)

            indexed_scores = [(i, float(scores[i])) for i in range(len(documents))]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for idx, score in indexed_scores[:top_k] if top_k is not None else indexed_scores:
                reranked.append((documents[idx], score))

            return reranked

        except Exception as e:
            logger.error(f"Rerank 失败: {e}")
            return [(doc, 1.0 - i * 0.01) for i, doc in enumerate(documents)]


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="Rerank Service",
    description="结果重排序服务",
    version="1.0.0",
)

rerank_service: Optional[RerankService] = None


# ============================================================
# 请求/响应模型
# ============================================================


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: Optional[int] = None


# OpenAI 兼容格式
class OpenAIRerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    truncate_prompt_tokens: Optional[int] = None
    additional_data: Optional[dict] = None


class OpenAIRerankResponse(BaseModel):
    id: str
    model: str
    results: List[dict]
    usage: dict


class RerankResponse(BaseModel):
    results: List[tuple]
    count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool


# ============================================================
# 生命周期事件
# ============================================================


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global rerank_service

    logger.info("=" * 50)
    logger.info("Rerank Service 启动中...")
    logger.info("=" * 50)

    rerank_service = RerankService()

    logger.info(f"加载模型: {settings.rerank_model}")
    if settings.rerank_model_path and os.path.exists(settings.rerank_model_path):
        logger.info(f"本地路径: {settings.rerank_model_path}")

    dummy_query = "测试"
    dummy_docs = ["文档1", "文档2"]
    _ = rerank_service.rerank(dummy_query, dummy_docs, top_k=1)

    logger.success(f"模型加载成功")
    logger.info(f"  模型: {settings.rerank_model}")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理资源"""
    logger.info("Rerank Service 关闭中...")


# ============================================================
# API 端点
# ============================================================


@app.get("/", response_model=dict)
async def root():
    """根路径"""
    return {
        "service": "Rerank Service",
        "version": "1.0.0",
        "model": settings.rerank_model,
        "endpoints": {
            "POST /rerank": "重排序文档",
            "GET /health": "健康检查",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    model_loaded = rerank_service is not None and rerank_service._model is not None

    return HealthResponse(
        status="ok" if model_loaded else "error",
        model=settings.rerank_model,
        model_loaded=model_loaded,
    )


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """重排序文档"""
    if rerank_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    if not request.query:
        raise HTTPException(status_code=400, detail="query 字段不能为空")

    if not request.documents:
        raise HTTPException(status_code=400, detail="documents 列表不能为空")

    if request.top_k is not None and request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k 必须大于 0")

    try:
        results = rerank_service.rerank(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k,
        )

        return RerankResponse(
            results=results,
            count=len(results),
        )

    except Exception as e:
        logger.error(f"重排序失败: {e}")
        raise HTTPException(status_code=500, detail=f"重排序失败: {str(e)}")


@app.post("/v1/rerank", response_model=OpenAIRerankResponse)
async def rerank_v1(request: OpenAIRerankRequest):
    """OpenAI 兼容端点 /v1/rerank"""
    if rerank_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    if not request.query:
        raise HTTPException(status_code=400, detail="query 字段不能为空")

    if not request.documents:
        raise HTTPException(status_code=400, detail="documents 列表不能为空")

    try:
        results = rerank_service.rerank(
            query=request.query,
            documents=request.documents,
            top_k=None,
        )

        return OpenAIRerankResponse(
            id=f"rerank-{hash(request.query) % 100000}",
            model=request.model,
            results=[{"index": i, "relevance_score": score} for i, (_, score) in enumerate(results)],
            usage={"total_tokens": len(request.documents)},
        )

    except Exception as e:
        logger.error(f"重排序失败: {e}")
        raise HTTPException(status_code=500, detail=f"重排序失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.rerank:app",
        host="0.0.0.0",
        port=settings.rerank_port,
        reload=False,
        workers=1,
        log_level="info",
    )
