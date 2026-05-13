"""
RAG API 服务 — FastAPI 接口

提供检索和 RAG 问答端点：
- POST /api/v1/chat/completions — 完整 RAG 流程（rewrite → retrieve → rerank → generate）
- POST /api/v1/search — 纯检索
- GET /health — 健康检查

基于 rag-knowledge-base api/main.py，移除：
- user_roles 参数
- chat_logger 依赖（用 loguru 代替）
- 文档上传/管理端点（rag-clean 有自己的 pipeline）
"""

import json
import time
from typing import Optional, List, Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from config import settings
from core.pipeline.simple_pipeline import SimplePipeline
from core.retrieve.retrieval_models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SourceInfo,
    TokenUsage,
)
from core.products.specs_service import query_specs, get_specs
from store import get_store


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="RAG Clean API",
    description="RAG 检索与问答 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 辅助函数
# ============================================================


def chunks_to_sources(chunks: list) -> list:
    """将检索分块转换为来源信息（含文档标题）"""
    # chunk.doc_title 已在 RetrievalService 中从 ES 填充，直接使用
    sources = []
    for chunk in chunks:
        snippet = (
            chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
        )
        sources.append(
            SourceInfo(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_name=chunk.doc_title or "",
                score=chunk.score,
                snippet=snippet,
            )
        )
    return sources


# ============================================================
# 健康检查
# ============================================================


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "RAG Clean API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/v1/chat/completions": "RAG 问答",
            "POST /api/v1/search": "检索",
            "POST /api/v1/parse": "PDF 解析（代理到 MinerU）",
            "GET /api/v1/documents": "文档列表",
            "GET /health": "健康检查",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    services_status = {}

    # 检查 Elasticsearch
    try:
        store = get_store()
        store.es.info()
        services_status["elasticsearch"] = "ok"
    except Exception as e:
        services_status["elasticsearch"] = f"error: {str(e)}"

    # 检查 Embedding 服务
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"http://localhost:{settings.embedding_port}/health")
            services_status["embedding_service"] = (
                "ok" if response.status_code == 200 else "error"
            )
    except Exception as e:
        services_status["embedding_service"] = f"error: {str(e)}"

    # 检查 Rerank 服务
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"http://localhost:{settings.rerank_port}/health")
            services_status["rerank_service"] = (
                "ok" if response.status_code == 200 else "error"
            )
    except Exception as e:
        services_status["rerank_service"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in services_status.values())

    return HealthResponse(
        status="healthy" if all_ok else "degraded", services=services_status
    )


# ============================================================
# RAG 问答接口
# ============================================================


@app.post("/api/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    RAG 知识库问答接口

    流程（quick 模式）：BM25 + Vector + RRF → Rerank → Parent Expand → Spec → LLM 生成
    流程（agent 模式）：ReAct Agent 自主推理检索 → 生成回答
    """
    timing = {}
    frontend_time = time.time()

    try:
        if request.mode == "agent":
            # ---- Agent 智能推理模式 ----
            from core.agent.react_agent import ReActAgent

            agent = ReActAgent(max_iterations=request.max_iterations)
            react_result = agent.run(query=request.query)

            answer = react_result.answer
            sources = chunks_to_sources(react_result.chunks)
            usage = react_result.usage or TokenUsage(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            )
            timing = react_result.timing
            timing["total"] = time.time() - frontend_time

            if not answer:
                raise HTTPException(
                    status_code=404,
                    detail="Agent 未能生成有效回答，请尝试其他问题或切换到快速问答模式",
                )

            logger.info(
                f"Agent问答成功: query='{request.query}', "
                f"iterations={react_result.total_iterations}, "
                f"reason={react_result.terminated_reason}, "
                f"chunks={len(sources)}, "
                f"time={timing.get('total', 0):.1f}s"
            )

            return ChatResponse(
                answer=answer,
                sources=sources,
                time=timing,
                usage=usage,
                chunks_count=len(sources),
            )

        # ---- Quick 快速问答模式（SimplePipeline） ----
        pipeline = SimplePipeline()

        # ---- BM25 + Vector + RRF + Spec 检索 ----
        pipeline_result = pipeline.run(
            query=request.query,
            top_k=request.top_k,
            use_hyde=request.use_hyde,
            use_rerank=request.use_rerank,
            rerank_top_k=request.rerank_top_k,
        )
        timing = pipeline_result.timing
        timing["total"] = time.time() - frontend_time

        # ---- LLM 生成 ----
        chunks = pipeline_result.chunks
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="未找到相关内容，请尝试其他问题或上传相关文档",
            )

        generation_svc = get_generation_service()
        answer, usage = generation_svc.generate(
            query=request.query,
            chunks=chunks,
            spec_context=pipeline_result.spec_context or "",
        )

        if not answer:
            raise HTTPException(
                status_code=404,
                detail="未找到相关内容，请尝试其他问题或上传相关文档",
            )

        # 构建响应
        sources = chunks_to_sources(chunks)

        logger.info(
            f"问答成功: query='{request.query}', "
            f"use_hyde={request.use_hyde}, "
            f"chunks={len(sources)}, "
            f"time={timing.get('total', 0):.1f}s"
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            time=timing,
            usage=usage,
            chunks_count=len(sources),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"问答失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


# ============================================================
# 流式 RAG 问答接口 (SSE)
# ============================================================


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式 RAG 问答 — SSE 端点

    流程与 chat_completions 完全一致：
    - quick 模式：pipeline.run(use_generation=True) 完成后流式推送
    - agent 模式：agent.run() 完成后流式推送

    事件格式:
    - event: sources  data: {sources: [...]}
    - event: steps    data: {steps: [{action, duration}, ...]}  — 仅 agent 模式
    - event: token    data: {content: "..."}
    - event: done     data: {usage: {...}, time: {...}}
    - event: error    data: {error: "..."}
    """
    async def event_stream():
        timing = {}
        frontend_time = time.time()

        try:
            if request.mode == "agent":
                # ---- Agent 智能推理模式（真正流式） ----
                from core.agent.react_agent import ReActAgent

                agent = ReActAgent(max_iterations=request.max_iterations)

                accumulated_sources = []

                for event in agent.run_stream(query=request.query):
                    if event.event_type == "step_start":
                        yield f"event: step_start\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"

                    elif event.event_type == "step_end":
                        yield f"event: step_end\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"

                    elif event.event_type == "answer_token":
                        yield f"event: token\ndata: {json.dumps({'content': event.data.get('content', '')}, ensure_ascii=False)}\n\n"

                    elif event.event_type == "done":
                        # done 事件中收集 sources
                        chunks = agent.tool_executor.accumulated_chunks
                        sources = chunks_to_sources(chunks)
                        sources_data = [s.model_dump() for s in sources]
                        if sources_data:
                            yield f"event: sources\ndata: {json.dumps({'sources': sources_data}, ensure_ascii=False)}\n\n"

                        done_data = {
                            **event.data,
                            "chunks_count": len(chunks),
                        }
                        yield f"event: done\ndata: {json.dumps(done_data, ensure_ascii=False)}\n\n"

                    elif event.event_type == "error":
                        yield f"event: error\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"

            else:
                # ---- Quick 快速问答模式（SimplePipeline + 真流式） ----
                pipeline = SimplePipeline()

                # ---- 1. 检索阶段 ----
                pipeline_result = pipeline.run(
                    query=request.query,
                    top_k=request.top_k,
                    use_hyde=request.use_hyde,
                    use_rerank=request.use_rerank,
                    rerank_top_k=request.rerank_top_k,
                )
                timing = pipeline_result.timing

                chunks = pipeline_result.chunks

                if not chunks:
                    yield f"event: error\ndata: {json.dumps({'error': '未找到相关内容'}, ensure_ascii=False)}\n\n"
                    return

                # ---- 2. 发送 sources 事件 ----
                sources = chunks_to_sources(chunks)
                sources_data = [s.model_dump() for s in sources]
                yield f"event: sources\ndata: {json.dumps({'sources': sources_data}, ensure_ascii=False)}\n\n"

                # ---- 3. 流式生成回答（逐 token 推送） ----
                from core.generation.generation import get_generation_service

                generation_svc = get_generation_service()
                gen_start = time.time()

                async for token in generation_svc.async_generate_stream(
                    query=request.query,
                    chunks=chunks,
                    spec_context=pipeline_result.spec_context or "",
                ):
                    yield f"event: token\ndata: {json.dumps({'content': token}, ensure_ascii=False)}\n\n"

                timing["generation"] = time.time() - gen_start
                timing["total"] = time.time() - frontend_time

                # ---- 4. 发送 done 事件 ----
                yield f"event: done\ndata: {json.dumps({'time': timing, 'chunks_count': len(chunks)}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"流式问答失败: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 文档管理接口
# ============================================================


@app.get("/api/v1/documents")
async def list_documents(page: int = 1, page_size: int = 20):
    """文档列表（分页）"""
    try:
        store = get_store()
        result = store.list_documents(page=page, page_size=page_size)
        return result
    except Exception as e:
        logger.error(f"文档列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


# ============================================================
# PDF 解析代理接口
# ============================================================


@app.post("/api/v1/parse")
async def parse_pdf(file: UploadFile = File(..., description="PDF 文件")):
    """
    PDF 解析代理 — 转发到 MinerU 服务 (port 8003)

    外部只需调用此接口，无需直连 MinerU 服务。
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    file_content = await file.read()
    if len(file_content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件大小不能超过 100MB")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"http://localhost:{settings.mineru_port}/parse",
                files={"file": (file.filename, file_content, "application/pdf")},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="MinerU 服务未启动")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


# ============================================================
# 检索接口
# ============================================================


@app.post("/api/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """检索接口 — 使用 SimplePipeline（BM25 + Vector + RRF + Spec）"""
    try:
        pipeline = SimplePipeline()

        # ---- SimplePipeline 检索 ----
        pipeline_result = pipeline.run(
            query=request.query,
            top_k=request.top_k,
            use_hyde=request.use_hyde,
            use_rerank=request.use_rerank,
            rerank_top_k=request.rerank_top_k,
        )

        chunks = pipeline_result.chunks

        # 转换 chunk 为 dict（doc_title 已在 pipeline 解析时填充）
        chunks_dict = []
        for chunk in chunks:
            chunks_dict.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title or "",
                    "content": (
                        chunk.content[:300] + "..."
                        if len(chunk.content) > 300
                        else chunk.content
                    ),
                    "chunk_type": chunk.chunk_type,
                    "score": chunk.score,
                }
            )

        return SearchResponse(
            query=request.query,
            total=len(chunks),
            chunks=chunks_dict,
            timing=pipeline_result.timing,
        )

    except Exception as e:
        logger.error(f"检索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


# ============================================================
# 产品参数查询
# ============================================================


@app.get("/api/v1/products/specs")
async def get_product_specs(
    product: Optional[str] = None,
    model: Optional[str] = None,
    entity: Optional[str] = None,
    filter: Optional[str] = None,
):
    """
    产品参数查询接口

    Args:
        product: 产品类型（星载智算机 / 星载路由器 / 星载激光通信机）
        model: 型号（如 NX1, G1, 智加G3）
        entity: 实体（如 G1、NX1），通过 query_products 智能匹配
        filter: 过滤条件，支持 <, <=, >, >=, ==（如 "重量<3kg", "算力>=10"）

    Examples:
        GET /api/v1/products/specs                           # 返回所有产品
        GET /api/v1/products/specs?product=星载智算机         # 返回智算机全系
        GET /api/v1/products/specs?entity=G1                 # 通过实体查询（如 G1、NX1）
        GET /api/v1/products/specs?filter=重量<3kg           # 返回重量<3kg的型号
    """
    try:
        if entity:
            # 实体查询（通过 query_products 智能匹配产品类型和型号）
            from core.products.spec_matcher import query_products

            results = query_products(
                target_models=[entity],
                required_fields=[],
                numerical_constraints={},
            )
        else:
            results = query_specs(product=product, model=model, filter_field=filter)

        return {
            "total": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"产品参数查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/api/v1/products")
async def list_products():
    """返回所有产品类型和型号列表"""
    specs = get_specs()
    return {
        "products": [
            {
                "type": prod_type,
                "models": list(models.keys()) if isinstance(models, dict) else [],
            }
            for prod_type, models in specs.items()
        ]
    }


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
