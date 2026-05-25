"""
MinerU PDF 解析服务 API

独立的 FastAPI 服务，专门负责 PDF → Markdown 转换

接口:
    POST /parse              上传 PDF → 返回 Markdown
    GET  /parse/{parse_id}   查询已解析的结果
    GET  /health             健康检查

调用示例:
    curl -X POST http://localhost:8003/parse -F "file=@document.pdf"
    curl http://localhost:8003/parse/abc123def456...
    curl http://localhost:8003/health
"""

import hashlib
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings

# MinerU 可用性检测 — 复用 core/ingestion/extractor.py
import core.ingestion.extractor as extractor_module
from core.ingestion.extractor import MinerUPDFProcessor, _ensure_mineru


# ============================================================
# 访问日志配置
# ============================================================

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

ACCESS_LOG = LOG_DIR / "mineru_access.log"

# 访问日志专用 sink: JSON 格式，方便后续检索
logger.add(
    str(ACCESS_LOG),
    format="{message}",
    rotation="50 MB",
    retention="30 days",
    encoding="utf-8",
    filter=lambda record: "extra" in record and record["extra"].get("access_log"),
)


# ============================================================
# 访问日志中间件
# ============================================================


class AccessLogMiddleware(BaseHTTPMiddleware):
    """记录所有外部请求的中间件"""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        client_ip = request.client.host if request.client else "unknown"

        try:
            response: Response = await call_next(request)
            duration_ms = int((time.time() - start) * 1000)

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "client_ip": client_ip,
                "method": request.method,
                "path": str(request.url.path),
                "query": str(request.query_params) or None,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
            logger.bind(access_log=True).info(json.dumps(log_entry, ensure_ascii=False))

            return response

        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "client_ip": client_ip,
                "method": request.method,
                "path": str(request.url.path),
                "status_code": 500,
                "duration_ms": duration_ms,
                "error": str(e),
            }
            logger.bind(access_log=True).info(json.dumps(log_entry, ensure_ascii=False))
            raise


# ============================================================
# 持久化存储
# ============================================================

PARSE_STORAGE_DIR = Path(settings.parse_backup_dir) / "mineru_service"
processor: Optional[MinerUPDFProcessor] = None


# ============================================================
# 请求/响应模型
# ============================================================


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    content: str
    page_count: int
    metadata: Optional[dict] = None


class ParseStatistics(BaseModel):
    total_pages: int
    content_length: int
    parse_time_ms: int


class ParseResponse(BaseModel):
    parse_id: str
    timestamp: str
    success: bool
    file_name: str
    file_size: int
    document: DocumentInfo
    statistics: ParseStatistics


class ParseStatusResponse(BaseModel):
    parse_id: str
    exists: bool
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    files: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    mineru_available: bool


# ============================================================
# Lifespan
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor

    logger.info("=" * 50)
    logger.info("MinerU Parse Service 启动中...")
    PARSE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"存储目录: {PARSE_STORAGE_DIR}")

    # 触发 MinerU 延迟导入（必须在 extractor_module.MINERU_AVAILABLE 判断前执行）
    _ensure_mineru()

    if extractor_module.MINERU_AVAILABLE:
        processor = MinerUPDFProcessor()
        logger.success("MinerU 处理器就绪")
    else:
        logger.warning("MinerU 不可用，服务将返回错误")
    logger.info("=" * 50)

    yield
    logger.info("MinerU Parse Service 关闭")


# ============================================================
# 应用
# ============================================================

app = FastAPI(
    title="MinerU Parse Service",
    description="PDF → Markdown 解析服务",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(AccessLogMiddleware)


# ============================================================
# 端点
# ============================================================


@app.get("/")
async def root():
    return {
        "service": "MinerU Parse Service",
        "version": "1.0.0",
        "mineru_available": extractor_module.MINERU_AVAILABLE,
        "storage_dir": str(PARSE_STORAGE_DIR),
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if extractor_module.MINERU_AVAILABLE else "unavailable",
        mineru_available=extractor_module.MINERU_AVAILABLE,
    )


@app.post("/parse", response_model=ParseResponse)
async def parse_pdf(
    file: UploadFile = File(..., description="PDF 文件"),
):
    """
    上传 PDF → 解析为 Markdown，文件持久化保存
    """
    if processor is None:
        raise HTTPException(status_code=503, detail="MinerU 不可用")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    file_content = await file.read()
    file_size = len(file_content)

    if file_size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件大小不能超过 100MB")

    # 基于文件内容 MD5 → 同文件 = 同 parse_id，幂等
    parse_id = hashlib.md5(file_content).hexdigest()
    timestamp = datetime.now().isoformat()

    # 工作目录: parse_backup_dir/mineru_service/{md5}/
    work_dir = PARSE_STORAGE_DIR / parse_id

    # ---- 缓存命中: 已解析过则直接返回 ----
    md_path = work_dir / f"{Path(file.filename).stem}.md"
    if md_path.exists():
        cached_md = md_path.read_text(encoding="utf-8")
        page_count = _estimate_pages(cached_md)

        logger.info(f"[MinerU] 命中缓存: {file.filename}, parse_id={parse_id}")

        return ParseResponse(
            parse_id=parse_id,
            timestamp=timestamp,
            success=True,
            file_name=file.filename,
            file_size=file_size,
            document=DocumentInfo(
                doc_id=parse_id,
                title=file.filename,
                content=cached_md,
                page_count=page_count,
                metadata={"cached": True},
            ),
            statistics=ParseStatistics(
                total_pages=page_count,
                content_length=len(cached_md),
                parse_time_ms=0,
            ),
        )

    # ---- 新文件: 解析 + 持久化 ----
    logger.info(f"[MinerU] 开始解析: {file.filename}, parse_id={parse_id}")
    start = time.time()

    work_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始 PDF
    pdf_path = work_dir / file.filename
    pdf_path.write_bytes(file_content)

    # MinerU 输出目录
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        md_content = processor.process_pdf(
            pdf_path=str(pdf_path),
            output_dir=str(output_dir),
        )

        # 额外保存一份 Markdown 到工作目录根
        md_path.write_text(md_content, encoding="utf-8")

        parse_time_ms = int((time.time() - start) * 1000)
        page_count = _estimate_pages(md_content)

        logger.success(
            f"[MinerU] 解析完成: parse_id={parse_id}, "
            f"pages={page_count}, time={parse_time_ms}ms"
        )

        return ParseResponse(
            parse_id=parse_id,
            timestamp=timestamp,
            success=True,
            file_name=file.filename,
            file_size=file_size,
            document=DocumentInfo(
                doc_id=parse_id,
                title=file.filename,
                content=md_content,
                page_count=page_count,
                metadata={"processor": "MinerU"},
            ),
            statistics=ParseStatistics(
                total_pages=page_count,
                content_length=len(md_content),
                parse_time_ms=parse_time_ms,
            ),
        )

    except Exception as e:
        logger.error(f"[MinerU] 解析失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF 解析失败: {str(e)}")


@app.get("/parse/{parse_id}", response_model=ParseStatusResponse)
async def get_parse_result(parse_id: str):
    """查询已解析的结果，parse_id 为文件 MD5"""
    work_dir = PARSE_STORAGE_DIR / parse_id

    if not work_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"未找到: {parse_id}")

    pdf_files = list(work_dir.glob("*.pdf"))
    md_files = list(work_dir.glob("*.md"))

    pdf_path = pdf_files[0] if pdf_files else None
    md_path = md_files[0] if md_files else None

    page_count = None
    if md_path and md_path.exists():
        page_count = _estimate_pages(md_path.read_text(encoding="utf-8"))

    return ParseStatusResponse(
        parse_id=parse_id,
        exists=True,
        file_name=pdf_path.name if pdf_path else None,
        file_size=pdf_path.stat().st_size if pdf_path and pdf_path.exists() else None,
        page_count=page_count,
        files={
            "pdf": str(pdf_path) if pdf_path else None,
            "markdown": str(md_path) if md_path else None,
            "work_dir": str(work_dir),
        },
    )


# ============================================================
# 工具函数
# ============================================================


def _estimate_pages(md_content: str) -> int:
    """根据 Markdown 内容估算页数"""
    lines = md_content.split("\n")
    page_breaks = [
        i for i, line in enumerate(lines) if line.strip() in ("---", "***", "___")
    ]
    return len(page_breaks) + 1 if page_breaks else max(1, len(lines) // 50)


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.mineru:app",
        host="0.0.0.0",
        port=settings.mineru_port,
        reload=False,
        workers=1,
        log_level="info",
    )
