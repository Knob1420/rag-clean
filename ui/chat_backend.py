"""
RAG 知识库前端 - 后端 API 通信封装
"""

import httpx
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger


class ChatBackend:
    """封装与后端API的通信"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def chat(
        self,
        query: str,
        top_k: int = 10,
        use_rewrite: bool = True,
        use_rerank: bool = True,
        rerank_top_k: Optional[int] = 5,
    ) -> Dict:
        """
        发送聊天请求到后端

        Args:
            query: 用户问题
            top_k: 检索返回的chunk数量
            use_rewrite: 是否使用查询改写
            use_rerank: 是否使用重排序
            rerank_top_k: Rerank后保留数量

        Returns:
            包含 answer, sources, usage 等字段的字典
        """
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                request_data = {
                    "query": query,
                    "top_k": top_k,
                    "use_rewrite": use_rewrite,
                    "use_rerank": use_rerank,
                }
                if rerank_top_k is not None:
                    request_data["rerank_top_k"] = rerank_top_k

                response = await client.post(
                    f"{self.base_url}/api/v1/chat/completions",
                    json=request_data,
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Chat success: query='{query[:50]}...'")
                return result

        except httpx.TimeoutException:
            logger.error(f"Chat timeout: query='{query[:50]}...'")
            return {"error": "请求超时，请检查网络连接", "error_type": "timeout"}

        except httpx.HTTPStatusError as e:
            logger.error(f"Chat HTTP error: {e.response.status_code}")
            if e.response.status_code == 404:
                return {
                    "error": "未找到相关内容，请尝试其他问题",
                    "error_type": "not_found",
                }
            elif e.response.status_code >= 500:
                return {"error": "服务器错误，请稍后重试", "error_type": "server_error"}
            return {
                "error": f"请求失败: {e.response.status_code}",
                "error_type": "http_error",
            }

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {"error": f"发生错误: {str(e)}", "error_type": "unknown"}

    async def list_documents(
        self,
        page: int = 1,
        page_size: int = 50,
        search: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> Dict:
        """
        获取文档列表

        Args:
            page: 页码
            page_size: 每页数量
            search: 搜索关键词
            doc_type: 文档类型过滤

        Returns:
            包含文档列表的字典
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {"page": page, "page_size": page_size}
                if search:
                    params["search"] = search
                if doc_type:
                    params["doc_type"] = doc_type

                response = await client.get(
                    f"{self.base_url}/api/v1/documents", params=params
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"List documents error: {e}")
            return {"error": str(e), "documents": [], "total": 0}

    async def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        获取文档详情

        Args:
            doc_id: 文档ID

        Returns:
            文档详情字典，失败返回 None
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/documents/{doc_id}"
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Document not found: {doc_id}")
                return None
            logger.error(f"Get document error: {e}")
            return None

        except Exception as e:
            logger.error(f"Get document error: {e}")
            return None

    async def search(
        self,
        query: str,
        top_k: int = 10,
        use_rerank: bool = True,
    ) -> Dict:
        """
        纯检索（不调用 LLM 生成）

        Args:
            query: 检索查询
            top_k: 返回数量
            use_rerank: 是否使用 Rerank

        Returns:
            包含 query, total, chunks, timing 的字典
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/search",
                    json={
                        "query": query,
                        "top_k": top_k,
                        "use_rerank": use_rerank,
                    },
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Search HTTP error: {e.response.status_code}")
            return {"error": f"检索失败: {e.response.status_code}", "chunks": [], "total": 0}

        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"error": f"检索错误: {str(e)}", "chunks": [], "total": 0}

    async def parse_pdf(
        self,
        file_path: str,
    ) -> Dict:
        """
        调用 MinerU 服务解析 PDF → Markdown

        Args:
            file_path: PDF 文件本地路径

        Returns:
            包含 parse_id, document(content/title/page_count), statistics 的字典
        """
        from config import settings
        mineru_url = f"http://localhost:{settings.mineru_port}"

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                with open(file_path, "rb") as f:
                    response = await client.post(
                        f"{mineru_url}/parse",
                        files={"file": (Path(file_path).name, f, "application/pdf")},
                    )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            detail = e.response.text
            logger.error(f"Parse HTTP error: {e.response.status_code} {detail}")
            return {"error": f"解析失败 ({e.response.status_code}): {detail}"}

        except httpx.ConnectError:
            logger.error("Parse service unavailable")
            return {"error": "MinerU 服务未启动，请先启动 MinerU 服务 (python run.py --mineru)"}

        except Exception as e:
            logger.error(f"Parse error: {e}")
            return {"error": f"解析错误: {str(e)}"}

    async def get_parse_result(self, parse_id: str) -> Dict:
        """
        查询 PDF 解析结果

        Args:
            parse_id: 解析 ID（文件 MD5）

        Returns:
            包含 parse_id, exists, file_name, page_count 的字典
        """
        from config import settings
        mineru_url = f"http://localhost:{settings.mineru_port}"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{mineru_url}/parse/{parse_id}")
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"error": f"未找到解析记录: {parse_id}"}
            return {"error": f"查询失败: {e.response.status_code}"}

        except Exception as e:
            logger.error(f"Get parse result error: {e}")
            return {"error": str(e)}

    async def check_health(self) -> bool:
        """
        检查后端健康状态

        Returns:
            True 如果后端正常，否则 False
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False


# 全局实例
default_backend = ChatBackend()
