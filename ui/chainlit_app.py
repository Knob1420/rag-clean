"""
RAG 知识库前端 - Chainlit 界面

功能：
- 流式聊天问答（SSE 对接后端）
- Sources 引用展示
- 文档管理（侧边栏）
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import chainlit as cl
import httpx
from loguru import logger

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

BACKEND_URL = "http://localhost:8000"


# ============================================================
# 启动 / 欢迎
# ============================================================


@cl.on_chat_start
async def on_chat_start():
    """会话开始"""
    # 检查后端健康
    healthy = await _check_health()
    if not healthy:
        await cl.Message(
            content="**后端 API 未启动**，请先运行 `python run.py --main`"
        ).send()
        return

    # 显示欢迎信息
    await cl.Message(
        content=(
            "## RAG Knowledge Base\n"
            "基于文档的智能问答系统，支持流式输出和来源引用。\n\n"
            "**使用方式：**\n"
            "- 直接输入问题进行问答\n"
            "- 回答下方会展示检索到的来源引用\n"
            "- 点击侧边栏「文档管理」查看知识库文档\n"
        )
    ).send()

    # 设置默认参数到 session
    cl.user_session.set("chat_history", [])


# ============================================================
# 聊天处理
# ============================================================


@cl.on_message
async def on_message(message: cl.Message):
    """处理用户消息 — 流式调用后端"""
    query = message.content.strip()
    if not query:
        await cl.Message(content="请输入问题").send()
        return

    # 命令处理
    text_lower = query.lower()
    if text_lower in ["/docs", "/documents", "文档管理"]:
        await _show_documents()
        return
    elif text_lower in ["/help", "帮助"]:
        await _show_help()
        return

    # 更新聊天历史
    chat_history: List[Dict[str, str]] = cl.user_session.get("chat_history", [])
    chat_history.append({"role": "user", "content": query})
    cl.user_session.set("chat_history", chat_history)

    # 调用后端流式端点
    msg = cl.Message(content="")
    await msg.send()

    sources_list = []
    full_answer = ""

    try:
        request_data = {
            "query": query,
            "top_k": 10,
            "use_rewrite": True,
            "use_rerank": True,
            "rerank_top_k": 5,
        }

        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/api/v1/chat/stream",
                json=request_data,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # 解析 SSE 事件
                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                    else:
                        continue

                    if event_type == "sources":
                        sources_list = data.get("sources", [])

                    elif event_type == "token":
                        token = data.get("content", "")
                        full_answer += token
                        await msg.stream_token(token)

                    elif event_type == "done":
                        timing = data.get("time", {})
                        gen_time = timing.get("generation", 0)
                        total_time = timing.get("total", 0)
                        chunks_count = data.get("chunks_count", 0)

                        # 添加时间信息
                        footer = (
                            f"\n\n---\n"
                            f"Generation: {gen_time:.2f}s | "
                            f"Total: {total_time:.2f}s | "
                            f"Chunks: {chunks_count}"
                        )
                        await msg.stream_token(footer)

                    elif event_type == "error":
                        error_msg = data.get("error", "未知错误")
                        await msg.stream_token(f"\n\n**Error:** {error_msg}")

        await msg.update()

        # 展示 sources 作为引用
        if sources_list:
            elements = []
            for src in sources_list[:10]:
                doc_name = src.get("doc_name", "Unknown")
                section = src.get("section_title", "")
                score = src.get("score", 0)
                snippet = src.get("snippet", "")

                label = f"{doc_name}"
                if section:
                    label += f" > {section}"
                label += f" ({score:.2f})"

                elements.append(
                    cl.Text(
                        name=label,
                        content=snippet,
                        display="side",
                    )
                )

            # 发送 sources 消息
            sources_msg = cl.Message(
                content=f"**引用来源 ({len(sources_list)})**",
                elements=elements,
            )
            await sources_msg.send()

        # 更新聊天历史
        chat_history.append({"role": "assistant", "content": full_answer})
        cl.user_session.set("chat_history", chat_history)

    except httpx.ConnectError:
        await cl.Message(
            content="**后端 API 未启动**，请先运行 `python run.py --main`"
        ).send()
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        await cl.Message(content=f"**Error:** {str(e)}").send()


# ============================================================
# 文档管理（命令）
# ============================================================


async def _show_documents(page: int = 1, page_size: int = 20):
    """展示文档列表"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/documents",
                params={"page": page, "page_size": page_size},
            )
            response.raise_for_status()
            result = response.json()

        docs = result.get("documents", [])
        total = result.get("total", 0)
        total_pages = max(1, (total + page_size - 1) // page_size)

        if not docs:
            await cl.Message(content="知识库暂无文档").send()
            return

        lines = [
            f"## 文档列表 ({total} 篇, 第 {page}/{total_pages} 页)\n"
        ]
        for doc in docs:
            title = doc.get("title", "Unknown")
            doc_type = doc.get("doc_type", "")
            status = doc.get("status", "")
            chunks = doc.get("chunks_count", 0)
            status_icon = "OK" if status == "completed" else "..."
            lines.append(f"- {status_icon} **{title}** ({doc_type}) - {chunks} chunks")

        await cl.Message(content="\n".join(lines)).send()

    except Exception as e:
        await cl.Message(content=f"获取文档列表失败: {str(e)}").send()


async def _show_help():
    """展示帮助信息"""
    help_text = (
        "## 使用帮助\n\n"
        "**直接输入问题** 进行问答\n\n"
        "**命令：**\n"
        "- `/docs` 或 `文档管理` — 查看知识库文档列表\n"
        "- `/help` 或 `帮助` — 显示帮助\n"
    )
    await cl.Message(content=help_text).send()


# ============================================================
# 健康检查
# ============================================================


async def _check_health() -> bool:
    """检查后端健康"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BACKEND_URL}/health")
            return response.status_code == 200
    except Exception:
        return False
