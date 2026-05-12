"""
RAG 知识库前端 - Chainlit 界面

功能：
- 流式聊天问答（SSE 对接后端 /api/v1/chat/stream）
- 参考来源展示（去重文档列表）
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import chainlit as cl
import httpx
from loguru import logger

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

BACKEND_URL = "http://localhost:8000"


# ============================================================
# 启动 / 欢迎
# ============================================================


@cl.on_chat_start
async def on_chat_start():
    """会话开始 — 健康检查"""
    healthy = await _check_health()
    if not healthy:
        await cl.Message(
            content="**后端 API 未启动**，请先运行 `python run.py --main`"
        ).send()
        return

    cl.user_session.set("chat_history", [])


# ============================================================
# 聊天处理
# ============================================================


@cl.on_message
async def on_message(message: cl.Message):
    """处理用户消息 — SSE 流式调用后端（Quick 模式）"""
    query = message.content.strip()
    if not query:
        return

    # 更新聊天历史
    chat_history: List[Dict[str, str]] = cl.user_session.get("chat_history", [])
    chat_history.append({"role": "user", "content": query})
    cl.user_session.set("chat_history", chat_history)

    msg = cl.Message(content="")
    await msg.send()

    full_answer = ""
    sources_list: list = []

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/api/v1/chat/stream",
                json={"query": query, "mode": "quick", "use_hyde": False},
            ) as response:
                response.raise_for_status()

                event_type = ""

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                        continue

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event_type == "token":
                        token = data.get("content", "")
                        full_answer += token
                        await msg.stream_token(token)

                    elif event_type == "sources":
                        sources_list = data.get("sources", [])

                    elif event_type == "done":
                        timing = data.get("time", {})
                        total_time = timing.get("total", 0)
                        gen_time = timing.get("generation", 0)
                        chunks_count = data.get("chunks_count", 0)
                        footer = (
                            f"\n\n---\n"
                            f"Generation: {gen_time:.2f}s | "
                            f"Total: {total_time:.2f}s | "
                            f"{chunks_count} 条参考"
                        )
                        await msg.stream_token(footer)

                    elif event_type == "error":
                        error_msg = data.get("error", "未知错误")
                        await msg.stream_token(f"\n\n**Error:** {error_msg}")

        # 追加去重后的参考文档列表
        if sources_list:
            seen: set = set()
            doc_lines = []
            for src in sources_list:
                doc_name = (src.get("doc_name") or "").strip()
                if doc_name and doc_name not in seen:
                    seen.add(doc_name)
                    score = src.get("score", 0)
                    doc_lines.append(f"  {len(seen)}. {doc_name}  (score={score:.2f})")
            if doc_lines:
                sources_text = "\n\n**参考文档**\n" + "\n".join(doc_lines)
                await msg.stream_token(sources_text)

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
