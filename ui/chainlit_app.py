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

CHAT_SETTINGS = cl.ChatSettings(
    inputs=[
        cl.input_widget.Select(
            id="mode",
            label="问答模式",
            items={"quick": "快速问答", "agent": "智能推理"},
            initial_value="quick",
        ),
        cl.input_widget.Slider(
            id="top_k",
            label="召回数量",
            min=1, max=50, step=1, initial=20,
        ),
        cl.input_widget.Switch(
            id="use_hyde",
            label="启用 HyDE",
            initial=False,
        ),
        cl.input_widget.Switch(
            id="use_rerank",
            label="启用 Rerank",
            initial=True,
        ),
        cl.input_widget.Slider(
            id="rerank_top_k",
            label="Rerank 保留数量",
            min=1, max=50, step=1, initial=10,
        ),
    ]
)


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

    await CHAT_SETTINGS.send()
    cl.user_session.set("mode", "quick")
    cl.user_session.set("top_k", 20)
    cl.user_session.set("use_hyde", False)
    cl.user_session.set("use_rerank", True)
    cl.user_session.set("rerank_top_k", 10)

    cl.user_session.set("chat_history", [])


_MODE_LABEL_TO_KEY = {"快速问答": "quick", "智能推理": "agent"}


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """用户修改侧边栏设置时同步到 session"""
    logger.info(f"Settings updated: {settings}")
    for key, value in settings.items():
        if key == "mode" and value in _MODE_LABEL_TO_KEY:
            value = _MODE_LABEL_TO_KEY[value]
        cl.user_session.set(key, value)


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
    # Agent 推理步骤（cl.Step 对象列表）
    agent_steps: list = []  # [{"iteration": 1, "step": cl.Step, "status": "active"}, ...]

    mode = cl.user_session.get("mode", "quick")
    top_k = int(cl.user_session.get("top_k", 20))
    use_hyde = cl.user_session.get("use_hyde", False)
    use_rerank = cl.user_session.get("use_rerank", True)
    rerank_top_k = int(cl.user_session.get("rerank_top_k", 10))

    request_payload = {
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "use_hyde": use_hyde,
        "use_rerank": use_rerank,
        "rerank_top_k": rerank_top_k,
    }

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/api/v1/chat/stream",
                json=request_payload,
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

                    elif event_type == "step_start":
                        iteration = data.get("iteration", 0)
                        max_iterations = data.get("max_iterations", 10)
                        # 创建新的 Step 显示推理过程
                        try:
                            step = cl.Step(
                                name=f"Step {iteration}/{max_iterations}: thinking...",
                                type="undefined",
                            )
                            await step.send()
                            agent_steps.append({
                                "iteration": iteration,
                                "step": step,
                                "status": "active",
                            })
                        except Exception as e:
                            logger.warning(f"Step send failed: {e}")

                    elif event_type == "thought":
                        iteration = data.get("iteration", 0)
                        thought = data.get("thought", "")
                        # 实时更新 step 名称，显示 Agent 思考内容
                        for item in agent_steps:
                            if item.get("iteration") == iteration and item.get("status") == "active":
                                step = item["step"]
                                step.name = f"Step {iteration}: thinking...\n└ {thought[:100]}"
                                await step.send()
                                break

                    elif event_type == "step_end":
                        iteration = data.get("iteration", 0)
                        action = data.get("action", "")
                        duration = data.get("duration", 0)
                        preview = data.get("observation_preview", "")
                        # 找到对应迭代的步骤并更新名称
                        for item in agent_steps:
                            if item.get("iteration") == iteration and item.get("status") == "active":
                                step = item["step"]
                                item["status"] = "done"
                                # 更新 step 名称
                                step.name = f"Step {iteration}: {action} ({duration:.2f}s)"
                                if preview:
                                    step.name += f"\n└ {preview[:80]}"
                                await step.send()
                                break

                    elif event_type == "sources":
                        sources_list = data.get("sources", [])

                    elif event_type == "done":
                        timing = data.get("time", {})
                        total_time = timing.get("total", 0)
                        chunks_count = data.get("chunks_count", 0)
                        footer = (
                            f"\n\n---\n"
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

        # 添加 Action 按钮
        actions = [
            cl.Action(
                name="regenerate",
                payload={"query": query},
                label="重新生成",
            ),
        ]
        if mode == "quick":
            actions.append(
                cl.Action(
                    name="switch_agent",
                    payload={"query": query},
                    label="切换到 Agent 模式",
                )
            )
        else:
            actions.append(
                cl.Action(
                    name="switch_quick",
                    payload={"query": query},
                    label="切换到快速问答",
                )
            )
        msg.actions = actions
        await msg.update()

        # 更新聊天历史
        chat_history.append({"role": "assistant", "content": full_answer})
        cl.user_session.set("chat_history", chat_history)

    except httpx.ConnectError:
        await cl.Message(
            content="**后端 API 未启动**，请先运行 `python run.py --main`"
        ).send()
    except Exception as e:
        import traceback
        logger.error(f"Chat error: {e}", exc_info=True)
        error_detail = str(e)
        if not error_detail:
            error_detail = traceback.format_exc()
        await cl.Message(content=f"**Error:** {error_detail[:500]}").send()


# ============================================================
# Action 回调
# ============================================================


@cl.action_callback("regenerate")
async def on_regenerate(action: cl.Action):
    query = action.payload.get("query", "")
    if not query:
        return
    await action.remove()
    await on_message(cl.Message(content=query))


@cl.action_callback("switch_agent")
async def on_switch_agent(action: cl.Action):
    query = action.payload.get("query", "")
    if not query:
        return
    await action.remove()
    cl.user_session.set("mode", "agent")
    await on_message(cl.Message(content=query))


@cl.action_callback("switch_quick")
async def on_switch_quick(action: cl.Action):
    query = action.payload.get("query", "")
    if not query:
        return
    await action.remove()
    cl.user_session.set("mode", "quick")
    await on_message(cl.Message(content=query))


# ============================================================
# 反馈
# ============================================================


@cl.on_feedback
async def on_feedback(feedback: cl.types.Feedback):
    sentiment = "赞" if feedback.value == 1 else "踩"
    logger.info(
        f"User feedback: {sentiment} "
        f"message={feedback.forId} "
        f"comment={feedback.comment or '(none)'}"
    )


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
