"""
RAG 知识库前端 - Gradio 主界面

简洁版本，包含问答和文档管理两个 Tab
"""

import gradio as gr
import asyncio
from typing import List, Tuple, Optional
from loguru import logger

from ui.css import custom_css
from ui.chat_backend import default_backend


# ============================================================
# 聊天处理函数
# ============================================================


async def chat_handler(message: str, history: List[Tuple[str, str]]):
    """
    处理聊天消息

    Args:
        message: 用户消息
        history: 聊天历史

    Returns:
        回答字符串（包含时间信息）
    """
    if not message or not message.strip():
        return "请输入问题"

    try:
        # 调用后端 API
        result = await default_backend.chat(
            query=message, top_k=10, use_rewrite=True, use_rerank=True, rerank_top_k=5
        )

        # 检查错误
        if "error" in result:
            return f"Error: {result['error']}"

        # 获取回答和时间信息
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        total_time = result.get("time", {})
        usage = result.get("usage", {})

        # 时间单位转换：后端返回秒，转换为更友好的显示
        query_rewrite_time = total_time.get("query_rewrite", 0)
        retrieval_time = total_time.get("retrieval", 0)
        rerank_time = total_time.get("rerank", 0)
        generation_time = total_time.get("generation", 0)

        # 格式化时间显示（<0.1秒显示毫秒，>=0.1秒显示秒）
        def fmt_time(t):
            if t < 0.1:
                return f"{t*1000:.0f}ms"
            else:
                return f"{t:.2f}s"

        # 在答案末尾添加时间信息
        timing_info = f"\n\n---\nProcessing time: "
        if query_rewrite_time > 0:
            timing_info += f"{fmt_time(query_rewrite_time)} (rewrite) + "
        timing_info += f"{fmt_time(retrieval_time)} (retrieval) + {fmt_time(rerank_time)} (rerank) + {fmt_time(generation_time)} (generation)"
        timing_info += f"\nTokens: {usage.get('total_tokens', 0)} (input:{usage.get('prompt_tokens', 0)} + output:{usage.get('completion_tokens', 0)})"

        logger.info(
            f"Chat success: message='{message[:30]}...', sources={len(sources)}"
        )

        return answer + timing_info

    except Exception as e:
        logger.error(f"Chat handler error: {e}", exc_info=True)
        return f"Error: {str(e)}"


# ============================================================
# 文档管理函数
# ============================================================


async def format_file_list(page: int = 1, page_size: int = 20):
    """格式化文档列表（带分页）"""
    try:
        result = await default_backend.list_documents(page=page, page_size=page_size)

        if "error" in result or not result.get("documents"):
            return "### No documents found", "1 / 1"

        docs = result.get("documents", [])
        total = result.get("total", 0)
        total_pages = (total + page_size - 1) // page_size

        lines = [f"### Documents ({total} total, page {page}/{total_pages})\n"]

        for doc in docs:
            title = doc.get("title", "Unknown")
            doc_type = doc.get("doc_type", "")
            status = doc.get("status", "")
            chunks = doc.get("chunks_count", 0)

            status_icon = "OK" if status == "completed" else "..."
            lines.append(f"{status_icon} **{title}** ({doc_type}) - {chunks} chunks\n")

        # 分页导航
        if total_pages > 1:
            nav = "Page: "
            if page > 1:
                nav += "Prev | "
            nav += f"{page}/{total_pages}"
            if page < total_pages:
                nav += " | Next"
            lines.append(f"\n{nav}")

        return "\n".join(lines), f"{page} / {total_pages}"

    except Exception as e:
        logger.error(f"Format file list error: {e}")
        return f"### Error loading documents\n\n{str(e)}", "1 / 1"


# ============================================================
# Gradio 应用
# ============================================================


def create_gradio_ui():
    """创建 Gradio 界面"""

    with gr.Blocks(
        title="RAG Knowledge Base",
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"),
    ) as demo:

        gr.Markdown(
            """
            # RAG Knowledge Base

            Document-based intelligent Q&A system
            """
        )

        with gr.Tabs():

            # ========================================
            # 聊天 Tab
            # ========================================
            with gr.Tab("Chat"):

                chatbot = gr.Chatbot(
                    height=600,
                    placeholder="Ask me anything about the documents!",
                    show_label=False,
                    type="tuples",
                )

                gr.ChatInterface(
                    fn=chat_handler,
                    chatbot=chatbot,
                    examples=[
                        "什么是三体计算星座？",
                        "智加G1和智加NX2有什么区别？",
                        "天基分布式操作系统的特点",
                        "智加G1的参数",
                    ],
                )

            # ========================================
            # 文档 Tab
            # ========================================
            with gr.Tab("Documents"):

                gr.Markdown("## Knowledge Base Documents")

                # 分页状态
                with gr.Row():
                    page_info = gr.Textbox(
                        value="1 / 1", label="Page", interactive=False, scale=1
                    )
                    page_size = gr.Slider(
                        10, 50, value=20, step=10, label="Per page", scale=2
                    )

                file_list = gr.Markdown(value="Loading...", elem_id="file-list-box")

                with gr.Row():
                    prev_btn = gr.Button("Prev", scale=1)
                    refresh_btn = gr.Button("Refresh", scale=1)
                    next_btn = gr.Button("Next", scale=1)

                # 分页状态（隐藏）
                current_page = gr.State(1)
                total_pages = gr.State(1)

                # 刷新文档列表
                async def refresh_docs(page, size):
                    content, page_str = await format_file_list(
                        page=page, page_size=size
                    )
                    # 提取总页数
                    parts = page_str.split("/")
                    total = int(parts[-1].strip()) if len(parts) > 1 else 1
                    return content, page_str, page, total

                # 上一页
                async def prev_page(page, size):
                    return await refresh_docs(max(1, page - 1), size)

                # 下一页
                async def next_page(page, total, size):
                    return await refresh_docs(min(total, page + 1), size)

                # 刷新按钮
                refresh_btn.click(
                    fn=refresh_docs,
                    inputs=[current_page, page_size],
                    outputs=[file_list, page_info, current_page, total_pages],
                )

                # 上一页
                prev_btn.click(
                    fn=prev_page,
                    inputs=[current_page, page_size],
                    outputs=[file_list, page_info, current_page, total_pages],
                )

                # 下一页
                next_btn.click(
                    fn=next_page,
                    inputs=[current_page, total_pages, page_size],
                    outputs=[file_list, page_info, current_page, total_pages],
                )

                # 页面大小改变时重新加载
                page_size.change(
                    fn=refresh_docs,
                    inputs=[current_page, page_size],
                    outputs=[file_list, page_info, current_page, total_pages],
                )

                # 页面加载时获取文档列表
                demo.load(
                    fn=refresh_docs,
                    inputs=[current_page, page_size],
                    outputs=[file_list, page_info, current_page, total_pages],
                )

    return demo


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    logger.info("Starting RAG Knowledge Base Frontend...")

    demo = create_gradio_ui()
    demo.launch(
        server_name=args.host, server_port=args.port, share=args.share, show_error=True
    )
