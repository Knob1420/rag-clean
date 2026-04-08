"""
RAG 知识库前端 - 自定义样式
简洁浅色主题
"""

custom_css = """
    /* ============================================
       全局样式
       ============================================ */
    .gradio-container {
        max-width: 1000px !important;
        width: 100% !important;
        margin: 0 auto !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        background: #ffffff !important;
    }

    * {
        box-shadow: none !important;
    }

    body {
        background: #f8f9fa !important;
    }

    footer {
        visibility: hidden;
    }

    /* ============================================
       文字颜色 - 确保可读
       ============================================ */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937 !important;
    }

    p, span, div {
        color: #374151 !important;
    }

    /* ============================================
       Tab 样式
       ============================================ */
    .tab-nav {
        background: #f3f4f6 !important;
        border-bottom: 2px solid #e5e7eb !important;
    }

    button[role="tab"] {
        color: #6b7280 !important;
        background: transparent !important;
        border: none !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
    }

    button[role="tab"]:hover {
        color: #374151 !important;
        background: #e5e7eb !important;
    }

    button[role="tab"][aria-selected="true"] {
        color: #3b82f6 !important;
        background: #ffffff !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    /* ============================================
       聊天界面
       ============================================ */
    .chatbot {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
    }

    /* 聊天气泡 - 用户消息 */
    .chatbot .message.user {
        background: #3b82f6 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
    }

    /* 聊天气泡 - AI消息 */
    .chatbot .message.bot {
        background: #f3f4f6 !important;
        color: #1f2937 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
    }

    /* 确保消息内的文字颜色正确 */
    .chatbot .message.user p,
    .chatbot .message.user span,
    .chatbot .message.user div {
        color: #ffffff !important;
    }

    .chatbot .message.bot p,
    .chatbot .message.bot span,
    .chatbot .message.bot div {
        color: #1f2937 !important;
    }

    /* ============================================
       输入框样式
       ============================================ */
    textarea {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        color: #374151 !important;
        padding: 12px 16px !important;
    }

    textarea:focus {
        border-color: #3b82f6 !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    /* ============================================
       文档列表框 (Markdown 组件)
       ============================================ */
    #file-list-box {
        overflow-y: auto !important;
        max-height: 500px !important;
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }

    /* 文档列表容器 */
    #file-list-box .markdown {
        font-size: 14px !important;
        line-height: 1.6 !important;
    }

    /* ============================================
       按钮样式
       ============================================ */
    button {
        border-radius: 8px !important;
        border: none !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    .primary {
        background: #3b82f6 !important;
        color: white !important;
    }

    .primary:hover {
        background: #2563eb !important;
    }

    .secondary {
        background: #6b7280 !important;
        color: white !important;
    }

    .secondary:hover {
        background: #4b5563 !important;
    }

    .stop {
        background: #ef4444 !important;
        color: white !important;
    }

    .stop:hover {
        background: #dc2626 !important;
    }

    /* ============================================
       Markdown 内容
       ============================================ */
    .markdown {
        color: #374151 !important;
    }

    .markdown h1, .markdown h2, .markdown h3 {
        color: #111827 !important;
    }

    .markdown strong {
        color: #1f2937 !important;
    }

    .markdown code {
        background: #f3f4f6 !important;
        color: #dc2626 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    /* ============================================
       示例问题区域
       ============================================ */
    .examples {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }

    /* ============================================
       响应式设计
       ============================================ */
    @media (max-width: 768px) {
        .gradio-container {
            max-width: 100% !important;
            padding: 8px !important;
        }
    }
"""
