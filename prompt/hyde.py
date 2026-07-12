"""HyDE 假设性文档生成 & Summary 生成 Prompt"""

# ── HyDE ─────────────────────────────────────────

HYDE_SYSTEM_PROMPT = (
    "根据用户问题，写出一段简短的假设性回答（50字以内）。"
    "只包含关键术语和核心信息，不需要完整论述。"
    "只输出内容，不加任何声明。"
)

HYDE_USER_TEMPLATE = "问题：{query}\n假设性回答："


# ── Summary 生成 ─────────────────────────────────

SUMMARY_SYSTEM_PROMPT = """你是一个文档分析专家，擅长提取核心信息。

你的任务是为给定的文档片段生成 50-100 字的简洁摘要，概括核心内容。

只返回 JSON 格式，不要任何其他内容。

输出格式：
{
    "summary": "简洁摘要（50-100字）"
}"""


def build_summary_prompt(content: str) -> str:
    """构建 summary 生成的 prompt"""
    return f"""为以下文档片段生成摘要：

{content[:2000]}

只返回 JSON 格式："""
