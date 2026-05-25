"""
Wiki 生成的核心 Prompts

两阶段 Chain-of-Thought：
1. Analysis - 分析源文档，提取实体、概念、关联
2. Generation - 基于分析结果生成 wiki 页面

Ported from LLM Wiki (src/lib/ingest.ts)
"""

from typing import Optional


def language_rule(source_content: str = "", output_lang: str = "zh") -> str:
    """
    构建语言规则 directive

    Args:
        source_content: 源文档内容（用于检测语言）
        output_lang: 输出语言，"zh" 或 "en"
    """
    if output_lang == "en":
        return """## ⚠️ MANDATORY OUTPUT LANGUAGE: English

You MUST write your entire response (including wiki page titles, content, descriptions, summaries, and any generated text) in **English**.
The source material or wiki content may be in a different language, but this is IRRELEVANT to your output language.
Ignore the language of any source content. Generate everything in English only.
Proper nouns should use standard English transliteration when appropriate.
DO NOT use any other language. This overrides all other instructions."""
    else:
        return """## ⚠️ MANDATORY OUTPUT LANGUAGE: 中文

你必须使用**中文**撰写所有回复（包括 wiki 页面标题、内容、描述、摘要和任何生成的文本）。
源材料或 wiki 内容可能是其他语言，但这与你的输出语言无关。
忽略任何源内容的语言，全部使用中文生成。
专有名词应使用标准中文音译（如适用）。
禁止使用其他语言。这会覆盖所有其他指令。"""


def build_analysis_prompt(
    purpose: str = "",
    index: str = "",
    source_content: str = "",
    folder_context: str = "",
    output_lang: str = "zh",
) -> str:
    """
    Step 1 Prompt: AI 读取源文档并生成结构化分析。

    这是 "discussion" 步骤 — AI 在写 wiki 页面之前先对源进行推理。

    Args:
        purpose: 项目 purpose.md 内容
        index: 当前 wiki 的 index.md 内容
        source_content: 源文档内容（用于语言检测）
        folder_context: 文件夹路径上下文（分类提示）
        output_lang: 输出语言

    Returns:
        Analysis system prompt
    """
    parts = [
        "You are an expert research analyst. Read the source document and produce a structured analysis.",
        "Do not output chain-of-thought, hidden reasoning, or a thinking transcript. Reason internally and write only the concise final analysis.",
        "",
        language_rule(source_content, output_lang),
        "",
        "Your analysis should cover:",
        "",
        "## Key Entities",
        "List people, organizations, products, datasets, tools mentioned. For each:",
        "- Name and type",
        "- Role in the source (central vs. peripheral)",
        "- Whether it likely already exists in the wiki (check the index)",
        "  IMPORTANT: Use BROAD names. If a product has models/variants, use the general product name.",
        "  BAD: \"星载智算机-高性能版\" (too specific) → GOOD: \"星载智算机\" (broader category)",
        "",
        "## Key Concepts",
        "List theories, methods, techniques, phenomena. For each:",
        "- Name and brief definition",
        "- Why it matters in this source",
        "- Whether it likely already exists in the wiki",
        "  IMPORTANT: Use GENERAL concepts, not specific implementations.",
        "  BAD: \"SCS-01遥测协议\" (too specific) → GOOD: \"遥测遥控协议\" (general concept)",
        "",
        "## Main Arguments & Findings",
        "- What are the core claims or results?",
        "- What evidence supports them?",
        "- How strong is the evidence?",
        "",
        "## Connections to Existing Wiki",
        "- What existing pages does this source relate to?",
        "- Does it strengthen, challenge, or extend existing knowledge?",
        "",
        "## Contradictions & Tensions",
        "- Does anything in this source conflict with existing wiki content?",
        "- Are there internal tensions or caveats?",
        "",
        "## Recommendations",
        "- What wiki pages should be created or updated?",
        "- What should be emphasized vs. de-emphasized?",
        "- Any open questions worth flagging for the user?",
        "",
        "Be thorough but concise. Focus on what's genuinely important.",
        "",
    ]

    if folder_context:
        parts.append(
            "If a folder context is provided, use it as a hint for categorization — "
            "the folder structure often reflects the user's organizational intent "
            "(e.g., 'papers/energy' suggests the file is an energy-related paper)."
        )
        parts.append(f"\n**Folder context:** {folder_context}")
        parts.append("")

    if purpose:
        parts.append(f"## Wiki Purpose (for context)\n{purpose}")
        parts.append("")

    if index:
        parts.append(f"## Current Wiki Index (for checking existing content)\n{index}")
        parts.append("")

    return "\n".join(parts)


def build_analysis_user_message(
    file_name: str, folder_context: str, content: str, max_chars: int = 150000
) -> str:
    """
    Step 1 User message - 分析阶段的用户消息

    Args:
        file_name: 源文件名
        folder_context: 文件夹上下文
        content: 源文档内容
        max_chars: 最大字符数（避免超出 context window）

    Returns:
        Analysis user message
    """
    truncated = content[:max_chars] if len(content) > max_chars else content

    msg = f"Analyze this source document:\n\n**File:** {file_name}"
    if folder_context:
        msg += f"\n**Folder context:** {folder_context}"
    msg += f"\n\n---\n\n{truncated}"

    return msg


# ══════════════════════════════════════════════════════════════════════════════
# Frontmatter 格式规则（用于 Generation Prompt）
# ══════════════════════════════════════════════════════════════════════════════

FRONTMATTER_RULES = """## Frontmatter Rules (CRITICAL — parser is strict)

Every page begins with a YAML frontmatter block. Format rules, in order of importance:

1. The VERY FIRST line of the file MUST be exactly `---` (three hyphens, nothing else).
   Do NOT wrap the file in a ```yaml ... ``` code fence.
   Do NOT prefix it with a `frontmatter:` key or any other line.
2. Each frontmatter line is a `key: value` pair on its own line.
3. The frontmatter ends with another `---` line on its own.
4. The next line after the closing `---` is the start of the page body.
5. Arrays use the standard YAML inline form `[a, b, c]` (no outer brackets around each item).
   Wikilinks belong in the BODY only — never write `related: [[a]], [[b]]` (invalid YAML);
   write `related: [a, b]` with bare slugs.

Required fields and types:
  • type     — one of: source | entity | concept | comparison | query | synthesis
  • title    — string (quote it if it contains a colon, e.g. `title: "Foo: Bar"`)
  • created  — date in YYYY-MM-DD form (no quotes)
  • updated  — same as created
  • tags     — array of bare strings: `tags: [microbiology, ai]`
  • related  — array of bare wiki page slugs: `related: [foo, bar-baz]`. Do NOT include
               `wiki/`, `.md`, or `[[…]]` here — slugs only.
  • sources  — array of source filenames; MUST include the source filename.

Optional fields:
  • description — brief description of the page
  • links       — for source pages, arrays of linked entity/concept page slugs
  • wikilinks   — in body content, use [[Page Title]] syntax for cross-references

Wiki pages are always .md files."""


# ══════════════════════════════════════════════════════════════════════════════
# File/Review Block 解析规则（用于 Generation Prompt）
# ══════════════════════════════════════════════════════════════════════════════

FILE_REVIEW_BLOCK_RULES = """## Output Format: FILE blocks and REVIEW blocks

Your output consists of FILE blocks and REVIEW blocks, alternating. Each starts with a marker line.

### FILE Block Format
```
FILE: concepts/my-concept.md
---
[frontmatter]
---
[content]
```

- The FILE marker line contains the path relative to the wiki root directory (NO "wiki/" prefix).
- The frontmatter and body follow the rules above.
- Every generated file gets its own FILE block.

### REVIEW Block Format
```
REVIEW: [brief title]
Type: [Create Page | Deep Research | Skip]
Priority: [high | medium | low]
Reason: [2-3 sentence explanation of why this matters]
Suggestions:
- [Specific, actionable suggestion 1]
- [Specific, actionable suggestion 2]
```

- REVIEW items are for topics that need human judgment or additional research.
- "Skip" means the topic is valid but doesn't need action right now.
- Each REVIEW block should be concise but informative.

### Combined Output Example
```
FILE: sources/SCS-01-软件研制任务书.md
---
type: source
title: "SCS-01 软件研制任务书"
created: 2026-05-19
updated: 2026-05-19
tags: [研制任务书, 软件, SCS-01]
related: [星载智算机, 之江首发星座]
sources: [SCS-01-软件研制任务书.md]
---
# SCS-01 软件研制任务书

...


REVIEW: 研制阶段定义不明确
Type: Create Page
Priority: medium
Reason: 需要创建统一的研制阶段定义页面，确保各文档使用一致的阶段术语。
Suggestions:
- Create concepts/研制阶段.md with standard phase definitions
- Cross-link from all source documents that mention phases
```
"""


def build_generation_prompt(
    schema: str = "",
    purpose: str = "",
    index: str = "",
    source_file_name: str = "",
    overview: str = "",
    source_content: str = "",
    output_lang: str = "zh",
) -> str:
    """
    Step 2 Prompt: AI 基于分析结果生成 wiki 文件和 Review 项。

    Args:
        schema: 项目 schema.md 内容
        purpose: 项目 purpose.md 内容
        index: 当前 wiki 的 index.md 内容
        source_file_name: 源文件名
        overview: 当前 wiki 的 overview.md 内容
        source_content: 源文档内容（用于语言检测）
        output_lang: 输出语言

    Returns:
        Generation system prompt
    """
    source_base_name = source_file_name.split(".")[0]

    parts = [
        "You are a wiki maintainer. Based on the analysis provided, generate wiki files.",
        "Do not output chain-of-thought, hidden reasoning, or explanatory preamble. "
        "Reason internally and output only the requested FILE/REVIEW blocks.",
        "",
        "## ⚠️ CRITICAL: Preventing Duplicate Entities/Concepts",
        "",
        "Before creating any new entity or concept page, you MUST check the index for existing entries.",
        "Follow these rules strictly:",
        "",
        "1. **Check the index first** — Scan all existing entities and concepts in index.md",
        "   - If a similar entry already exists, do NOT create a new page",
        "   - Instead, add the new information to the existing page by including it in sources/related",
        "",
        "2. **Use broad, general names** — Avoid overly specific names",
        "   - GOOD:  \"星载智算机\" (broad product category)",
        "   - BAD:   \"星载智算机-高性能模块\" (too specific, subset of above)",
        "   - GOOD:  \"卫星通信协议\" (broad concept)",
        "   - BAD:   \"SCS-01遥测协议V1.2\" (too specific, would be a source)",
        "",
        "3. **Never create subset entities/concepts** — If an existing page covers a topic,",
        "   new content should be merged into it, not create a child/subset page",
        "   - If \"星载智算机\" exists, do not create \"星载智算机-性能\" or \"星载智算机-接口\"",
        "   - Merge new information into the existing broader page",
        "",
        "4. **When in doubt, reuse existing** — If an entity/concept name could reasonably",
        "   be considered a subset of an existing one, treat it as the same entity",
        "",
        "",
        language_rule(source_content, output_lang),
        "",
        f"## IMPORTANT: Source File",
        f"The original source file is: **{source_file_name}**",
        f"All wiki pages generated from this source MUST include this filename in their frontmatter `sources` field.",
        "",
        "## What to generate",
        "",
        f"1. A source summary page at **sources/{source_base_name}.md** (MUST use this exact path)",
        "2. Entity pages in **entities/** for key entities identified in the analysis",
        "3. Concept pages in **concepts/** for key concepts identified in the analysis",
        "4. An updated **index.md** — add new entries to existing categories, preserve all existing entries",
        "5. A log entry for **log.md** (just the new entry to append, format: ## [YYYY-MM-DD] ingest | Title)",
        f"6. An updated **overview.md** — a high-level summary of what the entire wiki covers, "
        f"updated to reflect the newly ingested source. This should be a comprehensive 2-5 paragraph "
        f"overview of ALL topics in the wiki, not just the new source.",
        "",
        "NOTE: Use paths WITHOUT 'wiki/' prefix (e.g., 'sources/xxx.md' not 'wiki/sources/xxx.md')",
        "",
        FRONTMATTER_RULES,
        "",
        FILE_REVIEW_BLOCK_RULES,
        "",
    ]

    if schema:
        parts.append(f"## Wiki Schema (page structure rules)\n{schema}")
        parts.append("")

    if purpose:
        parts.append(f"## Wiki Purpose (for context)\n{purpose}")
        parts.append("")

    if index:
        parts.append(f"## Current Wiki Index (for checking existing content)\n{index}")
        parts.append("")

    if overview:
        parts.append(f"## Current Wiki Overview\n{overview}")
        parts.append("")

    return "\n".join(parts)


def build_generation_user_message(analysis: str, source_file_name: str) -> str:
    """
    Step 2 User message - 生成阶段的用户消息

    Args:
        analysis: Step 1 的分析结果
        source_file_name: 源文件名

    Returns:
        Generation user message
    """
    return f"""Based on the following analysis of **{source_file_name}**, generate the wiki files and review items.

---

{analysis}

---

Generate the wiki files now."""


# ══════════════════════════════════════════════════════════════════════════════
# Page Merger Prompt（合并同一页面的两个版本）
# ══════════════════════════════════════════════════════════════════════════════

PAGE_MERGER_PROMPT = """You are merging two versions of the same wiki page into one coherent document.
Both versions describe the same entity / concept; one is already on disk,
the other was just generated from a different source document.
Your job is to produce a single unified page that:
1. Keeps all unique information from both versions
2. Resolves contradictions in favor of the more recent or more authoritative version
3. Preserves the existing frontmatter style and structure
4. Uses [[wikilink]] syntax for cross-references to other wiki pages
5. Maintains consistent tone and depth

Output only the merged FILE block."""


# ══════════════════════════════════════════════════════════════════════════════
# Lint Prompt（修复破损的 wikilinks）
# ══════════════════════════════════════════════════════════════════════════════

LINT_PROMPT = """You are a wiki maintenance assistant. Your job is to fix broken [[wikilinks]] and improve page structure.

Given a list of wiki pages and the current index, you must:
1. Identify any [[wikilinks]] that point to pages not in the index — remove or fix them
2. Ensure all cross-references use consistent [[Page Title]] syntax
3. Fix any duplicate entries in index.md
4. Ensure each page has proper frontmatter (type, title, created, updated, tags, related, sources)

Output a JSON object with:
- pages_to_update: list of page paths and their corrected content
- index_updates: corrected index.md content
- removed_links: list of broken links that were removed

If everything is correct, output: {{"pages_to_update": [], "index_updates": null, "removed_links": []}}"""


# ══════════════════════════════════════════════════════════════════════════════
# Deep Research Prompt（网络搜索研究）
# ══════════════════════════════════════════════════════════════════════════════

DEEP_RESEARCH_SYSTEM_PROMPT = """You are a research assistant. Based on the search results provided, synthesize a comprehensive research report.

Your report should:
1. Summarize the key findings from the search results
2. Identify consensus and contradictions
3. Connect findings to the existing wiki knowledge
4. Highlight gaps in knowledge that need further research
5. Use proper citations [1], [2], etc. for all claims

Output a FILE block for the synthesis page in wiki/synthesis/, plus REVIEW items for any gaps identified."""


# ══════════════════════════════════════════════════════════════════════════════
# Chat System Prompt（基于知识库的问答）
# ══════════════════════════════════════════════════════════════════════════════

CHAT_SYSTEM_PROMPT = """You are a knowledgeable assistant helping to answer questions about a wiki knowledge base.

When answering:
1. Use the provided wiki content as your primary source
2. Cite pages using [1], [2], etc. notation
3. If information is not in the wiki, say so clearly
4. Connect related topics when relevant
5. Use [[Page Title]] syntax when mentioning other wiki pages

The wiki covers: {overview_summary}"""


# ══════════════════════════════════════════════════════════════════════════════
# Utility: 构建 slug（用于 frontmatter related 字段）
# ══════════════════════════════════════════════════════════════════════════════


def slugify(title: str) -> str:
    """
    将标题转换为 slug（用于 frontmatter related 字段）

    Args:
        title: 页面标题

    Returns:
        slug 格式的字符串
    """
    import re

    # 移除特殊字符，转小写，空格变连字符
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug.strip("-")


def build_related_slug(title: str) -> str:
    """
    构建用于 related 字段的 slug

    和 slugify 一样，但命名更语义化
    """
    return slugify(title)
