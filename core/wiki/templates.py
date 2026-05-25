"""
Wiki 模板 - 预定义的 purpose 和 schema 组合

Ported from LLM Wiki (src/lib/templates.ts)
"""

from dataclasses import dataclass
from typing import List


@dataclass
class WikiTemplate:
    """Wiki 模板定义"""

    id: str
    name: str
    description: str
    icon: str
    schema: str
    purpose: str
    extra_dirs: List[str]


# ══════════════════════════════════════════════════════════════════════════════
# 通用 Base 配置（被各模板引用）
# ══════════════════════════════════════════════════════════════════════════════

BASE_SCHEMA_TYPES = """| entity | wiki/entities/ | Named things (people, tools, organizations, datasets) |
| concept | wiki/concepts/ | Ideas, techniques, phenomena, frameworks |
| source | wiki/sources/ | Papers, articles, talks, books, blog posts |
| query | wiki/queries/ | Open questions under active investigation |
| comparison | wiki/comparisons/ | Side-by-side analysis of related entities |
| synthesis | wiki/synthesis/ | Cross-cutting summaries and conclusions |
| overview | wiki/ | High-level project summary (one per project) |"""

BASE_NAMING = """- Files: `kebab-case.md`
- Entities: match official name where possible (e.g., `openai.md`, `gpt-4.md`)
- Concepts: descriptive noun phrases (e.g., `chain-of-thought.md`)
- Sources: `author-year-slug.md` (e.g., `wei-2022-cot.md`)
- Queries: question as slug (e.g., `does-scale-improve-reasoning.md`)"""

BASE_FRONTMATTER = """All pages must include YAML frontmatter:

```yaml
---
type: entity | concept | source | query | comparison | synthesis | overview
title: Human-readable title
tags: []
related: []
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

Source pages also include:
```yaml
authors: []
year: YYYY
url: ""
venue: ""
```"""

BASE_INDEX_FORMAT = """`wiki/index.md` lists all pages grouped by type. Each entry:
```
- [[page-slug]] — one-line description
```"""

BASE_LOG_FORMAT = """`wiki/log.md` records activity in reverse chronological order:
```
## YYYY-MM-DD

- Action taken / finding noted
```"""

BASE_CROSSREF = """- Use `[[page-slug]]` syntax to link between wiki pages
- Every entity and concept should appear in `wiki/index.md`
- Queries link to the sources and concepts they draw on
- Synthesis pages cite all contributing sources via `related:`"""

BASE_CONTRADICTION = """When sources contradict each other:
1. Note the contradiction in the relevant concept or entity page
2. Create or update a query page to track the open question
3. Link both sources from the query page
4. Resolve in a synthesis page once sufficient evidence exists"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. General 模板（空白模板）
# ══════════════════════════════════════════════════════════════════════════════

generalTemplate = WikiTemplate(
    id="general",
    name="General",
    description="Minimal setup — a blank slate for any purpose",
    icon="📄",
    extra_dirs=[],
    schema=f"""# Wiki Schema

## Page Types

{BASE_SCHEMA_TYPES}

## Naming Conventions

{BASE_NAMING}

## Frontmatter

{BASE_FRONTMATTER}

## Index Format

{BASE_INDEX_FORMAT}

## Log Format

{BASE_LOG_FORMAT}

## Cross-referencing Rules

{BASE_CROSSREF}

## Contradiction Handling

{BASE_CONTRADICTION}
""",
    purpose="""# Project Purpose

## Goal

<!-- What are you trying to understand or build? -->

## Key Questions

<!-- List the primary questions driving this project -->

1.
2.
3.

## Scope

**In scope:**
-

**Out of scope:**
-

## Thesis

<!-- Your current working hypothesis or conclusion (update as the project progresses) -->

> TBD
""",
)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Business 模板（会议/决策/项目）
# ══════════════════════════════════════════════════════════════════════════════

businessTemplate = WikiTemplate(
    id="business",
    name="Business",
    description="Manage meetings, decisions, projects, and stakeholder context for a team",
    icon="💼",
    extra_dirs=[
        "wiki/meetings",
        "wiki/decisions",
        "wiki/projects",
        "wiki/stakeholders",
    ],
    schema=f"""# Wiki Schema — Business / Team

## Page Types

{BASE_SCHEMA_TYPES}
| meeting | wiki/meetings/ | Meeting notes, agendas, and action items |
| decision | wiki/decisions/ | Architectural or strategic decisions (ADR-style) |
| project | wiki/projects/ | Project briefs, status, and retrospectives |
| stakeholder | wiki/stakeholders/ | People, teams, and organisations involved |

## Naming Conventions

{BASE_NAMING}
- Meetings: `YYYY-MM-DD-slug.md` (e.g., `2024-03-15-sprint-planning.md`)
- Decisions: `NNN-slug.md` (e.g., `001-adopt-typescript.md`)
- Projects: descriptive slug (e.g., `payments-redesign.md`)
- Stakeholders: name or team in kebab-case (e.g., `alice-chen.md`, `platform-team.md`)

## Frontmatter

{BASE_FRONTMATTER}

Meeting pages also include:
```yaml
date: YYYY-MM-DD
attendees: []
action_items: []
```

Decision pages also include:
```yaml
status: proposed | accepted | deprecated | superseded
deciders: []
date: YYYY-MM-DD
supersedes: ""   # slug of ADR this replaces, if any
```

Project pages also include:
```yaml
status: planned | active | on-hold | complete | cancelled
owner: ""
start_date: YYYY-MM-DD
target_date: YYYY-MM-DD
```

## Index Format

{BASE_INDEX_FORMAT}

## Log Format

{BASE_LOG_FORMAT}

## Cross-referencing Rules

{BASE_CROSSREF}
- Meeting notes reference attendees via `attendees:` frontmatter and `[[stakeholder-slug]]` links
- Decision pages link to the meetings where the decision was discussed
- Project pages link to their key decisions via `related:`
- Stakeholder pages list projects and decisions they are involved in

## Contradiction Handling

{BASE_CONTRADICTION}

## Business-Specific Conventions

- Write meeting notes during or within 24 hours — memory fades fast
- Action items must have a named owner and due date to be actionable
- Decision pages capture *context and consequences*, not just the decision itself
- Deprecated decisions should link to the decision that superseded them
- Projects should have a retrospective section added on completion
""",
    purpose="""# Project Purpose — Business / Team

## Business Context

**Organisation / Team:**
**Domain:**
**Time period covered:**

## Objectives

<!-- What are the top-level business objectives this wiki supports? -->

1.
2.
3.

## Key Projects

<!-- High-level list — create detailed pages in wiki/projects/ -->

-
-

## Key Stakeholders

<!-- Who are the primary people or teams involved? -->

-
-

## Open Decisions

<!-- Decisions currently in flight — create ADR pages in wiki/decisions/ -->

-
-

## Metrics / Success Criteria

<!-- How does the team measure progress toward its objectives? -->

-

## Constraints and Risks

<!-- Known constraints (budget, time, org) and risks to track -->

-

## Review Cadence

**Weekly sync notes:**
**Monthly status update:**
**Quarterly retrospective:**
""",
)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Research 模板（深度研究）
# ══════════════════════════════════════════════════════════════════════════════

researchTemplate = WikiTemplate(
    id="research",
    name="Research",
    description="Deep-dive research with hypothesis tracking and methodology notes",
    icon="🔬",
    extra_dirs=["wiki/methodology", "wiki/findings", "wiki/thesis"],
    schema=f"""# Wiki Schema — Research Deep-Dive

## Page Types

{BASE_SCHEMA_TYPES}
| thesis | wiki/thesis/ | Working hypothesis and its evolution over time |
| methodology | wiki/methodology/ | Research methods, protocols, and study designs |
| finding | wiki/findings/ | Individual empirical results or observations |

## Naming Conventions

{BASE_NAMING}
- Theses: hypothesis as slug (e.g., `scaling-improves-reasoning.md`)
- Methodologies: method name (e.g., `systematic-review.md`, `ablation-study.md`)
- Findings: descriptive slug (e.g., `larger-models-better-few-shot.md`)

## Frontmatter

{BASE_FRONTMATTER}

Thesis pages also include:
```yaml
confidence: low | medium | high
status: speculative | supported | refuted | settled
```

Finding pages also include:
```yaml
source: "[[source-slug]]"
confidence: low | medium | high
replicated: true | false | null
```

## Index Format

{BASE_INDEX_FORMAT}

## Log Format

{BASE_LOG_FORMAT}

## Cross-referencing Rules

{BASE_CROSSREF}
- Findings link back to their source via the `source:` frontmatter field
- Thesis pages reference supporting and refuting findings via `related:`
- Methodology pages are cited by the findings that used them

## Contradiction Handling

{BASE_CONTRADICTION}

## Research-Specific Conventions

- Keep the thesis pages updated as evidence accumulates — they are living documents
- Every finding should assess replication status when known
- Methodology pages explain the *why* (rationale) not just the *how*
- Distinguish between direct evidence and inference in finding pages
""",
    purpose="""# Project Purpose — Research Deep-Dive

## Research Question

<!-- State the central question this research aims to answer. Be specific and falsifiable. -->

>

## Hypothesis / Working Thesis

<!-- Your current best guess. This will evolve — update it as evidence accumulates. -->

>

## Background

<!-- What prior work or context motivates this research? What gap does it fill? -->

## Sub-questions

<!-- Break down the main question into tractable sub-questions. -->

1.
2.
3.
4.

## Scope

**In scope:**
-

**Out of scope:**
-

## Methodology

<!-- How will you investigate this? What types of sources or experiments are relevant? -->

-

## Success Criteria

<!-- How will you know when you have a satisfying answer? -->

-

## Current Status

> Not started — update this section as research progresses.
""",
)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 航天型号软件研制模板（定制）
# ══════════════════════════════════════════════════════════════════════════════

aerospaceTemplate = WikiTemplate(
    id="aerospace",
    name="航天型号研制",
    description="航天型号软件研制全生命周期管理",
    icon="🛰️",
    extra_dirs=[],  # 不使用专用目录，统一使用 entities/concepts/sources
    schema="""# Wiki Schema — 航天型号软件研制

## Page Types

| Type | Directory | Purpose |
|------|-----------|---------|
| entity | wiki/entities/ | 组织、产品、单机设备 |
| concept | wiki/concepts/ | 理论、方法、技术、流程 |
| source | wiki/sources/ | 原始文档（需求/设计/测试报告等） |
| query | wiki/queries/ | 开放问题 |
| comparison | wiki/comparisons/ | 对比分析 |
| synthesis | wiki/synthesis/ | 综合报告 |
| overview | wiki/ | 项目总览 |

## 研制阶段

| 阶段 | 说明 |
|------|------|
| 需求阶段 | 商务协议、技术要求、任务书、系统方案 |
| 设计阶段 | 协议设计、详细设计 |
| 研制阶段 | 固件、FPGA/CPLD、基础软件 |
| 测试阶段 | 底层测试、配置项测试、拷机测试 |
| 桌面联试 | 单机对接、网络链路、桌面联试 |
| 验收交付 | 验收测试、软件研制总结 |
| 整星AIT | 整星集成与测试 |
| 发射场 | 发射场工作手册 |

## Naming Conventions

- Files: `{ProjectPrefix}-{Title}.md` (符合项目命名规范)
- Entities: 产品型号或组织名称 (e.g., `星载智算机`, `之江首发星座`)
- Concepts: 技术术语或方法名称 (e.g., `遥测遥控协议`, `FPGA设计`)
- Sources: 原始文档名

## Frontmatter

```yaml
---
type: entity | concept | source | query | comparison | synthesis | overview
title: Human-readable title
tags: []
related: []
sources: []        # 对于 source 类型，记录源文件名
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

## Index Format

`wiki/index.md` 列出所有页面：
```
- [[page-slug]] — 一句话描述
```

## Log Format

`wiki/log.md` 记录活动：
```
## YYYY-MM-DD

- [ingest] 文档名称
```

## Cross-referencing Rules

- 使用 `[[page-slug]]` 语法链接页面
- 每页必须出现在 `index.md` 中
""",
    purpose="""# Project Purpose — 航天型号软件研制

## 项目信息

**项目名称:**
**项目编号:**
**研制单位:**
**时间范围:**

## 核心目标

本知识库服务于卫星计算载荷研制全生命周期管理，涵盖从需求对接到发射场保障的完整研制流程。

## 研制阶段

1. **需求阶段** — 商务协议、技术要求、任务书、系统方案、投产通知单、IDS
2. **设计阶段** — 协议设计（遥测/遥控/上注）、详细设计、网络化改造
3. **研制阶段** — 基础软件、固件、FPGA/CPLD、陪测系统
4. **测试阶段** — 底层测试、配置项测试、拷机测试、集成测试
5. **桌面联试** — 单机对接、网络链路、桌面联试
6. **验收交付** — 验收测试、软件研制总结、质量报告、使用说明书
7. **整星AIT** — 整星集成与测试实验
8. **发射场** — 发射场工作手册

## 关键问题

- 协议版本与设计文档的一致性如何保证？
- 跨阶段技术变更如何追溯？
- 多单位接口对接如何协调？
- 各研制阶段的交付物完整性如何确认？

## 范围

**In scope:**
- 需求阶段：商务协议、技术要求、任务书、系统方案、投产通知单
- 设计阶段：遥测遥控/上注下传/载荷接入协议、概要/详细设计、IDS
- 研制阶段：基础软件、固件、FPGA/CPLD、陪测系统
- 测试阶段：底层测试、配置项测试、拷机测试、集成测试
- 桌面联试：单机对接、网络链路、桌面联试大纲/细则/报告
- 验收交付：验收测试、软件研制总结、质量报告、使用说明书
- 整星AIT：整星集成与测试实验
- 发射场：发射场工作手册

**Out of scope:**
- 非研制流程的商务沟通记录
- 竞争对手分析或市场调研

## Thesis
航天型号软件研制是高度规范化、阶段递进的技术工程。通过结构化管理确保：
1. 技术文档端到端可追溯
2. 跨专业接口协调有据可查
3. 阶段交付物完整性受控
""",
)


# ══════════════════════════════════════════════════════════════════════════════
# 导出所有模板
# ══════════════════════════════════════════════════════════════════════════════

templates = [
    generalTemplate,
    businessTemplate,
    researchTemplate,
    aerospaceTemplate,
    # Reading 和 Personal 模板暂不包含，如有需要可添加
]


def get_template(template_id: str) -> WikiTemplate:
    """
    根据 ID 获取模板

    Args:
        template_id: 模板 ID（如 "general", "business", "aerospace"）

    Returns:
        WikiTemplate 实例

    Raises:
        ValueError: 如果模板 ID 不存在
    """
    for t in templates:
        if t.id == template_id:
            return t
    raise ValueError(f'Unknown template id: "{template_id}"')


def list_templates() -> List[WikiTemplate]:
    """返回所有可用模板"""
    return templates


# ══════════════════════════════════════════════════════════════════════════════
# 便捷函数：创建项目目录结构
# ══════════════════════════════════════════════════════════════════════════════


def create_project_structure(project_dir: str, template_id: str = "general") -> None:
    """
    根据模板创建项目目录结构和配置文件

    Args:
        project_dir: 项目根目录
        template_id: 模板 ID
    """
    import os
    from pathlib import Path

    template = get_template(template_id)
    project = Path(project_dir)

    # 创建 wiki 目录结构（直接放在 project 下，不再有 wiki/ 子目录）
    wiki_dir = project
    wiki_dir.mkdir(parents=True, exist_ok=True)

    # 创建基础 wiki 目录
    (wiki_dir / "entities").mkdir(exist_ok=True)
    (wiki_dir / "concepts").mkdir(exist_ok=True)
    (wiki_dir / "sources").mkdir(exist_ok=True)
    (wiki_dir / "queries").mkdir(exist_ok=True)
    (wiki_dir / "synthesis").mkdir(exist_ok=True)
    (wiki_dir / "comparisons").mkdir(exist_ok=True)

    # 创建额外目录（如 requirements, design 等阶段目录）
    for extra_dir in template.extra_dirs:
        # extra_dir 格式如 "wiki/requirements" 或 "requirements"
        # 统一去掉 "wiki/" 前缀
        clean_dir = extra_dir.replace("wiki/", "")
        (project / clean_dir).mkdir(parents=True, exist_ok=True)

    # 写入 purpose.md 和 schema.md
    purpose_file = project / "purpose.md"
    schema_file = project / "schema.md"

    purpose_file.write_text(template.purpose, encoding="utf-8")
    schema_file.write_text(template.schema, encoding="utf-8")

    # 创建初始 index.md
    index_file = wiki_dir / "index.md"
    index_file.write_text(
        "# Wiki Index\n\n<!-- 页面将在 ingest 后自动添加 -->\n", encoding="utf-8"
    )

    # 创建初始 log.md
    log_file = wiki_dir / "log.md"
    from datetime import datetime

    log_file.write_text(
        f"# Wiki Log\n\n## {datetime.now().strftime('%Y-%m-%d')}\n\n- [init] Project created with template: {template.name}\n",
        encoding="utf-8",
    )

    print(f"✓ Created project structure at {project}")
    print(f"  - purpose.md")
    print(f"  - schema.md")
    print(f"  - index.md")
    print(f"  - log.md")
    print(f"  - entities/, concepts/, sources/, queries/, synthesis/, comparisons/")
    for extra_dir in template.extra_dirs:
        clean_dir = extra_dir.replace("wiki/", "")
        print(f"  - {clean_dir}/")
