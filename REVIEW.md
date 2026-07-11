# rag-clean 代码审查报告

> 基于 Karpathy 原则：简洁优先、只保留真正需要的代码、不做过度抽象。

审查日期：2026-05-26
分支：`review/karpathy-simplify`

## 一、总览

| 指标 | 数值 |
|------|------|
| 项目总行数 (Python) | ~24,360 |
| 完全死代码 | ~4,275 行 (17.5%) |
| 已废弃代码 (仅eval脚本使用) | ~1,570 行 (6.4%) |
| 可移除/归档总计 | **~5,845 行 (24%)** |

## 二、核心发现

### 发现 1：生产环境只用 SimplePipeline，RAGPipeline 已废弃

**现状：**
- `api/main.py` **只使用** `SimplePipeline` (389行)
- `RAGPipeline` (675行) 只被 4 个 eval 脚本引用
- 两个 Pipeline 之间 **复制粘贴了 ~165 行** 父块展开代码

**问题：**
- RAGPipeline 是"旧版完整流水线"，包含 QueryUnderstanding + QueryRewrite + SemanticRouter
- SimplePipeline 跳过这些步骤，直接用 HyDE + 检索
- 两者并存造成混淆

**建议：**
- 如果还需要 eval 脚本，将 RAGPipeline 移到 `scripts/legacy/` 下
- 如果不再需要，直接删除，同时删除依赖它的 4 个 eval 脚本
- 将父块展开代码提取为共享函数，消除 165 行重复

---

### 发现 2：core/preprocessing/ 整包是离线工具，运行时完全不使用

**文件 (3,421 行)：**
- `entity_extractor.py` (601行) - 实体抽取
- `entity_cleaner.py` (574行) - 实体清洗（零引用，完全死代码）
- `struct_extractor.py` (763行) - 结构化抽取
- `ontology_builder.py` (357行) - 本体构建
- `run_pipeline.py` (508行) - CLI 入口
- `chunker_ext.py` (371行) - 扩展分块
- `cleaner_ext.py` (201行) - 扩展清洗

**问题：**
- 运行时 API / Pipeline / Agent **没有一个导入此包**
- 只能通过 `python -m core.preprocessing.scripts.run_pipeline` 手动执行
- 放在 `core/` 下暗示是核心运行时组件，实际不是

**建议：**
- 移到 `scripts/preprocessing/` 或 `tools/preprocessing/`，明确其离线工具的定位
- 或如果实验已结束，直接删除

---

### 发现 3：core/router/ 几乎死代码

**文件 (295 行)：**
- `semantic_router.py` (163行) - 只被 `query_rewrite.py` 懒加载引用（而 query_rewrite 本身也仅 RAGPipeline 使用）
- `intent_prototypes.py` (71行) - 只被 semantic_router 引用
- `models.py` (33行) - 只有 4 个字符串常量被 `generation.py` 使用
- `__init__.py` (28行) - 无人导入

**问题：**
- 4 个 intent 常量 (`INTENT_SIMPLE_LOOKUP` 等) 放在 router 包里，导致 `generation.py` 被迫引入整个路由模块

**建议：**
- 将 4 个常量移到 `prompt.py` 或 `core/model/models.py`
- 删除 `semantic_router.py`、`intent_prototypes.py`

---

### 发现 4：query_understanding + query_rewrite 仅被废弃的 RAGPipeline 使用

- `query_understanding.py` (148行) - 子问题拆分 + 意图分类
- `query_rewrite.py` (452行) - 查询改写 + 同义词扩展

**两者都是 RAGPipeline 的组件。SimplePipeline 不用它们。**

**建议：**
- 随 RAGPipeline 一起移除或归档

---

### 发现 5：products/ 存在两套规格查询实现

| 文件 | 行数 | 使用者 |
|------|------|--------|
| `spec_matcher.py` | 574 | RAGPipeline (已废弃) |
| `specs_service.py` | 224 | SimplePipeline (生产) |

**问题：**
- 两套实现加载同一个 `products_specs.json`，做类似的事
- `spec_matcher.py` 有 574 行但只被废弃的 RAGPipeline 使用

**建议：**
- 保留 `specs_service.py` (更简单)
- `spec_matcher.py` 随 RAGPipeline 一起移除

---

### 发现 6：检索层 API 设计问题

**现状：**
- `RetrievalService.search()` 是高级接口（接受原始查询，内部做关键词提取+同义词+HyDE+编码）
- 但 **两个 Pipeline 都不调用** `search()`，而是直接调用私有方法 `_hybrid_search()`
- `retrieval copy.py` (529行) 是死文件，无人导入

**问题：**
- 公共 API 没人用，所有调用者都绕过它
- 这说明 `search()` 的抽象层级不对——它假设 Pipeline 不做预处理，但实际 Pipeline 想自己控制预处理流程

**建议：**
- 删除 `retrieval copy.py`
- 重新设计 `RetrievalService` 的公共 API，让 `_hybrid_search` 级别的方法成为正式接口

---

### 发现 7：GenerationService 不复用 LLMClient

**现状：**
- `core/generation/llm.py` 有 `LLMClient`（懒加载单例，8+ 模块使用）
- `core/generation/generation.py` 的 `generate()` 和 `generate_stream()` 各自 `openai.OpenAI(...)` 创建新客户端

**问题：**
- 绕过了 `LLMClient` 的单例模式
- 如果需要修改 LLM 配置，需要改两个地方

**建议：**
- `GenerationService` 应该使用 `LLMClient` 而不是自己创建客户端

---

### 发现 8：死文件和散落的杂项

| 文件 | 问题 |
|------|------|
| `batch_upload.py` (56行) | 上传到 RAGFlow 的脚本，是旧系统遗留，**包含硬编码 API key** |
| `ui/chat_backend.py` (269行) | Chainlit 前端后端，Chainlit 应用已不存在，**零引用** |
| `core/retrieve/retrieval copy.py` (529行) | 孤立备份文件 |
| `vllm_serve.log` + `scripts/vllm_serve.log` | 散落的日志文件 (2.3MB) |
| `run.py --frontend` | 引用不存在的 `ui/chainlit_app.py` |

---

## 三、架构简化建议

### 当前模块依赖（运行时）

```
api/main.py
├── core/pipeline/simple_pipeline.py  ← 主流水线
│   ├── core/client/embedder.py       ← 向量编码
│   ├── core/retrieve/retrieval.py    ← 混合检索 + Rerank
│   │   ├── core/query_engineer/keyword_extractor.py
│   │   └── core/query_engineer/synonym.py
│   ├── core/products/specs_service.py ← 规格查询
│   ├── core/generation/generation.py  ← 答案生成
│   │   └── core/generation/llm.py     ← LLM 客户端
│   └── core/query_engineer/hyde.py    ← HyDE
├── core/agent/react_agent.py          ← ReAct Agent
│   ├── core/agent/tools.py
│   └── core/generation/llm.py
└── core/wiki/wiki_graph.py            ← Wiki 知识图谱
```

### 可移除的模块

```
❌ core/preprocessing/           → 移到 scripts/ 或删除 (3,421行)
❌ core/router/semantic_router   → 随 RAGPipeline 删除
❌ core/router/intent_prototypes → 随 RAGPipeline 删除
❌ core/query_engineer/query_understanding.py → 随 RAGPipeline 删除
❌ core/query_engineer/query_rewrite.py       → 随 RAGPipeline 删除
❌ core/pipeline/rag_pipeline.py              → 移到 scripts/ 或删除 (675行)
❌ core/products/spec_matcher.py              → 随 RAGPipeline 删除
❌ core/retrieve/retrieval copy.py            → 删除 (529行)
❌ batch_upload.py                            → 删除 (含硬编码 key)
❌ ui/chat_backend.py                         → 删除
```

### 需要修复的问题

```
⚠️ simple_pipeline.py 复制了 rag_pipeline.py 的 165 行父块展开代码 → 提取共享函数
⚠️ generation.py 不复用 LLMClient → 改为使用 get_llm_client()
⚠️ 4 个 intent 常量从 core/router/models.py 移到 prompt.py
⚠️ RetrievalService 公共 API 没人用 → 重新设计
⚠️ run.py --frontend 引用不存在的文件 → 修复或移除该选项
```

---

## 四、推荐执行步骤

### Phase 1：删除明确死代码（无风险）
1. 删除 `core/retrieve/retrieval copy.py`
2. 删除 `batch_upload.py`
3. 删除 `ui/chat_backend.py`
4. 删除 `scripts/vllm_serve.log`
5. 删除 `core/preprocessing/entity_cleaner.py`（零引用）

### Phase 2：归档离线工具包
6. 将 `core/preprocessing/` 移到 `tools/preprocessing/`
7. 将 `run.py --frontend` 选项移除或修复

### Phase 3：清理废弃的 RAGPipeline 及其依赖
8. 移动 `core/pipeline/rag_pipeline.py` → `scripts/legacy/rag_pipeline.py`
9. 移动依赖它的 4 个 eval 脚本到 `scripts/legacy/`
10. 删除 `core/query_engineer/query_understanding.py`
11. 删除 `core/query_engineer/query_rewrite.py`
12. 删除 `core/router/semantic_router.py` + `intent_prototypes.py`
13. 将 intent 常量移到 `prompt.py`
14. 删除 `core/products/spec_matcher.py`

### Phase 4：代码质量修复
15. 提取父块展开为共享函数，消除 simple_pipeline 和 rag_pipeline 的重复
16. 修复 `generation.py` 使用 `LLMClient` 而非自建客户端
17. 重新设计 `RetrievalService` 公共 API

---

## 五、简化后的预期项目结构

```
rag-clean/
├── api/                    # 4个服务入口（不变）
├── config.py               # 配置（不变）
├── prompt.py               # 提示词 + intent 常量（扩展）
├── run.py                  # 启动脚本（修复）
├── core/
│   ├── agent/              # ReAct Agent（不变）
│   ├── client/             # Embedding + Rerank 客户端（不变）
│   ├── generation/         # LLM Client + Generation（修复复用）
│   ├── ingestion/          # 文档处理（不变）
│   ├── model/              # 数据模型（不变）
│   ├── pipeline/           # 只有 SimplePipeline
│   ├── products/           # 只有 specs_service
│   ├── query_engineer/     # hyde, keyword_extractor, synonym, rerank_query, term_weight
│   ├── retrieve/           # retrieval + retrieval_models（修复 API）
│   └── wiki/               # Wiki 模块（不变）
├── scripts/
│   ├── active/             # 生产工具脚本
│   └── legacy/             # 废弃的 eval 脚本 + RAGPipeline
└── tools/
    └── preprocessing/      # 离线预处理工具
```

预计 Python 代码从 **~24,360 行** 减少到 **~18,500 行**，减少 **24%**。
