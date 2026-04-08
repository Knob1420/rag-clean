# RAG Clean

企业级知识库检索增强生成（RAG）系统，支持多格式文档导入、智能分块、混合检索和 LLM 问答。

## 核心特性

- **多格式支持**：PDF、DOC、DOCX、PPTX、Markdown
- **智能分块**：LLM 驱动的层次化分块，含父子导航关系
- **混合检索**：BM25 全文检索 + 向量语义检索 + RRF 融合
- **查询路由**：自动识别查询类型（简单/对比/多步推理），选择最优检索策略
- **ReAct 多跳推理**：支持复杂多步问答
- **重排序**：BGE-reranker-v2-m3 交叉编码器精排
- **微服务架构**：各组件独立部署、按需启停

## 系统架构

```
┌─────────────────────┐
│   Gradio 前端 (7860) │
└──────────┬──────────┘
           │ HTTP
┌──────────▼──────────┐
│   主 API (8000)     │
│  Query → Rewrite →  │
│  [路由] → 检索 →    │
│  Rerank → LLM 生成  │
└──────────┬──────────┘
           │
     ┌─────┼─────┬─────────┐
     ▼     ▼     ▼         ▼
   ES   Embed   Rerank   MinerU
  :9200  :8001   :8002    :8003
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- Elasticsearch 8.x

### 2. 安装依赖

```bash
pip install fastapi uvicorn elasticsearch pydantic pydantic-settings
pip install loguru httpx gradio numpy
pip install FlagEmbedding openai python-docx markdown tqdm
```

### 3. 配置

复制 `.env` 文件并配置 API Key：

```env
DEEPSEEK_API_KEY=your-api-key
ES_URL=http://localhost:9200
```

### 4. 启动服务

```bash
# 分别启动各服务
python run.py --embedding  # Embedding 服务 (8001)
python run.py --rerank    # Rerank 服务 (8002)
python run.py --main      # 主 API (8000)
python run.py --frontend   # 前端界面 (7860)
```

### 5. 导入文档

```bash
python batch_import.py --dry-run  # 预览模式
python batch_import.py            # 导入所有文档
```

### 6. 问答

访问 http://localhost:7860 或调用 API：

```bash
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"query": "你的问题", "top_k": 10}'
```

## 项目结构

```
rag-clean/
├── api/              # FastAPI 微服务
├── core/            # 核心模块
│   ├── ingestion/   # 文档处理管道
│   ├── query_engineer/  # 查询重写、路由、ReAct
│   ├── retrieve/    # 混合检索
│   └── generation/  # LLM 生成
├── ui/              # Gradio 前端
├── eval/            # 评估框架
├── scripts/         # 工具脚本
├── docs/            # 详细文档
└── data/            # 数据目录
```

## 详细文档

见 [docs/user-guide.md](docs/user-guide.md)

## License

MIT
