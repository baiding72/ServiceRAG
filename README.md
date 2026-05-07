# ServiceRAG: 多模态客服 Agent RAG 系统

ServiceRAG 是一个面向电商/客服场景的 RAG 系统，保留比赛要求的 `/chat` API、Bearer Token 鉴权和 `data.answer` 输出格式，同时将检索、Query Understanding、图文证据、回答防幻觉、会话记忆和评测闭环拆成可复现模块。

## 已实现能力

- FastAPI `/chat` 服务，兼容原始请求/响应 schema。
- ChromaDB + BGE-M3 Dense 检索，保留 `retriever.py` 兼容脚本。
- BM25 召回、Dense + BM25 融合、可选 cross-encoder rerank、规则 rerank fallback。
- 规则 Query Analyzer：语言、意图、产品/型号、按钮/屏幕/指示灯、图片相关词、多子问题拆分。
- 图片 ID 证据融合：只允许引用检索证据里的 image_id，后处理对齐 `<PIC>` 和图片数组。
- Hallucination guard：低置信度 fallback、产品冲突澄清、无 key mock/fallback。
- 内存型 session memory：支持简单追问继承，带 TTL 和最大轮数。
- 本地 eval：Recall@K、MRR、image_id_recall、keyword hit、fallback rate、latency。
- pytest 基础覆盖：API、鉴权、query analyzer、postprocess、memory、retrieval result schema。

## 项目结构

```text
ServiceRAG/
├── main.py                         # FastAPI 入口，保持 /chat 兼容
├── retriever.py                    # 旧脚本兼容层：Chroma dense + BM25 + rerank
├── build_vector_db.py              # 向量库构建
├── parse_manuals.py                # 手册解析与 chunking
├── app/
│   ├── config.py                   # 环境变量配置
│   ├── schemas.py                  # Pydantic 请求/响应模型
│   ├── query_analyzer.py           # 规则 Query Understanding
│   ├── evidence.py                 # 文本/图片证据组织
│   ├── guard.py                    # 防幻觉与降级
│   ├── llm.py                      # OpenAI-compatible client + mock
│   ├── memory.py                   # session memory
│   ├── postprocess.py              # 输出格式与图片 ID 规范化
│   ├── retrieval/                  # Dense / BM25 / Hybrid / Rerank
│   └── evaluation/run_eval.py      # 小型可复现 eval
├── eval/eval_dataset.jsonl          # 小型样例评测集
├── tests/                          # pytest
└── docs/                           # 架构、检索、评测、简历说明
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置

复制 `.env.example` 后按需设置环境变量：

```bash
cp .env.example .env
```

关键配置：

- `AUTH_TOKEN` 或 `KAFU_API_TOKEN`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL_NAME`
- `LLM_MOCK_MODE`
- `CHROMA_DB_PATH`
- `CHROMA_COLLECTION`
- `EMBEDDING_MODEL_NAME`
- `RETRIEVAL_TOP_K`
- `HYBRID_RETRIEVAL_ENABLED`
- `RERANKER_ENABLED`
- `SKIP_IMAGE_RETRIEVER`
- `ENABLE_VL_RERANK`

默认不会在代码中内置真实 API key。未配置 `LLM_API_KEY` 或 `LLM_MOCK_MODE=true` 时，系统进入 deterministic mock/fallback，核心测试仍可运行。

## 建库

```bash
python parse_manuals.py
python build_vector_db.py
```

默认使用：

- `EMBEDDING_MODEL_NAME=BAAI/bge-m3`
- `CHROMA_DB_PATH=./data/chroma_db_m3`
- `CHROMA_COLLECTION=manuals_qa_m3`

向量库属于大体积二进制产物，默认通过 `.gitignore` 忽略。需要跨机器同步时建议使用对象存储、rsync 或明确 `git add -f`，不要混入日常代码提交。

## 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://localhost:8000/health
```

请求 `/chat`：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer kafu_test_token_2024" \
  -H "Content-Type: application/json" \
  -d '{"question":"我的DCB107电钻指示灯闪烁是什么意思？","images":[],"session_id":"demo","stream":false}'
```

## API 兼容性

请求体保持兼容：

- `question`
- `images`
- `session_id`
- `stream`

响应体保持兼容：

- `code`
- `msg`
- `data.answer`
- `data.session_id`
- `data.timestamp`

`data.answer` 保持比赛格式：

- 无图：`回答文本`
- 有图：`回答文本 <PIC> , ["image_id"]`

默认不返回 debug 字段，避免破坏官方评测。

## 测试

```bash
LLM_MOCK_MODE=true pytest
```

## 本地 Eval

```bash
LLM_MOCK_MODE=true python -m app.evaluation.run_eval --limit 20
```

输出：

- `eval_reports/latest.json`
- `eval_reports/latest.md`

该 eval 只作为本地回归指标，不代表官方榜单分数。

## 文档

- [Architecture](docs/architecture.md)
- [Retrieval Pipeline](docs/retrieval_pipeline.md)
- [Evaluation](docs/evaluation.md)
- [Resume Notes](docs/resume_notes.md)

## Roadmap

- 将视觉图片 rerank 切到可配置的真实 VL reranker，并做成本/延迟开关。
- 引入更严格的人工 gold set 或 pooled LLM-as-a-judge eval。
- 将内存 session store 替换为 Redis/SQLite 以支持多进程部署。
- 针对官方 badcase 做生成质量消融，而不是只看 retrieval recall。
