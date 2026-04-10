# 多模态客服智能体 RAG 系统

基于 FastAPI + ChromaDB + LLM 的智能客服问答系统，用于多模态客服智能体比赛。

## 功能特性

- ✅ Bearer Token 鉴权
- ✅ 向量语义检索（ChromaDB + BGE-small-zh）
- ✅ LLM 生成回答（兼容 OpenAI 格式）
- ✅ 多模态图片 ID 关联
- ✅ 降级模式（无 LLM 时自动使用检索结果）
- ✅ 完整的异常处理

## 项目结构

```
ServiceRAG/
├── main.py              # FastAPI 主程序（RAG + LLM 闭环）
├── retriever.py         # 向量检索器封装
├── build_vector_db.py   # 向量数据库构建脚本
├── parse_manuals.py     # 手册解析脚本
├── knowledge.py         # Mock 知识库（已弃用）
├── requirements.txt     # 依赖清单
├── README.md            # 项目说明
└── data/
    ├── manuals/                  # 原始手册目录
    ├── structured_knowledge.json # 结构化知识库
    └── chroma_db/                # 向量数据库
```

## 环境要求

- Python 3.10+
- uv 包管理器（推荐）

## 安装部署

### 1. 安装依赖

```bash
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

### 2. 构建向量数据库

```bash
# 解析手册（如已解析可跳过）
python parse_manuals.py

# 构建向量数据库
python build_vector_db.py
```

### 3. 配置 LLM API（重要！）

编辑 `main.py` 头部的配置常量：

```python
# ============================================
# 🔧 配置常量区（请在此处填入您的配置）
# ============================================

# LLM API 配置
LLM_API_KEY = "your-api-key-here"       # 大模型 API Key
LLM_BASE_URL = "https://api.openai.com/v1"  # API 地址
LLM_MODEL_NAME = "gpt-3.5-turbo"        # 模型名称
```

或通过环境变量配置：

```bash
export LLM_API_KEY="your-api-key-here"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL_NAME="gpt-3.5-turbo"
```

### 4. 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API 接口说明

### POST /chat

客服问答核心接口。

**请求头：**

| Header | 值 | 说明 |
|--------|-----|------|
| Authorization | Bearer {Token} | 必填，鉴权 Token |
| Content-Type | application/json | 必填 |

**请求体：**

```json
{
  "question": "请问电钻指示灯闪烁是什么意思？",
  "images": [],
  "session_id": "test-session-001",
  "stream": false
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| question | string | 是 | 用户的问题文本 |
| images | array | 否 | Base64 格式图片列表 |
| session_id | string | 否 | 会话 ID，不传则自动生成 |
| stream | boolean | 否 | 是否流式输出，默认 false |

**响应体：**

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "DCB107、DCB112电池组充电中 <PIC> 电池组已充满 <PIC> , [\"drill0_04\", \"drill0_05\"]",
    "session_id": "test-session-001",
    "timestamp": 1741008000
  }
}
```

## 测试命令

### 健康检查

```bash
curl -X GET "http://localhost:8000/health"
```

### 测试问答接口

```bash
curl -X POST "http://localhost:8000/chat" -H "Authorization: Bearer kafu_test_token_2024" -H "Content-Type: application/json" -d '{"question": "电钻指示灯闪烁是什么意思？"}'
```

### 测试鉴权失败

```bash
curl -X POST "http://localhost:8000/chat" -H "Authorization: Bearer wrong_token" -H "Content-Type: application/json" -d '{"question": "测试"}'
```

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         用户请求                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI /chat 接口                        │
│                    1. Bearer Token 鉴权                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Step 1: 向量检索                             │
│              retriever.search(question, top_k=3)             │
│          ┌──────────────────────────────────────┐           │
│          │   ChromaDB (cosine similarity)       │           │
│          │   BGE-small-zh Embedding Model       │           │
│          └──────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Step 2: 构建 Context                         │
│          拼接检索结果 + 图片 ID 信息                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Step 3: 调用 LLM                             │
│          System Prompt + User Prompt + Context              │
│          ┌──────────────────────────────────────┐           │
│          │   OpenAI API (兼容格式)               │           │
│          └──────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Step 4: 返回响应                             │
│          { code: 0, msg: "success", data: {...} }           │
└─────────────────────────────────────────────────────────────┘
```

## 降级模式

当 LLM API 未配置或调用失败时，系统自动进入降级模式：

1. 检索相关文档
2. 如果检索结果相关度较高（distance < 0.5），返回检索内容
3. 否则返回兜底回复

## 支持的产品

- VR头显、人体工学椅、健身单车、健身追踪器
- 儿童电动摩托车、冰箱、功能键盘、发电机
- 可编程温控器、吹风机、摩托艇、水泵
- 洗碗机、烤箱、电钻、相机、空气净化器
- 空调、蒸汽清洁机、蓝牙激光鼠标

## 常见问题

### Q: 服务启动很慢？

首次启动需要加载 Embedding 模型（约 100MB），后续启动会快很多。

### Q: LLM 回复不符合预期？

检查 System Prompt 是否正确设置，确保 LLM 理解输出格式要求。

### Q: 如何更换 Embedding 模型？

修改 `retriever.py` 和 `build_vector_db.py` 中的 `EMBEDDING_MODEL_NAME`。

## License

MIT
