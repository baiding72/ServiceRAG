# Architecture

ServiceRAG keeps the original FastAPI `/chat` entrypoint and moves reusable logic into small modules.

```mermaid
flowchart LR
  A["POST /chat"] --> B["Bearer auth"]
  B --> C["QueryAnalyzer"]
  C --> D["SessionMemory"]
  D --> E["HybridRetriever"]
  E --> F["EvidenceBuilder"]
  F --> G["HallucinationGuard"]
  G --> H["LLMClient or Mock"]
  H --> I["Postprocess"]
  I --> J["ChatResponse"]
```

Core modules:

- `app/config.py`: environment-driven runtime config.
- `app/query_analyzer.py`: rule-based intent, language, entity, and sub-question detection.
- `app/retrieval/`: adapters for Dense, BM25, Hybrid, and rule-based rerank.
- `app/evidence.py`: prompt context and image evidence formatting.
- `app/guard.py`: low-confidence fallback and product-conflict clarification.
- `app/llm.py`: OpenAI-compatible client with deterministic mock fallback.
- `app/postprocess.py`: answer cleanup and `<PIC>` / image-array alignment.
- `app/memory.py`: lightweight session memory with TTL.

The default response schema remains compatible with the competition API. Debug metadata is kept out of the default response.

