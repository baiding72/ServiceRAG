# Resume Notes

- Built a FastAPI-based multimodal customer-service RAG system with ChromaDB + BGE-M3 dense retrieval, BM25 lexical recall, reciprocal-rank fusion, and optional reranking.
- Implemented deterministic query understanding for product/model extraction, intent classification, sub-question splitting, and image-related routing without requiring an LLM key.
- Added image-ID evidence control so generated answers can only cite retrieved image IDs, with postprocessing to align `<PIC>` placeholders and final image arrays.
- Added hallucination guardrails including low-confidence fallback, product-conflict clarification, and OpenAI-compatible mock mode for reproducible testing.
- Built a local eval loop reporting Recall@K, MRR, image-ID recall, keyword hit rate, fallback rate, and latency from a checked-in sample dataset.
