# Retrieval Pipeline

The implemented retrieval flow is:

1. Analyze query language, intent, entities, image-related terms, and sub-questions.
2. Run Dense retrieval through ChromaDB + BGE-M3.
3. Run BM25 retrieval over the same manual corpus.
4. Merge candidates with reciprocal-rank fusion.
5. Optionally rerank with the existing cross-encoder; if unavailable, use rule-based rerank.
6. Preserve provenance fields: `chunk_id`, `product`, `section_title`, `score`, `image_ids`, and route metadata.

Image-aware behavior:

- Queries mentioning image, button, screen, indicator, location, size, diagram, or equivalent Chinese terms receive a small ranking preference for chunks with image IDs.
- The answer generator may only reference image IDs already present in retrieved evidence.
- Postprocess removes illegal image IDs and aligns `<PIC>` count with the final image array.

The legacy `retriever.py` remains available for existing scripts. New code should prefer `app.retrieval.HybridRetriever`.

