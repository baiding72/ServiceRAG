"""BM25 retriever adapter for the existing ManualRetriever implementation."""

from __future__ import annotations

from typing import List, Optional

from .types import RetrievalResult, normalize_result


class BM25Retriever:
    def __init__(self, manual_retriever):
        self.manual_retriever = manual_retriever

    def search(self, query: str, top_k: int = 10, where_filter: Optional[dict] = None) -> List[RetrievalResult]:
        results = self.manual_retriever.search_bm25(query, top_k=top_k, where_filter=where_filter)
        return [normalize_result(item, route="bm25") for item in results]

