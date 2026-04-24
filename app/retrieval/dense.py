"""Dense retriever adapter."""

from __future__ import annotations

from typing import List, Optional

from .types import RetrievalResult, normalize_result


class DenseRetriever:
    def __init__(self, manual_retriever):
        self.manual_retriever = manual_retriever

    def search(self, query: str, top_k: int = 10, where_filter: Optional[dict] = None) -> List[RetrievalResult]:
        results = self.manual_retriever.search_semantic(query, top_k=top_k, where_filter=where_filter)
        normalized = []
        for rank, item in enumerate(results, 1):
            item = dict(item)
            item["dense_score"] = 1.0 / rank
            normalized.append(normalize_result(item, route="dense"))
        return normalized

