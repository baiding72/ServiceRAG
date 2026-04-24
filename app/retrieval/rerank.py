"""Optional rerankers.

The default rule-based reranker is deterministic and dependency-light. It is
used when no cross-encoder reranker is configured or available.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from app.query_analyzer import QueryAnalysis

from .types import RetrievalResult


def _tokens(text: str) -> set[str]:
    lowered = (text or "").lower()
    words = set(re.findall(r"[a-z0-9][a-z0-9_-]{1,}", lowered))
    words.update(re.findall(r"[\u4e00-\u9fff]{2,}", text or ""))
    return words


class RuleBasedReranker:
    def score(self, query: str, result: RetrievalResult, analysis: QueryAnalysis | None = None) -> float:
        query_tokens = _tokens(query)
        haystack = " ".join(
            [
                result.get("content", ""),
                result.get("product", ""),
                result.get("section_title", ""),
                " ".join(result.get("matched_terms", []) or []),
            ]
        )
        overlap = len(query_tokens & _tokens(haystack))
        base = float(result.get("score", 0.0) or 0.0)
        dense = float(result.get("dense_score", 0.0) or 0.0)
        bm25 = float(result.get("bm25_score", 0.0) or 0.0)
        image_bonus = 0.0
        if analysis and analysis.is_image_related and result.get("image_ids"):
            image_bonus = 0.15
        product_bonus = 0.0
        if analysis:
            products = analysis.entities.get("products", [])
            if any(product.lower() in haystack.lower() for product in products):
                product_bonus = 0.1
        return base + dense + min(bm25, 1.0) + overlap * 0.03 + image_bonus + product_bonus

    def rerank(
        self,
        query: str,
        results: Iterable[RetrievalResult],
        top_k: int,
        analysis: QueryAnalysis | None = None,
    ) -> List[RetrievalResult]:
        enriched = []
        for item in results:
            copied = dict(item)
            copied["rerank_score"] = self.score(query, copied, analysis)
            enriched.append(copied)
        return sorted(enriched, key=lambda x: (-x.get("rerank_score", 0.0), -x.get("score", 0.0)))[:top_k]

