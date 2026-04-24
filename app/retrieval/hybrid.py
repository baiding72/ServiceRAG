"""Dense + BM25 hybrid retrieval pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional

from app.config import Settings, settings
from app.query_analyzer import QueryAnalysis, analyze_query

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .rerank import RuleBasedReranker
from .types import RetrievalResult, normalize_result


class HybridRetriever:
    def __init__(self, manual_retriever, config: Settings = settings):
        self.manual_retriever = manual_retriever
        self.config = config
        self.dense = DenseRetriever(manual_retriever)
        self.bm25 = BM25Retriever(manual_retriever)
        self.rule_reranker = RuleBasedReranker()

    def _merge(self, dense_results: List[RetrievalResult], bm25_results: List[RetrievalResult]) -> List[RetrievalResult]:
        merged: Dict[str, RetrievalResult] = {}
        rrf_k = self.config.rrf_k

        for rank, item in enumerate(dense_results, 1):
            key = item.get("chunk_id") or f"dense-{rank}"
            target = merged.setdefault(key, dict(item))
            target["dense_score"] = max(float(target.get("dense_score", 0.0)), 1.0 / rank)
            target["score"] = float(target.get("score", 0.0)) + 1.0 / (rrf_k + rank)
            target["route"] = (target.get("route") or "") + "|dense"

        for rank, item in enumerate(bm25_results, 1):
            key = item.get("chunk_id") or f"bm25-{rank}"
            target = merged.setdefault(key, dict(item))
            target["bm25_score"] = max(float(target.get("bm25_score", 0.0)), float(item.get("bm25_score", 0.0)))
            target["score"] = float(target.get("score", 0.0)) + 1.0 / (rrf_k + rank)
            target["route"] = (target.get("route") or "") + "|bm25"
            if not target.get("content") and item.get("content"):
                target.update(item)

        return sorted(
            [normalize_result(item, route=item.get("route", "hybrid")) for item in merged.values()],
            key=lambda x: (-x.get("score", 0.0), x.get("distance", 999.0), -x.get("bm25_score", 0.0)),
        )

    def search(
        self,
        query: str | QueryAnalysis,
        top_k: Optional[int] = None,
        where_filter: Optional[dict] = None,
    ) -> List[RetrievalResult]:
        analysis = query if isinstance(query, QueryAnalysis) else analyze_query(str(query))
        query_texts = [analysis.rewritten_query]
        for sub_question in analysis.sub_questions:
            if sub_question not in query_texts:
                query_texts.append(sub_question)

        top_k = top_k or self.config.retrieval_top_k
        pool: Dict[str, RetrievalResult] = {}
        for query_text in query_texts:
            dense_results = self.dense.search(query_text, top_k=self.config.semantic_candidate_k, where_filter=where_filter)
            bm25_results = []
            if self.config.hybrid_retrieval_enabled:
                bm25_results = self.bm25.search(query_text, top_k=self.config.bm25_candidate_k, where_filter=where_filter)
            for item in self._merge(dense_results, bm25_results):
                key = item.get("chunk_id") or item.get("content", "")[:80]
                existing = pool.get(key)
                if existing is None or item.get("score", 0.0) > existing.get("score", 0.0):
                    pool[key] = item

        candidates = sorted(pool.values(), key=lambda x: (-x.get("score", 0.0), x.get("distance", 999.0)))
        if self.config.reranker_enabled:
            try:
                legacy = [
                    {
                        **item,
                        "images": item.get("image_ids", []),
                        "retrieval_score": item.get("score", 0.0),
                    }
                    for item in candidates
                ]
                reranked = self.manual_retriever.rerank_results(
                    analysis.rewritten_query,
                    legacy,
                    top_k=max(top_k, self.config.retrieval_top_k),
                )
                return [normalize_result(item, route="hybrid|cross_encoder") for item in reranked[:top_k]]
            except Exception:
                return self.rule_reranker.rerank(analysis.rewritten_query, candidates, top_k, analysis)
        return candidates[:top_k]

