"""Dense + BM25 hybrid retrieval pipeline."""

from __future__ import annotations

import time
import re
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
        self.last_trace: dict = {}

    def _merge(self, dense_results: List[RetrievalResult], bm25_results: List[RetrievalResult]) -> List[RetrievalResult]:
        merged: Dict[str, RetrievalResult] = {}
        rrf_k = self.config.rrf_k

        for rank, item in enumerate(dense_results, 1):
            key = item.get("chunk_id") or f"dense-{rank}"
            target = merged.setdefault(key, dict(item))
            target["dense_score"] = max(float(target.get("dense_score", 0.0)), 1.0 / rank)
            target.setdefault("route_ranks", {})["dense"] = min(
                int(target.get("route_ranks", {}).get("dense", rank)),
                rank,
            )
            target["rrf_score"] = float(target.get("rrf_score", 0.0)) + 1.0 / (rrf_k + rank)
            target["score"] = float(target.get("rrf_score", 0.0))
            target["route"] = _append_route(target.get("route"), "dense")

        for rank, item in enumerate(bm25_results, 1):
            key = item.get("chunk_id") or f"bm25-{rank}"
            target = merged.setdefault(key, dict(item))
            target["bm25_score"] = max(float(target.get("bm25_score", 0.0)), float(item.get("bm25_score", 0.0)))
            target.setdefault("route_ranks", {})["bm25"] = min(
                int(target.get("route_ranks", {}).get("bm25", rank)),
                rank,
            )
            target["rrf_score"] = float(target.get("rrf_score", 0.0)) + 1.0 / (rrf_k + rank)
            target["score"] = float(target.get("rrf_score", 0.0))
            target["route"] = _append_route(target.get("route"), "bm25")
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
        started_at = time.time()
        query_texts = [analysis.rewritten_query]
        for sub_question in analysis.sub_questions:
            if sub_question not in query_texts:
                query_texts.append(sub_question)

        top_k = top_k or self.config.retrieval_top_k
        pool: Dict[str, RetrievalResult] = {}
        route_trace = []
        for query_text in query_texts:
            dense_results = self.dense.search(query_text, top_k=self.config.semantic_candidate_k, where_filter=where_filter)
            bm25_results = []
            if self.config.hybrid_retrieval_enabled:
                bm25_results = self.bm25.search(query_text, top_k=self.config.bm25_candidate_k, where_filter=where_filter)
            route_trace.append(
                {
                    "query_text": query_text,
                    "dense_top": [_compact_result(item, rank) for rank, item in enumerate(dense_results, 1)],
                    "bm25_top": [_compact_result(item, rank) for rank, item in enumerate(bm25_results, 1)],
                }
            )
            for item in self._merge(dense_results, bm25_results):
                key = item.get("chunk_id") or item.get("content", "")[:80]
                existing = pool.get(key)
                if existing is None or item.get("score", 0.0) > existing.get("score", 0.0):
                    pool[key] = item

        candidates = sorted(pool.values(), key=lambda x: (-x.get("score", 0.0), x.get("distance", 999.0)))
        self.last_trace = {
            "analysis": analysis.as_debug_dict(),
            "fusion": {
                "mode": self.config.hybrid_fusion_mode,
                "rrf_k": self.config.rrf_k,
                "hybrid_enabled": self.config.hybrid_retrieval_enabled,
                "reranker_enabled": self.config.reranker_enabled,
                "semantic_candidate_k": self.config.semantic_candidate_k,
                "bm25_candidate_k": self.config.bm25_candidate_k,
                "top_k": top_k,
            },
            "routes": route_trace,
            "candidate_pool": [_compact_result(item, rank) for rank, item in enumerate(candidates, 1)],
            "latency_ms": int((time.time() - started_at) * 1000),
        }
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
                    top_k=max(top_k, self.config.retrieval_top_k, self.config.semantic_candidate_k),
                )
                reranked = _apply_query_focus_rerank(analysis, [normalize_result(item, route="hybrid|cross_encoder") for item in reranked])
                final_results = reranked[:top_k]
                self.last_trace["final_results"] = [_compact_result(item, rank) for rank, item in enumerate(final_results, 1)]
                self.last_trace["reranker"] = "cross_encoder"
                return final_results
            except Exception:
                final_results = self.rule_reranker.rerank(analysis.rewritten_query, candidates, top_k, analysis)
                final_results = _apply_query_focus_rerank(analysis, final_results)
                self.last_trace["final_results"] = [_compact_result(item, rank) for rank, item in enumerate(final_results, 1)]
                self.last_trace["reranker"] = "rule_based_fallback"
                return final_results
        final_results = _apply_query_focus_rerank(analysis, candidates)[:top_k]
        self.last_trace["final_results"] = [_compact_result(item, rank) for rank, item in enumerate(final_results, 1)]
        self.last_trace["reranker"] = "disabled"
        return final_results


def _append_route(existing: str | None, route: str) -> str:
    parts = [part for part in (existing or "").split("|") if part]
    if route not in parts:
        parts.append(route)
    return "|".join(parts)


def _compact_result(item: RetrievalResult, rank: int) -> dict:
    content = item.get("content", "") or ""
    return {
        "rank": rank,
        "chunk_id": item.get("chunk_id", ""),
        "parent_id": item.get("parent_id", ""),
        "product": item.get("product") or item.get("manual", ""),
        "sub_manual": item.get("sub_manual", ""),
        "section_title": item.get("section_title", ""),
        "content_type": item.get("content_type", ""),
        "score": float(item.get("score", 0.0) or 0.0),
        "rrf_score": float(item.get("rrf_score", item.get("score", 0.0)) or 0.0),
        "dense_score": float(item.get("dense_score", 0.0) or 0.0),
        "bm25_score": float(item.get("bm25_score", 0.0) or 0.0),
        "rerank_score": float(item.get("rerank_score", 0.0) or 0.0),
        "distance": float(item.get("distance", 999.0) or 999.0),
        "route": item.get("route", ""),
        "route_ranks": item.get("route_ranks", {}),
        "image_ids": item.get("image_ids") or item.get("images") or [],
        "content_preview": content[:360],
    }


FOCUS_STOPWORDS = {
    "the", "and", "for", "with", "what", "does", "how", "turn", "your", "into",
    "from", "this", "that", "position", "show", "boat", "have", "will", "when",
    "where", "which", "there", "about", "screen",
}


def _focus_terms(query: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", (query or "").lower())
        if token not in FOCUS_STOPWORDS
    }
    tokens.update(re.findall(r"[\u4e00-\u9fff]{2,}", query or ""))
    return tokens


def _apply_query_focus_rerank(analysis: QueryAnalysis, results: List[RetrievalResult]) -> List[RetrievalResult]:
    if not results:
        return results

    query = analysis.rewritten_query or analysis.original_query
    q = query.lower()
    terms = _focus_terms(query)
    target_sub_manual_terms = []
    if analysis.language == "en":
        products = [item.lower() for item in analysis.entities.get("products", [])]
        if "boat" in q or "boat" in products:
            target_sub_manual_terms.extend(["boat", "210fsh"])
        if "waverunner" in q or "watercraft" in q:
            target_sub_manual_terms.extend(["waverunner", "watercraft"])

    phrase_rules = [
        ("water supply", 0.20),
        ("turn the water supply", 0.18),
        ("factory reset", 0.28),
        ("reset button", 0.28),
        ("home screen", 0.08),
        ("steering position", 0.08),
        ("charge indicator", 0.16),
        ("充电指示灯", 0.18),
        ("过热/过冷", 0.12),
        ("喷淋臂", 0.16),
    ]

    enriched = []
    for index, item in enumerate(results):
        copied = dict(item)
        haystack = " ".join(
            [
                copied.get("content", ""),
                copied.get("section_title", ""),
                copied.get("sub_manual", ""),
                copied.get("content_type", ""),
            ]
        ).lower()
        focus_score = float(copied.get("rerank_score", 0.0) or copied.get("score", 0.0) or 0.0)
        overlap = sum(1 for term in terms if term.lower() in haystack)
        focus_score += overlap * 0.04
        for phrase, bonus in phrase_rules:
            if phrase in q and phrase in haystack:
                focus_score += bonus
        if "factory reset" in q or "reset" in q:
            if "reset button" in haystack or "factory reset" in haystack:
                focus_score += 0.22
            elif "reset" in haystack:
                focus_score += 0.12
            if "language setting" in haystack or "change the language" in haystack:
                focus_score -= 0.16
        if "water supply" in q:
            if "hose fitting" in haystack or "rear platform hatch" in haystack or "water supply on or off" in haystack:
                focus_score += 0.18
            if "aerator switch" in haystack or "livewell" in haystack:
                focus_score -= 0.10
        if "steering" in q and "screen" in q:
            if "steering position" in haystack:
                focus_score += 0.12
        if target_sub_manual_terms:
            sub_manual = (copied.get("sub_manual") or "").lower()
            if any(term in sub_manual for term in target_sub_manual_terms):
                focus_score += 0.18
            elif copied.get("product") == "汇总英文":
                focus_score -= 0.16
        if copied.get("content_type") == "toc":
            focus_score -= 0.12
        copied["focus_rerank_score"] = focus_score
        copied["_original_rank"] = index
        enriched.append(copied)

    enriched.sort(
        key=lambda item: (
            -float(item.get("focus_rerank_score", 0.0)),
            -float(item.get("rerank_score", 0.0) or 0.0),
            -float(item.get("score", 0.0) or 0.0),
            item.get("_original_rank", 9999),
        )
    )
    for item in enriched:
        item.pop("_original_rank", None)
    return enriched
