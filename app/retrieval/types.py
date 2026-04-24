"""Shared retrieval result schema."""

from __future__ import annotations

from typing import Any, Dict, List

RetrievalResult = Dict[str, Any]


def _images(raw: Dict[str, Any]) -> List[str]:
    value = raw.get("image_ids", raw.get("images", []))
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def normalize_result(raw: Dict[str, Any], route: str = "") -> RetrievalResult:
    metadata = raw.get("metadata") or {}
    chunk_id = raw.get("chunk_id") or raw.get("id") or metadata.get("chunk_id") or ""
    product = raw.get("product") or raw.get("manual") or metadata.get("product") or "unknown"
    section_title = raw.get("section_title") or metadata.get("section_title") or raw.get("section") or ""
    return {
        "chunk_id": chunk_id,
        "content": raw.get("content") or raw.get("text") or raw.get("document") or "",
        "manual": product,
        "product": product,
        "section_title": section_title,
        "source_path": raw.get("source_path") or metadata.get("source_path") or "",
        "score": float(raw.get("score", raw.get("retrieval_score", 0.0)) or 0.0),
        "dense_score": float(raw.get("dense_score", raw.get("semantic_score", 0.0)) or 0.0),
        "bm25_score": float(raw.get("bm25_score", 0.0) or 0.0),
        "rerank_score": float(raw.get("rerank_score", 0.0) or 0.0),
        "distance": float(raw.get("distance", 999.0) or 999.0),
        "image_ids": _images(raw),
        "images": _images(raw),
        "matched_terms": raw.get("matched_terms", []),
        "parent_id": raw.get("parent_id") or metadata.get("parent_id") or "",
        "metadata": metadata or {k: v for k, v in raw.items() if k not in {"content", "text"}},
        "route": route or raw.get("route", ""),
    }

