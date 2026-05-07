"""Hallucination and confidence guards."""

from __future__ import annotations

from typing import Iterable, List

from app.query_analyzer import QueryAnalysis


def top_confidence(retrieved_docs: Iterable[dict]) -> float:
    docs = list(retrieved_docs or [])
    if not docs:
        return 0.0
    scores = [float(doc.get("score", doc.get("retrieval_score", 0.0)) or 0.0) for doc in docs]
    return max(scores) if scores else 0.0


def should_fallback(retrieved_docs: List[dict], low_confidence_threshold: float = 0.0) -> bool:
    if not retrieved_docs:
        return True
    if low_confidence_threshold <= 0:
        return False
    return top_confidence(retrieved_docs) < low_confidence_threshold


def has_product_conflict(retrieved_docs: List[dict], analysis: QueryAnalysis, min_docs: int = 3) -> bool:
    if analysis.entities.get("products") or len(retrieved_docs) < min_docs:
        return False
    products = [doc.get("product") or doc.get("manual") for doc in retrieved_docs[:min_docs]]
    unique = {product for product in products if product and product != "unknown"}
    return len(unique) >= min_docs


def fallback_message(language: str) -> str:
    if language == "en":
        return "Sorry, I could not find sufficiently relevant product information for your question. I am transferring you to a human agent."
    return "您好，暂未查询到相关产品信息，已为您转接人工客服。"


def clarification_message(language: str) -> str:
    if language == "en":
        return "To avoid giving inaccurate guidance, please provide the exact product model or product name so I can check the correct manual."
    return "为避免给出不准确的操作建议，请您补充具体产品型号或产品名称，我会再按对应说明书为您查询。"

