"""Small reproducible eval for retrieval and answer-surface checks.

This eval is intentionally lightweight. It does not claim to reproduce the
official benchmark; it provides a local regression signal for retrieval and
format quality.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings
from app.guard import should_fallback
from app.query_analyzer import analyze_query
from app.retrieval import HybridRetriever
from retriever import ManualRetriever


DATASET_PATH = Path("eval/eval_dataset.jsonl")
REPORT_DIR = Path("eval_reports")


def load_dataset(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _match_expected(item: Dict[str, Any], result: Dict[str, Any]) -> bool:
    expected_chunk = item.get("expected_chunk_id")
    if expected_chunk and result.get("chunk_id") == expected_chunk:
        return True
    expected_section = (item.get("expected_section") or "").lower()
    if expected_section and expected_section in (result.get("section_title") or "").lower():
        return True
    expected_product = (item.get("expected_product") or "").lower()
    haystack = " ".join([
        result.get("product", ""),
        result.get("manual", ""),
        result.get("content", "")[:300],
    ]).lower()
    return bool(expected_product and expected_product in haystack)


def _image_recall(expected_images: List[str], results: List[Dict[str, Any]]) -> float:
    if not expected_images:
        return 1.0
    found = set()
    for result in results:
        found.update(result.get("image_ids", result.get("images", [])) or [])
    expected = set(expected_images)
    return len(expected & found) / len(expected)


def _keyword_hit_rate(keywords: List[str], text: str) -> float:
    if not keywords:
        return 1.0
    lowered = (text or "").lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return hits / len(keywords)


def run_eval(limit: int | None = None) -> Dict[str, Any]:
    dataset = load_dataset(DATASET_PATH, limit=limit)
    retriever = ManualRetriever(
        persist_dir=settings.chroma_db_path,
        collection_name=settings.chroma_collection,
        model_name=settings.embedding_model_name,
        enable_rerank=settings.reranker_enabled,
        reranker_model_name=settings.reranker_model_name,
    )
    hybrid = HybridRetriever(retriever, settings)

    recall_hits = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []
    image_recalls = []
    keyword_rates = []
    fallback_count = 0
    latencies = []
    details = []

    for item in dataset:
        started = time.time()
        analysis = analyze_query(item["question"])
        results = hybrid.search(analysis, top_k=5)
        latency = time.time() - started
        latencies.append(latency)

        first_match_rank = None
        for rank, result in enumerate(results, 1):
            if _match_expected(item, result):
                first_match_rank = rank
                break

        for k in recall_hits:
            if first_match_rank is not None and first_match_rank <= k:
                recall_hits[k] += 1
        reciprocal_ranks.append(1 / first_match_rank if first_match_rank else 0.0)

        image_recalls.append(_image_recall(item.get("expected_image_ids", []), results))
        mock_answer_surface = " ".join(result.get("content", "") for result in results[:2])
        keyword_rates.append(_keyword_hit_rate(item.get("answer_keywords", []), mock_answer_surface))
        if should_fallback(results, settings.low_confidence_threshold):
            fallback_count += 1

        details.append({
            "id": item.get("id"),
            "question": item.get("question"),
            "intent": analysis.intent,
            "language": analysis.language,
            "first_match_rank": first_match_rank,
            "top_chunks": [result.get("chunk_id") for result in results],
            "latency_ms": round(latency * 1000, 2),
        })

    n = len(dataset) or 1
    metrics = {
        "count": len(dataset),
        "recall@1": recall_hits[1] / n,
        "recall@3": recall_hits[3] / n,
        "recall@5": recall_hits[5] / n,
        "mrr": sum(reciprocal_ranks) / n,
        "image_id_recall": sum(image_recalls) / n,
        "answer_keyword_hit_rate": sum(keyword_rates) / n,
        "fallback_rate": fallback_count / n,
        "average_latency_ms": round((sum(latencies) / n) * 1000, 2),
    }
    return {"metrics": metrics, "details": details}


def write_reports(report: Dict[str, Any]) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    json_path = REPORT_DIR / "latest.json"
    md_path = REPORT_DIR / "latest.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = report["metrics"]
    lines = ["# ServiceRAG Local Eval", "", "| Metric | Value |", "| --- | --- |"]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("This report is generated from the local sample eval set and is not an official benchmark score.")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--with-llm", action="store_true", help="Reserved for future full answer evaluation.")
    args = parser.parse_args()
    report = run_eval(limit=args.limit)
    write_reports(report)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

