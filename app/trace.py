"""Query trace writer for retrieval/debug analysis.

Trace generation is deliberately best-effort: failures are swallowed so the
competition API path is never affected by diagnostics.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from app.config import Settings, settings


def write_query_trace(
    *,
    request_id: str,
    session_id: str,
    question: str,
    effective_question: str,
    intent_type: str,
    query_analysis: Any,
    manual_trace: Mapping[str, Any] | None = None,
    manual_docs: list[dict] | None = None,
    faq_docs: list[dict] | None = None,
    visual_docs: list[dict] | None = None,
    answer: str = "",
    config: Settings = settings,
) -> str | None:
    if not config.query_trace_enabled:
        return None

    try:
        trace_dir = Path(config.query_trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        safe_question = _slug(question)[:48] or "query"
        trace_path = trace_dir / f"{timestamp}_{safe_question}_{request_id[:8]}.json"
        payload = {
            "request_id": request_id,
            "session_id": session_id,
            "timestamp": timestamp,
            "question": question,
            "effective_question": effective_question,
            "intent_type": intent_type,
            "analysis": _analysis_dict(query_analysis),
            "manual_retrieval": manual_trace or {},
            "manual_docs": [_compact_doc(item, rank) for rank, item in enumerate(manual_docs or [], 1)],
            "faq_docs": [_compact_doc(item, rank) for rank, item in enumerate(faq_docs or [], 1)],
            "visual_docs": [_compact_doc(item, rank) for rank, item in enumerate(visual_docs or [], 1)],
            "answer_preview": answer[:1000],
            "answer_pic_count": answer.count("<PIC>"),
            "answer_images": _extract_answer_images(answer),
        }
        trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(trace_path)
    except Exception as exc:
        print(f"query trace 写入失败: {exc}")
        return None


def new_request_id() -> str:
    return uuid.uuid4().hex


def _analysis_dict(query_analysis: Any) -> dict:
    if hasattr(query_analysis, "as_debug_dict"):
        return query_analysis.as_debug_dict()
    if isinstance(query_analysis, Mapping):
        return dict(query_analysis)
    return {}


def _compact_doc(item: Mapping[str, Any], rank: int) -> dict:
    content = str(
        item.get("content")
        or item.get("text")
        or item.get("answer_guideline")
        or item.get("title")
        or ""
    )
    images = item.get("image_ids") or item.get("images") or []
    return {
        "rank": rank,
        "chunk_id": item.get("chunk_id") or item.get("id") or "",
        "parent_id": item.get("parent_id") or (item.get("metadata") or {}).get("parent_id", ""),
        "product": item.get("product") or item.get("manual") or "",
        "sub_manual": item.get("sub_manual") or (item.get("metadata") or {}).get("sub_manual", ""),
        "section_title": item.get("section_title") or item.get("section") or "",
        "content_type": item.get("content_type") or (item.get("metadata") or {}).get("content_type", ""),
        "score": item.get("score", item.get("retrieval_score", 0.0)),
        "distance": item.get("distance"),
        "dense_score": item.get("dense_score"),
        "bm25_score": item.get("bm25_score"),
        "rerank_score": item.get("rerank_score"),
        "route": item.get("route", ""),
        "route_ranks": item.get("route_ranks", {}),
        "image_ids": images if isinstance(images, list) else [],
        "image_group_expanded": bool(item.get("image_group_expanded")),
        "content_preview": content[:500],
    }


def _extract_answer_images(answer: str) -> list[str]:
    match = re.search(r",\s*(\[[^\]]*\])\s*$", answer or "")
    if not match:
        return []
    try:
        parsed = json.loads(match.group(1))
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        return []
    return []


def _slug(text: str) -> str:
    cleaned = re.sub(r"\s+", "_", (text or "").strip())
    cleaned = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "", cleaned)
    return cleaned or "query"
