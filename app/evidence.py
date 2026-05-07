"""Evidence formatting and image evidence controls."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

from app.query_analyzer import QueryAnalysis


@lru_cache(maxsize=4)
def known_image_ids(image_dir: str = "") -> set[str]:
    directory = Path(image_dir or os.getenv("IMAGE_ID_DIR", "data/手册/插图"))
    if not directory.exists():
        return set()
    return {
        file.stem
        for file in directory.rglob("*")
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }


def filter_known_image_ids(image_ids: Iterable[str], image_dir: str = "") -> List[str]:
    ids = list(dict.fromkeys(str(item).strip() for item in image_ids if str(item).strip()))
    known = known_image_ids(image_dir)
    if not known:
        return ids
    return [image_id for image_id in ids if image_id in known]


def extract_allowed_image_ids(retrieved_docs: Iterable[dict], visual_docs: Iterable[dict] | None = None) -> List[str]:
    images: List[str] = []
    for doc in retrieved_docs or []:
        for image_id in doc.get("image_ids", doc.get("images", [])) or []:
            if image_id:
                images.append(str(image_id))
    for doc in visual_docs or []:
        image_id = doc.get("image_id")
        if image_id:
            images.append(str(image_id))
        image_path = doc.get("image_path")
        if image_path:
            images.append(Path(str(image_path)).stem)
    return filter_known_image_ids(images)


@lru_cache(maxsize=1)
def _image_group_index() -> Dict[str, Dict[str, object]]:
    path = Path(os.getenv("STRUCTURED_KNOWLEDGE_PATH", "data/structured_knowledge.json"))
    if not path.exists():
        return {"chunk_to_group": {}, "parent_to_images": {}, "product_to_images": {}}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"chunk_to_group": {}, "parent_to_images": {}, "product_to_images": {}}

    chunk_to_group: Dict[str, List[str]] = {}
    parent_to_images: Dict[str, List[str]] = {}
    product_to_images: Dict[str, List[str]] = {}

    for item in data:
        images = [str(image) for image in item.get("images", []) if str(image)]
        product = item.get("product", "")
        if product:
            product_to_images.setdefault(product, [])
            for image in images:
                if image not in product_to_images[product]:
                    product_to_images[product].append(image)

        if item.get("level") == "parent" and images:
            parent_id = item.get("chunk_id", "")
            parent_to_images[parent_id] = images

    for item in data:
        chunk_id = item.get("chunk_id", "")
        if not chunk_id:
            continue
        images = [str(image) for image in item.get("images", []) if str(image)]
        parent_id = item.get("parent_id") or ""
        parent_images = parent_to_images.get(parent_id, [])
        if parent_images:
            chunk_to_group[chunk_id] = parent_images
        elif images:
            chunk_to_group[chunk_id] = images

    return {
        "chunk_to_group": chunk_to_group,
        "parent_to_images": parent_to_images,
        "product_to_images": product_to_images,
    }


def expand_group_images_for_docs(docs: List[dict], analysis: QueryAnalysis, max_images_per_doc: int = 4) -> List[dict]:
    """
    补全同一图组/相邻图片。

    说明书常用连续图片表达同一状态组，比如电钻充电中/充满/过热过冷。
    检索命中其中一张时，将 parent 内相邻图片补进 evidence，使 LLM 可引用完整图组。
    """
    if not docs or not analysis.is_image_related:
        return docs

    index = _image_group_index()
    chunk_to_group = index.get("chunk_to_group", {})
    product_to_images = index.get("product_to_images", {})
    expanded_docs = []
    for doc in docs:
        images = [str(image) for image in (doc.get("image_ids") or doc.get("images") or []) if str(image)]
        if not images:
            expanded_docs.append(doc)
            continue

        group_images = chunk_to_group.get(doc.get("chunk_id", ""))
        if not group_images:
            product_images = product_to_images.get(doc.get("product") or doc.get("manual") or "", [])
            group_images = _filtered_product_group(images, product_images)
        if not group_images:
            group_images = images
        expanded = _expand_images_by_local_group(images, group_images, analysis, max_images=max_images_per_doc)
        copied = dict(doc)
        copied["images"] = expanded
        copied["image_ids"] = expanded
        copied["image_group_expanded"] = expanded != images
        expanded_docs.append(copied)
    return expanded_docs


def _filtered_product_group(images: List[str], product_images: List[str]) -> List[str]:
    """
    Chroma 旧库与 structured JSON 的 chunk_id 不一致时，用产品图片序列做兜底。

    但只允许明显同前缀的专用图片组，例如 drill0_04 -> drill0_*。
    Manual11_0 这类 OCR/通用页图不参与产品级扩展，避免污染 allowed image pool。
    """
    if not images or not product_images:
        return []
    first = images[0]
    match = re.match(r"^([A-Za-z_]+)", first)
    if not match:
        return []
    prefix = match.group(1)
    if prefix.lower().startswith("manual"):
        return []
    return [image for image in product_images if image.startswith(prefix)]


def _expand_images_by_local_group(images: List[str], group_images: List[str], analysis: QueryAnalysis, max_images: int) -> List[str]:
    if not group_images:
        return images

    q = (analysis.original_query or analysis.rewritten_query or "").lower()
    desired = 3 if any(token in q for token in ["indicator", "light", "flash", "flashing", "指示灯", "闪烁", "充电"]) else max_images
    desired = min(max_images, max(desired, len(images)))

    selected: List[str] = []
    for image in images:
        if image in group_images:
            idx = group_images.index(image)
            window = _contiguous_image_window(group_images, idx, desired)
            for candidate in window:
                if candidate not in selected:
                    selected.append(candidate)
        elif image not in selected:
            selected.append(image)

    for image in images:
        if image not in selected:
            selected.append(image)
    return filter_known_image_ids(selected[:max_images])


def _contiguous_image_window(group_images: List[str], idx: int, desired: int) -> List[str]:
    target = group_images[idx]
    match = re.match(r"^(.*?)(\d+)$", target)
    if not match:
        start = max(0, idx - desired + 1)
        end = min(len(group_images), start + desired)
        return group_images[start:end]

    prefix = match.group(1)
    candidates = [i for i, image in enumerate(group_images) if image.startswith(prefix)]
    if idx not in candidates:
        candidates = list(range(max(0, idx - desired + 1), min(len(group_images), idx + 1)))

    pos = candidates.index(idx) if idx in candidates else len(candidates) - 1
    start_pos = max(0, pos - desired + 1)
    end_pos = min(len(candidates), start_pos + desired)
    if end_pos - start_pos < desired:
        start_pos = max(0, end_pos - desired)
    return [group_images[i] for i in candidates[start_pos:end_pos]]


def prioritize_image_evidence(docs: List[dict], analysis: QueryAnalysis) -> List[dict]:
    if not analysis.is_image_related:
        return docs
    return sorted(docs, key=lambda item: (0 if item.get("image_ids") or item.get("images") else 1, -float(item.get("score", 0.0) or 0.0)))


def build_manual_context(retrieved_docs: List[dict], analysis: QueryAnalysis | None = None) -> str:
    if not retrieved_docs:
        return "暂无相关参考知识。"
    docs = prioritize_image_evidence(retrieved_docs, analysis) if analysis else retrieved_docs
    parts = []
    for index, doc in enumerate(docs, 1):
        images = doc.get("image_ids", doc.get("images", [])) or []
        part = [
            f"【参考文档 {index}】",
            f"产品类别: {doc.get('product') or doc.get('manual') or '未知产品'}",
            f"章节: {doc.get('section_title', '')}",
            f"Chunk ID: {doc.get('chunk_id', '')}",
            f"相关度: {float(doc.get('score', doc.get('distance', 0.0)) or 0.0):.4f}",
            f"内容: {doc.get('content', '')}",
        ]
        if images:
            part.append(f"关联图片ID: {json.dumps(images, ensure_ascii=False)}")
            if doc.get("image_group_expanded"):
                part.append("图片说明: 这些图片属于同一连续状态/步骤图组；如回答涉及该状态组，请按顺序逐一插入<PIC>并返回相同顺序的图片ID。")
        parts.append("\n".join(part) + "\n" + "-" * 40)
    if analysis and analysis.is_image_related and not extract_allowed_image_ids(docs):
        parts.append("【图片证据提示】未检索到对应图片证据。")
    return "\n\n".join(parts)


def build_faq_context(faq_docs: List[dict]) -> str:
    if not faq_docs:
        return ""
    parts = []
    for index, doc in enumerate(faq_docs, 1):
        lines = [
            f"【FAQ参考 {index}】",
            f"主题: {doc.get('title', '')}",
            f"类别: {doc.get('category', '')}",
            f"回答要点: {doc.get('answer_guideline', '')}",
        ]
        service_tips = doc.get("service_tips", [])
        if service_tips:
            lines.append(f"处理提示: {'；'.join(service_tips)}")
        parts.append("\n".join(lines) + "\n" + "-" * 40)
    return "\n\n".join(parts)


def build_debug_evidence(retrieved_docs: List[dict], faq_docs: List[dict], analysis: QueryAnalysis) -> Dict[str, object]:
    return {
        "query_analysis": analysis.as_debug_dict(),
        "manual_chunks": [
            {
                "chunk_id": item.get("chunk_id", ""),
                "product": item.get("product", ""),
                "section_title": item.get("section_title", ""),
                "score": item.get("score", item.get("retrieval_score", 0.0)),
                "image_ids": item.get("image_ids", item.get("images", [])),
            }
            for item in retrieved_docs
        ],
        "faq_titles": [item.get("title", "") for item in faq_docs],
    }
