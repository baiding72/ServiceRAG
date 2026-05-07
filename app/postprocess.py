"""Answer post-processing utilities."""

from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional, Tuple


def dedupe_sentences(text: str) -> str:
    parts = re.split(r"(?<=[。！？!?])|(?<=[.?!])", (text or "").strip())
    seen = set()
    kept = []
    for part in parts:
        sentence = re.sub(r"\s+", " ", part).strip()
        if not sentence:
            continue
        key = sentence.lower()
        if key not in seen:
            seen.add(key)
            kept.append(sentence)
    return " ".join(kept) if kept else (text or "").strip()


def remove_stiff_phrases(text: str) -> str:
    cleaned = text or ""
    for phrase in ["根据上下文，", "根据参考知识，", "根据提供的信息，", "Based on the context, ", "According to the context, "]:
        cleaned = cleaned.replace(phrase, "")
    return cleaned.strip()


def normalize_image_alignment(
    text: str,
    images: Iterable[str],
    allowed_images: Optional[Iterable[str]] = None,
) -> Tuple[str, List[str]]:
    sanitized_text = (text or "").strip()
    sanitized_images = list(dict.fromkeys(str(image).strip() for image in images or [] if str(image).strip()))
    if allowed_images is not None:
        allowed_set = set(str(item) for item in allowed_images)
        sanitized_images = [image for image in sanitized_images if image in allowed_set]

    placeholder_count = sanitized_text.count("<PIC>")
    if not sanitized_images:
        return re.sub(r"\s+", " ", sanitized_text.replace("<PIC>", "")).strip(), []

    if placeholder_count == 0:
        return sanitized_text, []
    if placeholder_count > len(sanitized_images):
        for _ in range(placeholder_count - len(sanitized_images)):
            sanitized_text = sanitized_text[::-1].replace(">CIP<", "", 1)[::-1]
    elif placeholder_count < len(sanitized_images):
        sanitized_images = sanitized_images[:placeholder_count]

    if not sanitized_images:
        sanitized_text = sanitized_text.replace("<PIC>", "")
    return re.sub(r"\s+", " ", sanitized_text).strip(), sanitized_images


def limit_answer_length(text: str, max_chars: int = 650) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    suffix = "." if re.search(r"[A-Za-z]", cleaned[:80]) else "。"
    return cleaned[:max_chars].rstrip(" ，,;；") + suffix


def format_competition_answer(text: str, images: Iterable[str]) -> str:
    cleaned = (text or "").strip()
    image_list = list(images or [])
    if image_list:
        return f"{cleaned} , {json.dumps(image_list, ensure_ascii=False)}"
    return cleaned


def postprocess_answer(
    text: str,
    images: Iterable[str],
    allowed_images: Optional[Iterable[str]] = None,
    max_chars: int = 650,
) -> Tuple[str, List[str]]:
    cleaned = limit_answer_length(dedupe_sentences(remove_stiff_phrases(text)), max_chars=max_chars)
    return normalize_image_alignment(cleaned, images, allowed_images=allowed_images)

