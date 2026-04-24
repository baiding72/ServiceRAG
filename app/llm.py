"""OpenAI-compatible LLM wrapper with deterministic mock fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from app.config import Settings


@dataclass
class LLMResult:
    text: str
    images: List[str]
    raw: str = ""


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mock_mode = settings.llm_mock_mode or not bool(settings.llm_api_key)
        self.client: Optional[OpenAI] = None
        if not self.mock_mode:
            self.client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=settings.llm_timeout,
            )

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> LLMResult:
        if self.mock_mode or self.client is None:
            return LLMResult(text=fallback_text, images=[], raw="mock")

        response = self.client.chat.completions.create(
            model=model or self.settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=max_tokens,
        )
        raw = (response.choices[0].message.content or "").strip()
        text, images = parse_json_answer(raw)
        return LLMResult(text=text, images=images, raw=raw)


def parse_json_answer(raw_response: str) -> tuple[str, List[str]]:
    cleaned = (raw_response or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            text = str(result.get("text", "")).strip()
            images = result.get("images", [])
            if isinstance(images, list):
                return text, [str(image) for image in images if str(image).strip()]
            return text, []
    except json.JSONDecodeError:
        pass

    text_match = re.search(r'"text"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', raw_response, re.DOTALL)
    text = text_match.group(1).replace('\\"', '"').replace("\\n", "\n") if text_match else cleaned[:500]
    images_match = re.search(r'"images"\s*:\s*\[([^\]]*)\]', raw_response)
    images = re.findall(r'"([^"]+)"', images_match.group(1)) if images_match else []
    return text.strip(), images

