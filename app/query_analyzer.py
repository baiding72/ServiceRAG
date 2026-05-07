"""Rule-based query understanding for ServiceRAG.

The analyzer is intentionally lightweight and deterministic so tests and eval
can run without an LLM key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


INTENTS = {
    "usage_guide",
    "troubleshooting",
    "warranty_return",
    "image_related",
    "comparison",
    "unknown",
}

SERVICE_TERMS_ZH = {
    "退货", "换货", "退款", "发票", "物流", "快递", "运费", "投诉", "售后",
    "补发", "少发", "错发", "取消订单", "维修费用", "寄修", "发货", "到货",
    "签收", "保修", "质保", "维修",
}
SERVICE_TERMS_EN = {
    "return", "refund", "exchange", "invoice", "shipping", "delivery",
    "courier", "complaint", "warranty", "repair fee", "replacement",
}
TROUBLE_TERMS_ZH = {"故障", "报错", "闪烁", "不亮", "无法", "不能", "异常", "原因", "怎么办"}
TROUBLE_TERMS_EN = {"error", "fault", "blink", "flashing", "cannot", "can't", "fail", "failed", "trouble"}
IMAGE_TERMS_ZH = {"图片", "图示", "如图", "照片", "指示灯", "图标", "按钮", "界面", "屏幕", "尺寸", "位置"}
IMAGE_TERMS_EN = {"image", "picture", "diagram", "indicator", "icon", "button", "screen", "display", "size", "location"}
USAGE_TERMS_ZH = {"如何", "怎么", "步骤", "安装", "组装", "设置", "打开", "关闭", "使用", "操作"}
USAGE_TERMS_EN = {"how", "step", "install", "assembly", "setup", "turn on", "turn off", "operate", "use"}
COMPARISON_TERMS_ZH = {"区别", "对比", "哪个", "更适合", "相比"}
COMPARISON_TERMS_EN = {"compare", "difference", "which", "versus", "vs"}

PRODUCT_HINTS = [
    "DCB107", "DCB112", "电钻", "健身追踪器", "船", "boat", "water supply",
    "livewell", "battery", "drill", "charger", "Manual16",
]


@dataclass
class QueryAnalysis:
    original_query: str
    rewritten_query: str
    intent: str
    entities: Dict[str, List[str]] = field(default_factory=dict)
    is_image_related: bool = False
    language: str = "zh"
    sub_questions: List[str] = field(default_factory=list)

    def as_debug_dict(self) -> Dict[str, object]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "intent": self.intent,
            "entities": self.entities,
            "is_image_related": self.is_image_related,
            "language": self.language,
            "sub_questions": self.sub_questions,
        }


def detect_language(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "zh"
    cjk_chars = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    return "en" if ascii_letters > 0 and cjk_chars == 0 else "zh"


def split_sub_questions(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return []
    parts = [normalized]
    for part in re.split(r"[？?！!；;。\n]+", normalized):
        cleaned = part.strip(" ，,、:：")
        if len(cleaned) >= 4:
            parts.append(cleaned)
    seen = set()
    deduped = []
    for item in parts:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _contains_any(text: str, raw_text: str, zh_terms: set[str], en_terms: set[str]) -> bool:
    return any(term in raw_text for term in zh_terms) or any(term in text for term in en_terms)


def extract_entities(query: str) -> Dict[str, List[str]]:
    normalized = (query or "").lower()
    entities: Dict[str, List[str]] = {
        "products": [],
        "models": [],
        "error_codes": [],
        "indicator_terms": [],
        "ui_terms": [],
        "operation_terms": [],
        "image_terms": [],
    }

    for hint in PRODUCT_HINTS:
        if hint.lower() in normalized or hint in query:
            entities["products"].append(hint)

    for model in re.findall(r"(?<![A-Za-z0-9])[A-Z]{2,}[A-Z0-9-]{2,}(?![A-Za-z0-9])", query):
        entities["models"].append(model)

    for code in re.findall(r"\b(?:E|ERR|ERROR)[-_]?\d{1,4}\b", query, flags=re.IGNORECASE):
        entities["error_codes"].append(code)

    for term in ["指示灯", "闪烁", "indicator", "flashing", "light"]:
        if term.lower() in normalized or term in query:
            entities["indicator_terms"].append(term)

    for term in ["按钮", "菜单", "界面", "屏幕", "button", "menu", "screen", "display", "setting"]:
        if term.lower() in normalized or term in query:
            entities["ui_terms"].append(term)

    for term in ["安装", "组装", "设置", "打开", "关闭", "install", "assembly", "setup", "turn on", "turn off"]:
        if term.lower() in normalized or term in query:
            entities["operation_terms"].append(term)

    for term in sorted(IMAGE_TERMS_ZH | IMAGE_TERMS_EN):
        if term.lower() in normalized or term in query:
            entities["image_terms"].append(term)

    return {key: list(dict.fromkeys(values)) for key, values in entities.items() if values}


def analyze_query(query: str) -> QueryAnalysis:
    original = query or ""
    normalized = re.sub(r"\s+", " ", original.strip().lower())
    language = detect_language(original)
    entities = extract_entities(original)
    is_image_related = bool(entities.get("image_terms")) or _contains_any(normalized, original, IMAGE_TERMS_ZH, IMAGE_TERMS_EN)

    if _contains_any(normalized, original, SERVICE_TERMS_ZH, SERVICE_TERMS_EN):
        intent = "warranty_return"
    elif is_image_related:
        intent = "image_related"
    elif _contains_any(normalized, original, COMPARISON_TERMS_ZH, COMPARISON_TERMS_EN):
        intent = "comparison"
    elif _contains_any(normalized, original, TROUBLE_TERMS_ZH, TROUBLE_TERMS_EN):
        intent = "troubleshooting"
    elif _contains_any(normalized, original, USAGE_TERMS_ZH, USAGE_TERMS_EN):
        intent = "usage_guide"
    else:
        intent = "unknown"

    prefixes = []
    if entities.get("products"):
        prefixes.append(" ".join(entities["products"]))
    if entities.get("models"):
        prefixes.append(" ".join(entities["models"]))
    rewritten = " ".join(prefixes + [original]).strip() if prefixes else original

    return QueryAnalysis(
        original_query=original,
        rewritten_query=rewritten,
        intent=intent if intent in INTENTS else "unknown",
        entities=entities,
        is_image_related=is_image_related,
        language=language,
        sub_questions=split_sub_questions(original),
    )


def route_intent(analysis: QueryAnalysis) -> str:
    if analysis.intent == "warranty_return":
        return "service_faq"
    if analysis.intent in {"usage_guide", "troubleshooting", "image_related", "comparison"}:
        return "manual_technical"
    return "manual_technical"
