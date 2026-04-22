"""
FAQ 检索器

用途：
1. 加载结构化 FAQ 知识
2. 针对售后 / 物流 / 发票 / 安装等问题做轻量召回
3. 为主服务提供 FAQ 证据，而不是完全依赖 LLM 临场生成
"""

import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List


FAQ_DATA_PATH = os.getenv("FAQ_DATA_PATH", "./data/service_faq.json")
BM25_K1 = float(os.getenv("FAQ_BM25_K1", "1.5"))
BM25_B = float(os.getenv("FAQ_BM25_B", "0.75"))
ENGLISH_STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "be", "to", "of", "on", "in", "at", "for",
    "and", "or", "if", "my", "your", "their", "what", "how", "do", "does", "did",
    "can", "could", "should", "would", "i", "you", "we", "they", "it", "this", "that",
    "these", "those", "about", "with", "from", "after", "before"
}


class FAQRetriever:
    def __init__(self, data_path: str = FAQ_DATA_PATH):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"FAQ data not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        self._items = [self._prepare_item(item) for item in raw_items]
        self._build_bm25_index()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower()).strip()

    def _tokenize(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        tokens: List[str] = []

        for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", normalized):
            if token not in ENGLISH_STOPWORDS:
                tokens.append(token)

        for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
            tokens.append(phrase)
            if len(phrase) >= 2:
                for idx in range(0, len(phrase) - 1):
                    tokens.append(phrase[idx: idx + 2])
            if len(phrase) >= 3:
                for idx in range(0, len(phrase) - 2):
                    tokens.append(phrase[idx: idx + 3])

        return tokens

    def _prepare_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        question_patterns = item.get("question_patterns", [])
        service_tips = item.get("service_tips", [])
        keywords = item.get("keywords", [])
        retrieval_text = " ".join(
            [
                item.get("title", ""),
                " ".join(keywords),
                " ".join(question_patterns),
                item.get("answer_guideline", ""),
                " ".join(service_tips),
            ]
        ).strip()
        return {
            **item,
            "retrieval_text": retrieval_text,
            "normalized_text": self._normalize_text(retrieval_text),
            "tokens": self._tokenize(retrieval_text),
        }

    def _build_bm25_index(self) -> None:
        self._doc_freqs: List[Counter] = []
        self._lengths: List[int] = []
        self._df = Counter()

        for item in self._items:
            term_freq = Counter(item.get("tokens", []))
            self._doc_freqs.append(term_freq)
            self._lengths.append(sum(term_freq.values()))
            for token in term_freq.keys():
                self._df[token] += 1

        self._doc_count = max(1, len(self._items))
        total_length = sum(self._lengths)
        self._avgdl = total_length / self._doc_count if total_length else 1.0

    def _bm25_idf(self, term: str) -> float:
        doc_freq = self._df.get(term, 0)
        return math.log(1 + (self._doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_norm = self._normalize_text(query)
        results: List[Dict[str, Any]] = []

        for idx, item in enumerate(self._items):
            score = 0.0
            doc_len = self._lengths[idx] or 1
            term_freq = self._doc_freqs[idx]

            for token in query_tokens:
                freq = term_freq.get(token, 0)
                if not freq:
                    continue
                idf = self._bm25_idf(token)
                denom = freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / self._avgdl)
                score += idf * (freq * (BM25_K1 + 1)) / denom

            exact_pattern_bonus = 0.0
            for pattern in item.get("question_patterns", []):
                if pattern and self._normalize_text(pattern) in query_norm:
                    exact_pattern_bonus += 1.0

            keyword_hits = 0
            for keyword in item.get("keywords", []):
                keyword_norm = self._normalize_text(keyword)
                if keyword_norm and keyword_norm in query_norm:
                    keyword_hits += 1

            total_score = score + exact_pattern_bonus + min(1.5, keyword_hits * 0.3)
            if total_score <= 0:
                continue

            results.append(
                {
                    "faq_id": item.get("id", ""),
                    "category": item.get("category", ""),
                    "title": item.get("title", ""),
                    "keywords": item.get("keywords", []),
                    "question_patterns": item.get("question_patterns", []),
                    "answer_guideline": item.get("answer_guideline", ""),
                    "service_tips": item.get("service_tips", []),
                    "score": float(total_score),
                }
            )

        results.sort(key=lambda x: -x.get("score", 0.0))
        return results[:top_k]
