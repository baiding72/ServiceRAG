"""In-memory session store with TTL."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import settings
from app.query_analyzer import QueryAnalysis


FOLLOWUP_TERMS = {"继续", "还有吗", "这个怎么处理", "怎么处理", "然后呢", "还有呢", "continue", "what about this", "and then"}


@dataclass
class SessionTurn:
    question: str
    answer: str
    product: str = ""
    intent: str = "unknown"
    entities: Dict[str, List[str]] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class SessionMemoryStore:
    def __init__(self, max_turns: int = settings.session_max_turns, ttl_seconds: int = settings.session_ttl_seconds):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, List[SessionTurn]] = {}

    def _cleanup(self) -> None:
        now = time.time()
        stale = [
            session_id for session_id, turns in self._store.items()
            if not turns or now - turns[-1].timestamp > self.ttl_seconds
        ]
        for session_id in stale:
            self._store.pop(session_id, None)

    def add_turn(self, session_id: str, question: str, answer: str, analysis: QueryAnalysis) -> None:
        self._cleanup()
        products = analysis.entities.get("products", []) or analysis.entities.get("models", [])
        turn = SessionTurn(
            question=question,
            answer=answer[:240],
            product=products[0] if products else "",
            intent=analysis.intent,
            entities=analysis.entities,
        )
        turns = self._store.setdefault(session_id, [])
        turns.append(turn)
        self._store[session_id] = turns[-self.max_turns:]

    def get_last_turn(self, session_id: str) -> Optional[SessionTurn]:
        self._cleanup()
        turns = self._store.get(session_id, [])
        return turns[-1] if turns else None

    def enrich_question(self, session_id: str, question: str) -> str:
        normalized = (question or "").strip().lower()
        if not normalized or not any(term in normalized for term in FOLLOWUP_TERMS):
            return question
        last = self.get_last_turn(session_id)
        if not last:
            return question
        context_bits = [last.product, last.intent, last.question]
        context = " ".join(bit for bit in context_bits if bit)
        return f"{context}；追问：{question}" if context else question

    def clear(self) -> None:
        self._store.clear()


session_memory = SessionMemoryStore()

