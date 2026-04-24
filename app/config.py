"""Centralized runtime configuration for ServiceRAG."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_token: str = os.getenv("KAFU_API_TOKEN") or os.getenv("AUTH_TOKEN", "kafu_test_token_2024")

    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "qwen-plus")
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "30"))
    llm_mock_mode: bool = _bool_env("LLM_MOCK_MODE", False)

    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db_m3")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "manuals_qa_m3")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    semantic_candidate_k: int = int(os.getenv("SEMANTIC_CANDIDATE_K", "18"))
    bm25_candidate_k: int = int(os.getenv("BM25_CANDIDATE_K", "12"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    hybrid_retrieval_enabled: bool = _bool_env("HYBRID_RETRIEVAL_ENABLED", True)
    reranker_enabled: bool = _bool_env("RERANKER_ENABLED", _bool_env("ENABLE_RERANK", True))
    reranker_model_name: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
    low_confidence_threshold: float = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.0"))

    skip_image_retriever: bool = _bool_env("SKIP_IMAGE_RETRIEVER", False)
    enable_vl_rerank: bool = _bool_env("ENABLE_VL_RERANK", True)
    vl_rerank_model_name: str = os.getenv("VL_RERANK_MODEL_NAME", "qwen3-vl-rerank")
    vl_rerank_top_k: int = int(os.getenv("VL_RERANK_TOP_K", "2"))
    vl_rerank_timeout: int = int(os.getenv("VL_RERANK_TIMEOUT", "30"))
    vl_rerank_endpoint: str = os.getenv(
        "VL_RERANK_ENDPOINT",
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    )

    session_max_turns: int = int(os.getenv("SESSION_MAX_TURNS", "6"))
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
    debug_response: bool = _bool_env("DEBUG_RESPONSE", False)


settings = Settings()

