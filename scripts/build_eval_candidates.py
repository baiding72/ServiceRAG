import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from retriever import ManualRetriever


QUESTION_FILE = ROOT_DIR / "data/question_public.csv"
DEFAULT_OUTPUT = ROOT_DIR / "pooled_candidates.json"

DENSE_TOP_K = 30
BM25_TOP_K = 30
HYBRID_TOP_K = 30
REWRITE_TOP_K = 20
RRF_K = 60


def load_questions() -> List[dict]:
    with QUESTION_FILE.open("r", encoding="utf-8") as infile:
        return list(csv.DictReader(infile))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def clean_question_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = cleaned.strip('"')
    cleaned = re.sub(r'"\s*,\s*"', " ", cleaned)
    cleaned = cleaned.replace('\\"', '"')
    return normalize_whitespace(cleaned)


def is_mostly_english(text: str) -> bool:
    letters = re.findall(r"[A-Za-z]", text or "")
    chinese = re.findall(r"[\u4e00-\u9fff]", text or "")
    return bool(letters) and len(letters) >= max(12, len(chinese) * 2)


def classify_question(question: str) -> str:
    q = normalize_whitespace(question)
    q_lower = q.lower()

    if any(token in q for token in ["退货", "换货", "发票", "物流", "退款", "投诉", "补发", "售后", "运费"]):
        return "general_service"

    multi_markers = len(re.findall(r"[？?]", q)) >= 2 or len(re.findall(r'"\s*,\s*"', q)) >= 1
    if multi_markers:
        return "multi_subquestion_en" if is_mostly_english(q) else "multi_subquestion_zh"

    if any(token in q_lower for token in ["screen", "button", "menu", "setting", "settings", "display", "indicator", "switch"]) or any(token in q for token in ["界面", "按键", "按钮", "菜单", "设置", "显示屏", "指示灯", "开关"]):
        return "screen_button_en" if is_mostly_english(q) else "screen_button_zh"

    if any(token in q_lower for token in ["step", "steps", "install", "assembly", "replace", "remove", "clean", "charge", "connect", "start", "how to"]) or any(token in q for token in ["步骤", "安装", "组装", "更换", "拆卸", "清洁", "充电", "连接", "启动", "如何"]):
        return "steps_en" if is_mostly_english(q) else "steps_zh"

    return "general_english_tech" if is_mostly_english(q) else "general_chinese_tech"


def stratified_sample(rows: List[dict], sample_size: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        item = dict(row)
        item["question"] = clean_question_text(item["question"])
        item["category"] = classify_question(item["question"])
        buckets[item["category"]].append(item)

    for items in buckets.values():
        rng.shuffle(items)

    ordered_categories = sorted(
        buckets.keys(),
        key=lambda name: (
            0 if "english" in name else 1,
            0 if "steps" in name or "screen_button" in name else 1,
            name,
        )
    )

    sampled: List[dict] = []
    while len(sampled) < sample_size:
        progressed = False
        for category in ordered_categories:
            if buckets[category]:
                sampled.append(buckets[category].pop())
                progressed = True
                if len(sampled) >= sample_size:
                    break
        if not progressed:
            break

    sampled.sort(key=lambda row: int(row["id"]))
    return sampled


def expand_query_variants(question: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", question).strip()
    if not normalized:
        return []

    variants = [normalized]

    split_parts = re.split(r"[？?！!；;。\n]+", normalized)
    for part in split_parts:
        cleaned = part.strip(" ，,、:：")
        if len(cleaned) >= 4:
            variants.append(cleaned)

    numbered_parts = re.split(r"(?:^|[，,；;。.\s])(?:\d+[.、]|[一二三四五六七八九十]+[、.])", normalized)
    for part in numbered_parts:
        cleaned = part.strip(" ，,、:：")
        if len(cleaned) >= 4:
            variants.append(cleaned)

    deduped = []
    seen = set()
    for item in variants:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped


def rewrite_query(question: str) -> Optional[str]:
    q = normalize_whitespace(question)
    if not q:
        return None

    parts = expand_query_variants(q)
    if len(parts) > 1:
        # 多问句优先拼成更紧凑的“主题 + 子问”版本
        head = parts[0]
        tail = " ; ".join(parts[1:3])
        rewritten = normalize_whitespace(f"{head} {tail}")
        if rewritten and rewritten != q:
            return rewritten

    q_lower = q.lower()
    if is_mostly_english(q):
        q2 = re.sub(r"\b(can you|could you|please|i'd like to|i want to|i need to)\b", " ", q_lower, flags=re.I)
        q2 = re.sub(r"\bwhat are the steps to\b", "how to", q2, flags=re.I)
        q2 = re.sub(r"\bhow can you\b", "how to", q2, flags=re.I)
        q2 = normalize_whitespace(q2)
        return q2 if q2 and q2 != q_lower else None

    q2 = q
    q2 = re.sub(r"^(请问|请教一下|麻烦问下|我想了解一下|我想咨询一下)", "", q2)
    q2 = normalize_whitespace(q2)
    return q2 if q2 and q2 != q else None


def serialize_candidate(item: dict) -> dict:
    return {
        "chunk_id": item.get("chunk_id", ""),
        "text": item.get("content", ""),
        "parent_id": item.get("parent_id", ""),
        "product": item.get("product", ""),
        "sub_manual": item.get("sub_manual", ""),
        "section_title": item.get("section_title", ""),
        "language": item.get("language", ""),
        "content_type": item.get("content_type", ""),
        "images": item.get("images", []),
        "distance": item.get("distance", 999.0),
        "retrieval_score": item.get("retrieval_score", 0.0),
        "bm25_score": item.get("bm25_score", 0.0),
        "rerank_score": item.get("rerank_score"),
        "routes": item.get("routes", []),
        "route_ranks": item.get("route_ranks", {}),
    }


def merge_route_results(pool: Dict[str, dict], results: List[dict], route_name: str) -> None:
    for rank, item in enumerate(results, start=1):
        chunk_id = str(item.get("chunk_id", ""))
        if not chunk_id:
            continue

        existing = pool.get(chunk_id)
        if existing is None:
            existing = dict(item)
            existing["routes"] = []
            existing["route_ranks"] = {}
            pool[chunk_id] = existing

        if route_name not in existing["routes"]:
            existing["routes"].append(route_name)
        existing["route_ranks"][route_name] = min(rank, existing["route_ranks"].get(route_name, rank))

        existing["distance"] = min(item.get("distance", 999.0), existing.get("distance", 999.0))
        existing["bm25_score"] = max(item.get("bm25_score", 0.0), existing.get("bm25_score", 0.0))
        existing["retrieval_score"] = max(item.get("retrieval_score", 0.0), existing.get("retrieval_score", 0.0))
        rerank_score = item.get("rerank_score")
        if rerank_score is not None:
            current_rerank = existing.get("rerank_score")
            if current_rerank is None or rerank_score > current_rerank:
                existing["rerank_score"] = rerank_score


def dense_only(retriever: ManualRetriever, question: str, top_k: int) -> List[dict]:
    return retriever.search_semantic(question, top_k=top_k)


def bm25_only(retriever: ManualRetriever, question: str, top_k: int) -> List[dict]:
    return retriever.search_bm25(question, top_k=top_k)


def hybrid_rrf(retriever: ManualRetriever, question: str, top_k: int) -> List[dict]:
    merged: Dict[str, dict] = {}
    semantic_results = retriever.search_semantic(question, top_k=DENSE_TOP_K)
    bm25_results = retriever.search_bm25(question, top_k=BM25_TOP_K)

    for rank, item in enumerate(semantic_results, 1):
        chunk_id = str(item.get("chunk_id", ""))
        candidate = merged.setdefault(chunk_id, dict(item))
        candidate["distance"] = min(
            item.get("distance", 999.0),
            candidate.get("distance", 999.0),
        )
        candidate["retrieval_score"] = candidate.get("retrieval_score", 0.0) + 1.0 / (RRF_K + rank)

    for rank, item in enumerate(bm25_results, 1):
        chunk_id = str(item.get("chunk_id", ""))
        candidate = merged.setdefault(chunk_id, dict(item))
        candidate["bm25_score"] = max(
            item.get("bm25_score", 0.0),
            candidate.get("bm25_score", 0.0),
        )
        candidate["retrieval_score"] = candidate.get("retrieval_score", 0.0) + 1.0 / (RRF_K + rank)

    ranked = sorted(
        merged.values(),
        key=lambda x: (
            -x.get("retrieval_score", 0.0),
            x.get("distance", 999.0),
            -x.get("bm25_score", 0.0),
        ),
    )
    return ranked[:top_k]


def rewrite_hybrid(retriever: ManualRetriever, question: str, top_k: int) -> List[dict]:
    rewritten = rewrite_query(question)
    if not rewritten:
        return []
    return hybrid_rrf(retriever, rewritten, top_k=top_k)


def build_query_pool(retriever: ManualRetriever, question: str) -> Dict[str, dict]:
    pool: Dict[str, dict] = {}

    merge_route_results(pool, dense_only(retriever, question, DENSE_TOP_K), "dense")
    merge_route_results(pool, bm25_only(retriever, question, BM25_TOP_K), "bm25")
    merge_route_results(pool, hybrid_rrf(retriever, question, HYBRID_TOP_K), "hybrid")
    rewrite_results = rewrite_hybrid(retriever, question, REWRITE_TOP_K)
    if rewrite_results:
        merge_route_results(pool, rewrite_results, "rewrite_hybrid")

    return pool


def route_priority(candidate: dict) -> tuple:
    routes = candidate.get("routes", [])
    route_ranks = candidate.get("route_ranks", {})
    return (
        -len(routes),
        route_ranks.get("hybrid", 999),
        route_ranks.get("dense", 999),
        route_ranks.get("bm25", 999),
        route_ranks.get("rewrite_hybrid", 999),
        candidate.get("distance", 999.0),
        -candidate.get("bm25_score", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 pool-based gold eval 候选池")
    parser.add_argument("--sample-size", type=int, default=100, help="抽样题目数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 pooled_candidates.json")
    args = parser.parse_args()

    rows = load_questions()
    sampled = stratified_sample(rows, sample_size=args.sample_size, seed=args.seed)
    retriever = ManualRetriever(enable_rerank=True)

    payload = []
    print(f"[pool] selected={len(sampled)} output={args.output}", flush=True)
    print(
        f"[pool] dense@{DENSE_TOP_K} bm25@{BM25_TOP_K} hybrid@{HYBRID_TOP_K} rewrite@{REWRITE_TOP_K}",
        flush=True,
    )

    for idx, row in enumerate(sampled, 1):
        query = row["question"]
        pooled = build_query_pool(retriever, query)
        ranked_pool = sorted(pooled.values(), key=route_priority)

        payload.append(
            {
                "id": str(row["id"]),
                "query": query,
                "category": row["category"],
                "pool_size": len(ranked_pool),
                "candidates": [serialize_candidate(item) for item in ranked_pool],
            }
        )

        if idx % 5 == 0:
            print(f"  processed {idx}/{len(sampled)}", flush=True)

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(payload)} pooled queries to {args.output}", flush=True)


if __name__ == "__main__":
    main()
