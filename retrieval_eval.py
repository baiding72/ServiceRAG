import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from retriever import ManualRetriever


QUESTION_FILE = Path("data/question_public.csv")
EVAL_SET_FILE = Path("eval/retrieval_eval_set.json")
SEMANTIC_CANDIDATE_K = 18
BM25_CANDIDATE_K = 12
RRF_K = 60


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def load_questions() -> Dict[int, str]:
    rows = csv.DictReader(QUESTION_FILE.open("r", encoding="utf-8"))
    return {int(row["id"]): row["question"].strip().strip('"') for row in rows}


def load_eval_set() -> List[dict]:
    return json.loads(EVAL_SET_FILE.read_text(encoding="utf-8"))


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


def semantic_only(retriever: ManualRetriever, question: str, top_k: int) -> List[dict]:
    seen = {}
    for query in expand_query_variants(question):
        for item in retriever.search_semantic(query, top_k=max(top_k, SEMANTIC_CANDIDATE_K)):
            key = (item.get("chunk_id", ""), item.get("product", ""))
            distance = item.get("distance", 999.0)
            if key not in seen or distance < seen[key].get("distance", 999.0):
                seen[key] = item
    return sorted(seen.values(), key=lambda x: x.get("distance", 999.0))[:top_k]


def semantic_candidates(retriever: ManualRetriever, question: str, top_k: int) -> List[dict]:
    seen = {}

    for query in expand_query_variants(question):
        for item in retriever.search_semantic(query, top_k=SEMANTIC_CANDIDATE_K):
            key = (item.get("chunk_id", ""), item.get("product", ""))
            distance = item.get("distance", 999.0)
            if key not in seen or distance < seen[key].get("distance", 999.0):
                seen[key] = item

    merged = list(seen.values())
    merged.sort(key=lambda x: x.get("distance", 999.0))
    return merged[: max(top_k, SEMANTIC_CANDIDATE_K)]


def dense_plus_bm25_candidates(retriever: ManualRetriever, question: str, top_k: int) -> List[dict]:
    merged = {}

    for query in expand_query_variants(question):
        semantic_results = retriever.search_semantic(query, top_k=SEMANTIC_CANDIDATE_K)
        bm25_results = retriever.search_bm25(query, top_k=BM25_CANDIDATE_K)

        for rank, item in enumerate(semantic_results, 1):
            key = (item.get("chunk_id", ""), item.get("product", ""))
            candidate = merged.setdefault(key, dict(item))
            candidate["distance"] = min(
                item.get("distance", 999.0),
                candidate.get("distance", 999.0),
            )
            candidate["retrieval_score"] = candidate.get("retrieval_score", 0.0) + 1.0 / (RRF_K + rank)

        for rank, item in enumerate(bm25_results, 1):
            key = (item.get("chunk_id", ""), item.get("product", ""))
            candidate = merged.setdefault(key, dict(item))
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
        )
    )
    return ranked[: max(top_k, SEMANTIC_CANDIDATE_K, BM25_CANDIDATE_K)]


def semantic_expanded(retriever: ManualRetriever, question: str, top_k: int) -> List[dict]:
    return semantic_candidates(retriever, question, top_k)


def semantic_with_rerank(retriever: ManualRetriever, question: str, top_k: int) -> List[dict]:
    candidates = semantic_candidates(retriever, question, top_k=max(top_k, 24))
    return retriever.rerank_results(question, candidates, top_k=top_k)


def dense_plus_bm25(retriever: ManualRetriever, question: str, top_k: int) -> List[dict]:
    return dense_plus_bm25_candidates(retriever, question, top_k=top_k)


def dense_plus_bm25_with_rerank(retriever: ManualRetriever, question: str, top_k: int) -> List[dict]:
    candidates = dense_plus_bm25_candidates(retriever, question, top_k=max(top_k, 24))
    return retriever.rerank_results(question, candidates, top_k=top_k)


def doc_matches(doc: dict, expected_product: str, phrases: List[str]) -> bool:
    content_norm = normalize(doc.get("content", ""))
    if expected_product and doc.get("product") != expected_product:
        return False

    if not phrases:
        return True

    content_ok = all(normalize(phrase) in content_norm for phrase in phrases)
    if content_ok:
        return True

    joined_phrases = " ".join(normalize(phrase) for phrase in phrases)
    return joined_phrases in content_norm


def first_hit_rank(docs: List[dict], expected_product: str, phrases: List[str]) -> int:
    for idx, doc in enumerate(docs, 1):
        if doc_matches(doc, expected_product, phrases):
            return idx
    return -1


def image_hit(docs: List[dict], expected_product: str, phrases: List[str]) -> bool:
    for doc in docs:
        if doc_matches(doc, expected_product, phrases) and doc.get("images"):
            return True
    return False


def summarize(mode_name: str, rows: List[dict]) -> None:
    print(f"\n=== {mode_name} ===")
    total = len(rows)
    hit10 = sum(1 for row in rows if 0 < row["rank"] <= 10)
    hit20 = sum(1 for row in rows if 0 < row["rank"] <= 20)
    print(f"Overall Recall@10: {hit10}/{total} = {hit10 / total:.3f}")
    print(f"Overall Recall@20: {hit20}/{total} = {hit20 / total:.3f}")

    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)

    print("By Category:")
    for category, items in sorted(by_cat.items()):
        c_hit10 = sum(1 for row in items if 0 < row["rank"] <= 10)
        c_hit20 = sum(1 for row in items if 0 < row["rank"] <= 20)
        image_items = [row for row in items if row["prefer_image"]]
        if image_items:
            image_hit_count = sum(1 for row in image_items if row["image_hit"])
            image_text = f", image-hit={image_hit_count}/{len(image_items)}"
        else:
            image_text = ""
        print(
            f"  - {category}: R@10={c_hit10}/{len(items)} ({c_hit10 / len(items):.3f}), "
            f"R@20={c_hit20}/{len(items)} ({c_hit20 / len(items):.3f}){image_text}"
        )

    misses = [row for row in rows if row["rank"] < 0]
    if misses:
        print("Misses:")
        for row in misses:
            print(f"  - qid={row['qid']} [{row['category']}] {row['question']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    questions = load_questions()
    eval_set = load_eval_set()

    modes = {
        "semantic_only": semantic_only,
        "dense_plus_bm25": dense_plus_bm25,
        "dense_plus_bm25_with_rerank": dense_plus_bm25_with_rerank,
    }

    retriever = ManualRetriever(enable_rerank=True)

    all_results = {}
    for mode_name, fn in modes.items():
        rows = []
        print(f"[mode] {mode_name}", flush=True)
        for idx, item in enumerate(eval_set, 1):
            qid = item["qid"]
            question = questions[qid]
            docs = fn(retriever, question, args.top_k)
            rank = first_hit_rank(docs, item["expected_product"], item.get("phrases", []))
            rows.append(
                {
                    "qid": qid,
                    "question": question,
                    "category": item["category"],
                    "prefer_image": item.get("prefer_image", False),
                    "rank": rank,
                    "image_hit": image_hit(docs, item["expected_product"], item.get("phrases", [])),
                }
            )
            if idx % 5 == 0:
                print(f"  processed {idx}/{len(eval_set)}", flush=True)
        all_results[mode_name] = rows
        summarize(mode_name, rows)

    out_dir = Path("experiments") / "retrieval_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
