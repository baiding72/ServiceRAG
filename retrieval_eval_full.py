import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from retrieval_eval import (
    semantic_only,
    dense_plus_bm25,
    dense_plus_bm25_with_rerank,
)
from retriever import ManualRetriever


QUESTION_FILE = Path("data/question_public.csv")
CUSTOMER_SERVICE_TERMS = [
    "退货", "换货", "发票", "物流", "退款", "投诉", "快递", "运费", "开发票", "维修",
    "售后", "补发", "客服", "包装", "7天无理由", "发货", "赔偿", "取消订单", "质保卡",
]
CHINESE_GENERIC_TERMS = {
    "如何", "什么", "哪些", "时候", "需要", "使用", "进行", "操作", "功能", "步骤",
    "告诉", "请问", "可以", "这个", "那个", "实现", "正确", "怎么", "怎样", "为什么",
}
ENGLISH_STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "be", "to", "of", "on", "in", "at", "for",
    "and", "or", "if", "my", "your", "their", "what", "how", "do", "does", "did", "can",
    "could", "should", "would", "i", "you", "we", "they", "it", "this", "that", "these",
    "those", "before", "after", "with", "when", "while", "into", "from", "about", "want",
    "ready", "using", "use", "used", "according", "manual", "proper", "correct", "different",
    "position", "should", "there", "know", "tell", "need"
}


PRODUCT_RULES: List[Tuple[range, str, str]] = [
    (range(64, 70), "吹风机", "steps"),
    (range(70, 86), "空调", "screen_button"),
    (range(86, 89), "蒸汽清洁机", "steps"),
    (range(89, 92), "人体工学椅", "chinese_tech"),
    (range(92, 104), "洗碗机", "steps"),
    (range(104, 113), "空气净化器", "steps"),
    (range(113, 123), "健身单车", "chinese_tech"),
    (range(123, 131), "电钻", "steps"),
    (range(131, 145), "健身追踪器", "chinese_tech"),
    (range(145, 147), "冰箱", "steps"),
    (range(153, 173), "发电机", "steps"),
    (range(173, 181), "摩托艇", "steps"),
    (range(181, 186), "水泵", "steps"),
    (range(186, 191), "可编程温控器", "screen_button"),
    (range(200, 206), "功能键盘", "steps"),
    (range(206, 208), "儿童电动摩托车", "steps"),
    (range(208, 216), "蓝牙激光鼠标", "steps"),
    (range(216, 228), "烤箱", "steps"),
    (range(228, 235), "相机", "steps"),
    (range(241, 411), "汇总英文", "english"),
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def load_questions() -> List[Dict[str, str]]:
    return list(csv.DictReader(QUESTION_FILE.open("r", encoding="utf-8")))


def infer_product_and_category(qid: int, question: str) -> Tuple[Optional[str], str, bool]:
    if any(term in question for term in CUSTOMER_SERVICE_TERMS):
        return None, "customer_service", False

    for qid_range, product, category in PRODUCT_RULES:
        if qid in qid_range:
            if product == "汇总英文":
                if qid >= 300:
                    if any(term in question.lower() for term in ("screen", "button", "menu", "history", "indicator", "led")):
                        category = "screen_button"
                    elif "?" in question and question.lower().count("what ") >= 2:
                        category = "multi_subquestion"
                return product, category, True
            if product == "可编程温控器":
                return product, "screen_button", True
            return product, category, True

    if re.search(r"[A-Za-z]{3,}", question):
        return "汇总英文", "english", True

    return None, "unmapped", False


def extract_weak_phrases(question: str) -> List[str]:
    phrases: List[str] = []
    q = normalize(question)

    english_tokens = [
        token for token in re.findall(r"[a-z0-9][a-z0-9/_-]{1,}", q)
        if token not in ENGLISH_STOPWORDS and len(token) >= 3
    ]
    if english_tokens:
        for size in (2, 1):
            for idx in range(0, len(english_tokens) - size + 1):
                phrase = " ".join(english_tokens[idx: idx + size])
                if phrase not in phrases:
                    phrases.append(phrase)
                if len(phrases) >= 3:
                    return phrases

    chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", question)
    for chunk in chinese_chunks:
        if chunk in CHINESE_GENERIC_TERMS:
            continue
        if len(chunk) <= 6:
            if chunk not in phrases:
                phrases.append(chunk)
        else:
            for sub in re.findall(r"[\u4e00-\u9fff]{2,4}", chunk):
                if sub in CHINESE_GENERIC_TERMS:
                    continue
                if sub not in phrases:
                    phrases.append(sub)
                if len(phrases) >= 3:
                    return phrases
        if len(phrases) >= 3:
            return phrases

    return phrases[:3]


def doc_matches_weak(doc: dict, expected_product: str, phrases: List[str]) -> bool:
    if expected_product and doc.get("product") != expected_product:
        return False

    if not phrases:
        return True

    content_norm = normalize(doc.get("content", ""))
    hits = sum(1 for phrase in phrases if normalize(phrase) in content_norm)
    return hits >= 1


def first_hit_rank(docs: List[dict], expected_product: str, phrases: List[str]) -> int:
    for idx, doc in enumerate(docs, 1):
        if doc_matches_weak(doc, expected_product, phrases):
            return idx
    return -1


def summarize(mode_name: str, rows: List[dict], scored_rows: List[dict]) -> None:
    print(f"\n=== {mode_name} ===")
    total = len(scored_rows)
    hit10 = sum(1 for row in scored_rows if 0 < row["rank"] <= 10)
    hit20 = sum(1 for row in scored_rows if 0 < row["rank"] <= 20)
    print(f"Scored Questions: {total}/{len(rows)}")
    print(f"Recall@10: {hit10}/{total} = {hit10 / total:.3f}")
    print(f"Recall@20: {hit20}/{total} = {hit20 / total:.3f}")

    by_cat = defaultdict(list)
    for row in scored_rows:
        by_cat[row["category"]].append(row)

    print("By Category:")
    for category, items in sorted(by_cat.items()):
        c_hit10 = sum(1 for row in items if 0 < row["rank"] <= 10)
        c_hit20 = sum(1 for row in items if 0 < row["rank"] <= 20)
        print(
            f"  - {category}: R@10={c_hit10}/{len(items)} ({c_hit10 / len(items):.3f}), "
            f"R@20={c_hit20}/{len(items)} ({c_hit20 / len(items):.3f})"
        )

    unscored = Counter(row["category"] for row in rows if not row["scored"])
    if unscored:
        print(f"Unscored: {dict(unscored)}")


def main() -> None:
    questions = load_questions()
    retriever = ManualRetriever(enable_rerank=True)

    modes = {
        "semantic_only": semantic_only,
        "dense_plus_bm25": dense_plus_bm25,
        "dense_plus_bm25_with_rerank": dense_plus_bm25_with_rerank,
    }

    all_results = {}
    for mode_name, fn in modes.items():
        rows = []
        print(f"[mode] {mode_name}", flush=True)
        for idx, row in enumerate(questions, 1):
            qid = int(row["id"])
            question = row["question"].strip().strip('"')
            expected_product, category, scored = infer_product_and_category(qid, question)
            phrases = extract_weak_phrases(question) if scored else []
            docs = fn(retriever, question, 20)
            rank = first_hit_rank(docs, expected_product or "", phrases) if scored else -1
            rows.append(
                {
                    "qid": qid,
                    "question": question,
                    "category": category,
                    "scored": scored,
                    "expected_product": expected_product,
                    "phrases": phrases,
                    "rank": rank,
                }
            )
            if idx % 25 == 0:
                print(f"  processed {idx}/{len(questions)}", flush=True)

        scored_rows = [row for row in rows if row["scored"]]
        summarize(mode_name, rows, scored_rows)
        all_results[mode_name] = rows

    out_dir = Path("experiments") / "retrieval_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "full_weak_latest.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
