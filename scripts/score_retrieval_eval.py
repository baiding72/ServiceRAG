import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_CANDIDATES = Path("pooled_candidates.json")
DEFAULT_LABELS = Path("labeled_gold_set.jsonl")
DEFAULT_OUTPUT = Path("experiments/retrieval_eval/pooled_gold_eval_scores.json")


def load_candidates(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[dict] = []

    for item in data:
        query_id = str(item.get("id", ""))
        query = item.get("query", "")
        category = item.get("category", "unknown")

        for cand in item.get("candidates", []):
            route_ranks = cand.get("route_ranks", {}) or {}
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "category": category,
                    "chunk_id": str(cand.get("chunk_id", "")),
                    "parent_id": str(cand.get("parent_id", "") or ""),
                    "hybrid_rank": route_ranks.get("hybrid"),
                    "dense_rank": route_ranks.get("dense"),
                    "bm25_rank": route_ranks.get("bm25"),
                    "rewrite_hybrid_rank": route_ranks.get("rewrite_hybrid"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"候选集为空: {path}")
    return df


def load_labels(path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(
                {
                    "query_id": str(record["query_id"]),
                    "chunk_id": str(record["chunk_id"]),
                    "grade": int(record["grade"]),
                    "label_parent_id": str(record.get("parent_id", "") or ""),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"标注集为空: {path}")

    df = (
        df.sort_values(["query_id", "chunk_id", "grade"], ascending=[True, True, False])
        .drop_duplicates(subset=["query_id", "chunk_id"], keep="first")
        .reset_index(drop=True)
    )
    return df


def dcg_at_k(grades: np.ndarray) -> float:
    if grades.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, grades.size + 2))
    gains = np.power(2, grades) - 1
    return float(np.sum(gains / discounts))


def ndcg_for_query(ranked_grades: List[int], all_pool_grades: List[int], k: int) -> Optional[float]:
    if not all_pool_grades:
        return None

    ideal = sorted(all_pool_grades, reverse=True)[:k]
    ideal_dcg = dcg_at_k(np.array(ideal, dtype=np.int64))
    if ideal_dcg <= 0:
        return None

    ranked = ranked_grades[:k]
    if len(ranked) < k:
        ranked = ranked + [0] * (k - len(ranked))

    actual_dcg = dcg_at_k(np.array(ranked, dtype=np.int64))
    return actual_dcg / ideal_dcg


def compute_metrics(candidates_df: pd.DataFrame, labels_df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = candidates_df.merge(labels_df, on=["query_id", "chunk_id"], how="left")
    df["grade"] = df["grade"].fillna(0).astype(int)

    query_meta = (
        df[["query_id", "query", "category"]]
        .drop_duplicates(subset=["query_id"])
        .reset_index(drop=True)
    )

    pool_stats = (
        df.groupby("query_id", as_index=False)
        .agg(
            total_relevant_in_pool=("grade", lambda s: int((s >= 1).sum())),
            total_strong_relevant_in_pool=("grade", lambda s: int((s == 2).sum())),
            relevant_parent_ids=("parent_id", lambda s: sorted({x for x in s if x})),
            pool_grades=("grade", list),
        )
    )

    hybrid_df = df[df["hybrid_rank"].notna()].copy()
    hybrid_df["hybrid_rank"] = hybrid_df["hybrid_rank"].astype(int)
    topk = hybrid_df[hybrid_df["hybrid_rank"] <= k].copy()

    topk_stats = (
        topk.groupby("query_id", as_index=False)
        .agg(
            relevant_in_topk=("grade", lambda s: int((s >= 1).sum())),
            strong_relevant_in_topk=("grade", lambda s: int((s == 2).sum())),
            topk_grades=("grade", lambda s: [grade for _, grade in sorted(zip(topk.loc[s.index, "hybrid_rank"], s))]),
            hit_parent_ids=("parent_id", lambda s: sorted({x for x in s if x})),
        )
    )

    result = query_meta.merge(pool_stats, on="query_id", how="left").merge(topk_stats, on="query_id", how="left")

    result["relevant_in_topk"] = result["relevant_in_topk"].fillna(0).astype(int)
    result["strong_relevant_in_topk"] = result["strong_relevant_in_topk"].fillna(0).astype(int)
    result["topk_grades"] = result["topk_grades"].apply(lambda x: x if isinstance(x, list) else [])
    result["hit_parent_ids"] = result["hit_parent_ids"].apply(lambda x: x if isinstance(x, list) else [])

    def _true_recall(row):
        denom = row["total_relevant_in_pool"]
        if denom <= 0:
            return np.nan
        return row["relevant_in_topk"] / denom

    def _strong_recall(row):
        denom = row["total_strong_relevant_in_pool"]
        if denom <= 0:
            return np.nan
        return row["strong_relevant_in_topk"] / denom

    def _hit(row):
        return 1.0 if row["relevant_in_topk"] > 0 else 0.0

    def _strong_hit(row):
        return 1.0 if row["strong_relevant_in_topk"] > 0 else 0.0

    def _precision(row):
        return row["relevant_in_topk"] / k

    def _ndcg(row):
        return ndcg_for_query(row["topk_grades"], row["pool_grades"], k)

    result[f"true_recall@{k}"] = result.apply(_true_recall, axis=1)
    result[f"strong_recall@{k}"] = result.apply(_strong_recall, axis=1)
    result[f"hit@{k}"] = result.apply(_hit, axis=1)
    result[f"strong_hit@{k}"] = result.apply(_strong_hit, axis=1)
    result[f"precision@{k}"] = result.apply(_precision, axis=1)
    result[f"ndcg@{k}"] = result.apply(_ndcg, axis=1)

    return result


def summarize_metrics(per_query_df: pd.DataFrame, k: int) -> pd.DataFrame:
    metric_cols = [
        f"true_recall@{k}",
        f"strong_recall@{k}",
        f"hit@{k}",
        f"strong_hit@{k}",
        f"precision@{k}",
        f"ndcg@{k}",
    ]

    rows = []

    def _row_for(name: str, frame: pd.DataFrame) -> dict:
        row = {"scope": name, "queries": int(frame["query_id"].nunique())}
        for col in metric_cols:
            row[col] = round(float(frame[col].dropna().mean()), 4) if frame[col].notna().any() else None
        return row

    rows.append(_row_for("overall", per_query_df))

    for category, frame in sorted(per_query_df.groupby("category"), key=lambda x: x[0]):
        rows.append(_row_for(category, frame))

    return pd.DataFrame(rows)


def print_pretty_tables(summary_k10: pd.DataFrame, summary_k20: pd.DataFrame) -> None:
    merged = summary_k10.merge(summary_k20, on=["scope", "queries"], how="inner")

    display = merged.copy()
    print("\n=== Overall / Category Summary ===")
    print(
        display.to_string(
            index=False,
            formatters={
                "true_recall@10": lambda v: "-" if pd.isna(v) else f"{v:.4f}",
                "strong_recall@10": lambda v: "-" if pd.isna(v) else f"{v:.4f}",
                "hit@10": lambda v: "-" if pd.isna(v) else f"{v:.4f}",
                "strong_hit@10": lambda v: "-" if pd.isna(v) else f"{v:.4f}",
                "precision@10": lambda v: "-" if pd.isna(v) else f"{v:.4f}",
                "ndcg@10": lambda v: "-" if pd.isna(v) else f"{v:.4f}",
                "true_recall@20": lambda v: "-" if pd.isna(v) else f"{v:.4f}",
                "strong_recall@20": lambda v: "-" if pd.isna(v) else f"{v:.4f}",
                "hit@20": lambda v: "-" if pd.isna(v) else f"{v:.4f}",
                "strong_hit@20": lambda v: "-" if pd.isna(v) else f"{v:.4f}",
                "precision@20": lambda v: "-" if pd.isna(v) else f"{v:.4f}",
                "ndcg@20": lambda v: "-" if pd.isna(v) else f"{v:.4f}",
            },
        )
    )

    metric_order = [
        ("true_recall", "True Recall"),
        ("strong_recall", "Strong Recall"),
        ("hit", "Hit"),
        ("strong_hit", "Strong Hit"),
        ("precision", "Precision"),
        ("ndcg", "nDCG"),
    ]

    print("\n=== Compact Comparison ===")
    for _, row in merged.iterrows():
        print(f"[{row['scope']}] n={row['queries']}")
        for metric_key, metric_name in metric_order:
            v10 = row.get(f"{metric_key}@10")
            v20 = row.get(f"{metric_key}@20")
            s10 = "-" if pd.isna(v10) else f"{v10:.4f}"
            s20 = "-" if pd.isna(v20) else f"{v20:.4f}"
            print(f"  {metric_name:<14} K=10 {s10} | K=20 {s20}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 pooled gold labels 计算真实检索指标")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    candidates_df = load_candidates(args.candidates)
    labels_df = load_labels(args.labels)

    per_query_k10 = compute_metrics(candidates_df, labels_df, k=10)
    per_query_k20 = compute_metrics(candidates_df, labels_df, k=20)

    summary_k10 = summarize_metrics(per_query_k10, k=10)
    summary_k20 = summarize_metrics(per_query_k20, k=20)

    print_pretty_tables(summary_k10, summary_k20)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary_k10": summary_k10.to_dict(orient="records"),
        "summary_k20": summary_k20.to_dict(orient="records"),
        "per_query_k10": per_query_k10.to_dict(orient="records"),
        "per_query_k20": per_query_k20.to_dict(orient="records"),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
