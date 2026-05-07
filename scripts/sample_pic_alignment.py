"""Sample <PIC> to image-id bindings for visual alignment review.

The output JSON is machine-readable, while the Markdown report is meant for
manual review in any Markdown reader. Image links in the Markdown are relative
to the report file location, and each sample also prints the source image path.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PIC = "<PIC>"
HIGH_RISK_PRODUCTS = {"发电机", "可编程温控器", "洗碗机"}
IMAGE_DENSE_PRODUCTS = {"汇总英文", "健身追踪器", "相机"}
DEFAULT_STRATA = {
    "raw_mismatch_high_risk": 30,
    "image_dense": 20,
    "ordinary_cn": 20,
    "random_backfill": 10,
}


def load_image_index(image_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for file in image_dir.rglob("*"):
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            index.setdefault(file.stem, file)
    return index


def pic_positions(text: str) -> List[int]:
    positions: List[int] = []
    start = 0
    while True:
        index = text.find(PIC, start)
        if index < 0:
            return positions
        positions.append(index)
        start = index + len(PIC)


def make_snippet(text: str, pic_index: int, window: int) -> Tuple[str, str, str]:
    positions = pic_positions(text)
    if pic_index >= len(positions):
        return "", "", ""
    pos = positions[pic_index]
    before = text[max(0, pos - window):pos].replace("\n", " ").strip()
    after = text[pos + len(PIC):pos + len(PIC) + window].replace("\n", " ").strip()
    around = text[max(0, pos - window):pos + len(PIC) + window].replace("\n", " ").strip()
    return before, after, around


def collect_bindings(structured_path: Path, image_index: Dict[str, Path], snippet_window: int) -> List[Dict[str, Any]]:
    data = json.loads(structured_path.read_text(encoding="utf-8"))
    bindings: List[Dict[str, Any]] = []
    for item in data:
        if item.get("level", "child") != "child":
            continue
        content = str(item.get("content") or "")
        images = item.get("images") or []
        if not isinstance(images, list) or not images:
            continue
        if content.count(PIC) != len(images):
            continue
        product = str(item.get("product") or "")
        for index, image_id in enumerate(images):
            before, after, around = make_snippet(content, index, snippet_window)
            image_path = image_index.get(str(image_id))
            bindings.append(
                {
                    "product": product,
                    "chunk_id": item.get("chunk_id"),
                    "parent_id": item.get("parent_id"),
                    "section_title": item.get("section_title") or "",
                    "source_file": item.get("source_file") or "",
                    "pic_order": index + 1,
                    "image_id": str(image_id),
                    "image_path": str(image_path) if image_path else "",
                    "image_exists": image_path is not None,
                    "snippet_before": before,
                    "snippet_after": after,
                    "snippet_around": around,
                    "content_type": item.get("content_type") or "",
                    "language": item.get("language") or "",
                }
            )
    return bindings


def sample_without_replacement(
    rng: random.Random,
    candidates: Sequence[Dict[str, Any]],
    count: int,
    used_keys: set[Tuple[str, int, str]],
) -> List[Dict[str, Any]]:
    available = [
        item for item in candidates
        if (str(item["chunk_id"]), int(item["pic_order"]), str(item["image_id"])) not in used_keys
    ]
    if not available or count <= 0:
        return []
    picked = rng.sample(list(available), min(count, len(available)))
    for item in picked:
        used_keys.add((str(item["chunk_id"]), int(item["pic_order"]), str(item["image_id"])))
    return picked


def stratified_sample(bindings: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    used: set[Tuple[str, int, str]] = set()

    high_risk = [b for b in bindings if b["product"] in HIGH_RISK_PRODUCTS]
    dense = [b for b in bindings if b["product"] in IMAGE_DENSE_PRODUCTS]
    ordinary = [
        b for b in bindings
        if b["product"] not in HIGH_RISK_PRODUCTS
        and b["product"] not in IMAGE_DENSE_PRODUCTS
        and b.get("language") != "en"
    ]

    plan = DEFAULT_STRATA.copy()
    if n != sum(DEFAULT_STRATA.values()):
        scale = n / sum(DEFAULT_STRATA.values())
        plan = {key: max(1, round(value * scale)) for key, value in DEFAULT_STRATA.items()}
        delta = n - sum(plan.values())
        plan["random_backfill"] += delta

    samples: List[Dict[str, Any]] = []
    for stratum, candidates in [
        ("raw_mismatch_high_risk", high_risk),
        ("image_dense", dense),
        ("ordinary_cn", ordinary),
    ]:
        picked = sample_without_replacement(rng, candidates, plan[stratum], used)
        for item in picked:
            item["stratum"] = stratum
        samples.extend(picked)

    remaining_count = n - len(samples)
    picked = sample_without_replacement(rng, bindings, remaining_count, used)
    for item in picked:
        item["stratum"] = "random_backfill"
    samples.extend(picked)

    return samples[:n]


def relpath_for_markdown(path: str, markdown_path: Path) -> str:
    if not path:
        return ""
    import os

    return os.path.relpath(Path(path).resolve(), markdown_path.parent.resolve())


def write_markdown(samples: List[Dict[str, Any]], markdown_path: Path) -> None:
    lines = [
        "# PIC Alignment Samples",
        "",
        "每个样本按 `第 N 个 <PIC> -> 第 N 个 image_id` 展示。人工标注建议：`2=匹配`，`1=弱匹配`，`0=错位`。",
        "",
        "| 字段 | 说明 |",
        "|---|---|",
        "| stratum | 抽样分层 |",
        "| snippet_before / snippet_after | `<PIC>` 前后文字 |",
        "| image_path | 原始图片路径 |",
        "",
    ]

    for idx, item in enumerate(samples, start=1):
        image_path = item.get("image_path", "")
        md_image_path = relpath_for_markdown(image_path, markdown_path)
        lines.extend(
            [
                f"## Sample {idx:03d}",
                "",
                f"- stratum: `{item.get('stratum', '')}`",
                f"- product: `{item.get('product', '')}`",
                f"- chunk_id: `{item.get('chunk_id', '')}`",
                f"- parent_id: `{item.get('parent_id', '')}`",
                f"- section_title: `{item.get('section_title', '')}`",
                f"- pic_order: `{item.get('pic_order', '')}`",
                f"- image_id: `{item.get('image_id', '')}`",
                f"- image_exists: `{item.get('image_exists', False)}`",
                f"- image_path: `{image_path}`",
                "",
            ]
        )
        if md_image_path:
            lines.extend([f"![{item.get('image_id', '')}]({md_image_path})", ""])
        else:
            lines.extend(["**Image path missing.**", ""])
        lines.extend(
            [
                "**snippet_before**",
                "",
                f"> {item.get('snippet_before', '')}",
                "",
                "**PIC binding**",
                "",
                f"> `<PIC>` -> `{item.get('image_id', '')}`",
                "",
                "**snippet_after**",
                "",
                f"> {item.get('snippet_after', '')}",
                "",
                "**snippet_around**",
                "",
                f"> {item.get('snippet_around', '')}",
                "",
                "**manual label**: `[ ] 2 match` `[ ] 1 weak` `[ ] 0 wrong`",
                "",
                "---",
                "",
            ]
        )

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured", default="data/structured_knowledge.json")
    parser.add_argument("--image-dir", default="data/手册/插图")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--snippet-window", type=int, default=120)
    parser.add_argument("--out", default="eval_reports/pic_alignment_samples.json")
    parser.add_argument("--markdown", default="eval_reports/pic_alignment_samples.md")
    args = parser.parse_args()

    image_index = load_image_index(Path(args.image_dir))
    bindings = collect_bindings(Path(args.structured), image_index, args.snippet_window)
    samples = stratified_sample(bindings, args.n, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_path = Path(args.markdown)
    write_markdown(samples, markdown_path)

    by_stratum: Dict[str, int] = {}
    missing_images = 0
    for item in samples:
        by_stratum[item["stratum"]] = by_stratum.get(item["stratum"], 0) + 1
        if not item.get("image_exists"):
            missing_images += 1

    print(
        json.dumps(
            {
                "total_bindings": len(bindings),
                "sampled": len(samples),
                "by_stratum": by_stratum,
                "missing_images_in_samples": missing_images,
                "json": str(out_path),
                "markdown": str(markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
