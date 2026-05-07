"""Audit <PIC> placeholder order against image id lists.

This script checks two levels:
1. Raw manual records: number of <PIC> placeholders vs source image ids.
2. Structured chunks: each chunk must have exactly one image id per <PIC>.

For position/order review, it also emits snippets around each <PIC> with the
image id that the parser binds to that placeholder.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List


PIC = "<PIC>"


def load_parse_manuals() -> Any:
    module_path = Path("parse_manuals.py")
    spec = importlib.util.spec_from_file_location("parse_manuals_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snippet_around_pic(text: str, pic_index: int, window: int = 80) -> str:
    start = 0
    positions: List[int] = []
    while True:
        index = text.find(PIC, start)
        if index < 0:
            break
        positions.append(index)
        start = index + len(PIC)

    if pic_index >= len(positions):
        return ""

    index = positions[pic_index]
    left = max(0, index - window)
    right = min(len(text), index + len(PIC) + window)
    return text[left:right].replace("\n", " ").strip()


def audit_raw_manuals(manual_dir: Path, product_filter: str | None = None) -> List[Dict[str, Any]]:
    pm = load_parse_manuals()
    rows: List[Dict[str, Any]] = []
    for path in sorted(manual_dir.glob("*.txt")):
        product = path.stem.replace("手册", "")
        if product_filter and product_filter not in product and product_filter not in path.name:
            continue
        raw = pm.read_file_with_encoding(path)
        if raw is None:
            rows.append({"manual": path.name, "error": "read_failed"})
            continue
        records = pm.parse_raw_manual_records(raw, path.name)
        pic_count = sum(record_text.count(PIC) for record_text, _ in records)
        image_count = sum(len(record_images) for _, record_images in records)
        rows.append(
            {
                "manual": path.name,
                "records": len(records),
                "pic_count": pic_count,
                "image_count": image_count,
                "diff_image_minus_pic": image_count - pic_count,
                "status": "ok" if pic_count == image_count else "mismatch",
            }
        )
    return rows


def audit_structured_chunks(structured_path: Path, product_filter: str | None = None, sample_limit: int = 5) -> Dict[str, Any]:
    data = json.loads(structured_path.read_text(encoding="utf-8"))
    by_product: Dict[str, Dict[str, Any]] = {}
    mismatches: List[Dict[str, Any]] = []
    samples: Dict[str, List[Dict[str, Any]]] = {}

    for item in data:
        product = str(item.get("product") or "")
        if product_filter and product_filter not in product and product_filter not in str(item.get("source_file") or ""):
            continue
        if item.get("level", "child") != "child":
            continue

        content = str(item.get("content") or "")
        images = item.get("images") or []
        if not isinstance(images, list):
            images = []

        stat = by_product.setdefault(
            product,
            {
                "child_chunks": 0,
                "chunks_with_images": 0,
                "pic_count": 0,
                "image_count": 0,
                "mismatch_chunks": 0,
            },
        )
        pic_count = content.count(PIC)
        stat["child_chunks"] += 1
        stat["pic_count"] += pic_count
        stat["image_count"] += len(images)
        if images:
            stat["chunks_with_images"] += 1

        if pic_count != len(images):
            stat["mismatch_chunks"] += 1
            mismatches.append(
                {
                    "product": product,
                    "chunk_id": item.get("chunk_id"),
                    "pic_count": pic_count,
                    "image_count": len(images),
                    "images": images,
                    "text_head": content[:240],
                }
            )
            continue

        if images and len(samples.setdefault(product, [])) < sample_limit:
            bindings = []
            for index, image_id in enumerate(images):
                bindings.append(
                    {
                        "pic_order": index + 1,
                        "image_id": image_id,
                        "snippet": snippet_around_pic(content, index),
                    }
                )
            samples[product].append(
                {
                    "chunk_id": item.get("chunk_id"),
                    "section_title": item.get("section_title"),
                    "bindings": bindings,
                }
            )

    return {
        "by_product": by_product,
        "mismatches": mismatches,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-dir", default="data/手册")
    parser.add_argument("--structured", default="data/structured_knowledge.json")
    parser.add_argument("--product", default="")
    parser.add_argument("--sample-limit", type=int, default=3)
    parser.add_argument("--report", default="eval_reports/pic_order_audit.json")
    args = parser.parse_args()

    product_filter = args.product or None
    report = {
        "raw_manuals": audit_raw_manuals(Path(args.manual_dir), product_filter),
        "structured": audit_structured_chunks(Path(args.structured), product_filter, args.sample_limit),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Raw manual count check:")
    print(f'{"manual":<34} {"records":>7} {"PIC":>5} {"IDs":>5} {"diff":>6} status')
    for row in report["raw_manuals"]:
        print(
            f'{row["manual"]:<34} {row.get("records", 0):>7} '
            f'{row.get("pic_count", 0):>5} {row.get("image_count", 0):>5} '
            f'{row.get("diff_image_minus_pic", 0):>6} {row.get("status", "error")}'
        )

    print("\nStructured chunk check:")
    print(f'{"product":<28} {"child":>5} {"with_img":>8} {"PIC":>5} {"imgs":>5} {"bad":>5}')
    for product, stat in sorted(report["structured"]["by_product"].items()):
        print(
            f'{product:<28} {stat["child_chunks"]:>5} {stat["chunks_with_images"]:>8} '
            f'{stat["pic_count"]:>5} {stat["image_count"]:>5} {stat["mismatch_chunks"]:>5}'
        )

    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
