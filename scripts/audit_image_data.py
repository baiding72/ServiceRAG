"""Audit image ids in structured knowledge and submission files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List


PIC = "<PIC>"


def image_file_ids(image_dir: Path) -> set[str]:
    return {
        file.stem
        for file in image_dir.rglob("*")
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }


def audit_structured_knowledge(path: Path, available_ids: set[str]) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = []
    duplicate_inside_chunk = []
    pic_mismatch = []
    image_counter = Counter()

    for item in data:
        images = item.get("images") or []
        if not isinstance(images, list):
            images = []
        for image_id in images:
            image_counter[str(image_id)] += 1
            if str(image_id) not in available_ids:
                missing.append({"chunk_id": item.get("chunk_id"), "image_id": image_id, "product": item.get("product")})
        if len(images) != len(set(images)):
            duplicate_inside_chunk.append({"chunk_id": item.get("chunk_id"), "images": images})
        pic_count = str(item.get("content", "")).count(PIC)
        if pic_count and pic_count != len(images):
            pic_mismatch.append({"chunk_id": item.get("chunk_id"), "pic_count": pic_count, "image_count": len(images), "images": images})

    return {
        "records": len(data),
        "records_with_images": sum(1 for item in data if item.get("images")),
        "unique_image_ids_in_knowledge": len(image_counter),
        "missing_image_refs": missing,
        "duplicate_images_inside_chunk": duplicate_inside_chunk,
        "content_pic_image_mismatch": pic_mismatch,
        "most_reused_images": image_counter.most_common(20),
    }


def audit_submission(path: Path, available_ids: set[str]) -> Dict[str, object]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    missing = []
    mismatch = []
    malformed = []
    trailing_array_re = re.compile(r"(?:,|，)?\s*(\[[^\[\]]*\])\s*$")
    for row in rows:
        ret = row.get("ret", "")
        match = trailing_array_re.search(ret)
        images: List[str] = []
        if match:
            try:
                images = json.loads(match.group(1))
            except json.JSONDecodeError:
                malformed.append(row.get("id"))
        if ret.count(PIC) != len(images):
            if ret.count(PIC) or images:
                mismatch.append({"id": row.get("id"), "pic_count": ret.count(PIC), "image_count": len(images)})
        for image_id in images:
            if str(image_id) not in available_ids:
                missing.append({"id": row.get("id"), "image_id": image_id})
    ids = [row.get("id") for row in rows]
    duplicates = sorted([item for item, count in Counter(ids).items() if count > 1], key=lambda x: int(x) if str(x).isdigit() else str(x))
    return {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": duplicates,
        "malformed_image_arrays": malformed,
        "pic_image_mismatch": mismatch,
        "missing_image_refs": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured", default="data/structured_knowledge.json")
    parser.add_argument("--submission", default="data/submission.csv")
    parser.add_argument("--image-dir", default="data/手册/插图")
    parser.add_argument("--report", default="eval_reports/image_data_audit.json")
    args = parser.parse_args()

    available_ids = image_file_ids(Path(args.image_dir))
    report = {
        "available_image_files": len(available_ids),
        "structured_knowledge": audit_structured_knowledge(Path(args.structured), available_ids),
        "submission": audit_submission(Path(args.submission), available_ids),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "available_image_files": report["available_image_files"],
        "structured_missing_refs": len(report["structured_knowledge"]["missing_image_refs"]),
        "structured_pic_mismatch": len(report["structured_knowledge"]["content_pic_image_mismatch"]),
        "submission_missing_refs": len(report["submission"]["missing_image_refs"]),
        "submission_pic_mismatch": len(report["submission"]["pic_image_mismatch"]),
        "report": str(report_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
