"""Clean and validate competition submission CSV files.

The official evaluator is strict about the answer string format:

    answer text <PIC> ... , ["image_id_1", "image_id_2"]

This script normalizes CSV encoding, removes unsafe control characters, repairs
malformed image arrays when possible, and enforces one-to-one alignment between
`<PIC>` placeholders and the trailing image id array.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


TRAILING_ARRAY_RE = re.compile(r"(?:,|，)?\s*(\[[^\[\]]*\])\s*$")
PIC = "<PIC>"


@dataclass
class RowFix:
    row_id: str
    actions: List[str] = field(default_factory=list)
    before_pic_count: int = 0
    before_image_count: int = 0
    after_pic_count: int = 0
    after_image_count: int = 0


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("＜PIC＞", PIC).replace("<pic>", PIC).replace("<Pic>", PIC)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    chars = []
    for ch in text:
        code = ord(ch)
        category = unicodedata.category(ch)
        if code < 32 and ch not in "\t\n\r":
            continue
        if code == 127 or code > 0xFFFF:
            continue
        if category in {"Cf", "Cs"}:
            continue
        chars.append(ch)
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def parse_image_array(raw: str) -> Tuple[List[str], bool]:
    candidate = raw.strip()
    if not candidate:
        return [], True
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            ids = re.findall(r'["\']([^"\']+)["\']', candidate)
            return [str(item).strip() for item in ids if str(item).strip()], False
    if not isinstance(parsed, list):
        return [], False
    return [str(item).strip() for item in parsed if str(item).strip()], True


def split_answer_and_images(answer: str) -> Tuple[str, List[str], bool]:
    match = TRAILING_ARRAY_RE.search(answer)
    if not match:
        return answer, [], True
    images, parsed_cleanly = parse_image_array(match.group(1))
    text = answer[: match.start()].rstrip(" ,，")
    return text, images, parsed_cleanly


def remove_last_pic(text: str) -> str:
    index = text.rfind(PIC)
    if index < 0:
        return text
    return (text[:index] + text[index + len(PIC):]).strip()


def align_pics_and_images(
    text: str,
    images: Iterable[str],
    allowed_image_ids: Optional[set[str]] = None,
) -> Tuple[str, List[str], List[str]]:
    actions: List[str] = []
    deduped = list(dict.fromkeys(str(item).strip() for item in images if str(item).strip()))
    if len(deduped) != len(list(images)):
        actions.append("dedupe_image_ids")

    if allowed_image_ids is not None:
        before = len(deduped)
        deduped = [image_id for image_id in deduped if image_id in allowed_image_ids]
        if len(deduped) != before:
            actions.append("drop_unknown_image_ids")

    pic_count = text.count(PIC)
    if not deduped:
        if pic_count:
            actions.append("remove_all_pic_without_images")
        return re.sub(r"\s+", " ", text.replace(PIC, "")).strip(), [], actions

    if pic_count == 0:
        actions.append("drop_images_without_pic")
        return text, [], actions

    if pic_count > len(deduped):
        actions.append("remove_extra_pic")
        while text.count(PIC) > len(deduped):
            text = remove_last_pic(text)
    elif pic_count < len(deduped):
        actions.append("truncate_extra_images")
        deduped = deduped[:pic_count]

    if not deduped:
        text = text.replace(PIC, "")
    return re.sub(r"\s+", " ", text).strip(), deduped, actions


def format_answer(text: str, images: List[str]) -> str:
    if images:
        return f"{text.strip()} , {json.dumps(images, ensure_ascii=False)}"
    return text.strip()


def load_allowed_image_ids(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    if path.is_dir():
        return {
            file.stem
            for file in path.rglob("*")
            if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        }
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(item) for item in data}
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def clean_submission(input_path: Path, output_path: Path, report_path: Path, allowed_image_ids: Optional[set[str]]) -> dict:
    with input_path.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        if "id" not in (reader.fieldnames or []) or "ret" not in (reader.fieldnames or []):
            raise ValueError(f"CSV must contain id and ret columns, got: {reader.fieldnames}")
        source_rows = list(reader)

    cleaned_rows = []
    fixes: List[RowFix] = []
    seen_ids = set()
    duplicate_ids = []

    for row in source_rows:
        row_id = str(row.get("id", "")).strip()
        if row_id in seen_ids:
            duplicate_ids.append(row_id)
            continue
        seen_ids.add(row_id)

        original = normalize_text(row.get("ret", ""))
        text, images, parsed_cleanly = split_answer_and_images(original)
        fix = RowFix(
            row_id=row_id,
            before_pic_count=original.count(PIC),
            before_image_count=len(images),
        )
        if not parsed_cleanly:
            fix.actions.append("repair_malformed_image_array")

        text, images, actions = align_pics_and_images(text, images, allowed_image_ids=allowed_image_ids)
        fix.actions.extend(actions)
        fix.after_pic_count = text.count(PIC)
        fix.after_image_count = len(images)
        answer = format_answer(text, images)

        if answer != original or fix.actions:
            fixes.append(fix)
        cleaned_rows.append({"id": row_id, "ret": answer})

    cleaned_rows.sort(key=lambda item: int(item["id"]) if item["id"].isdigit() else item["id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["id", "ret"])
        writer.writeheader()
        writer.writerows(cleaned_rows)

    report = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "row_count_in": len(source_rows),
        "row_count_out": len(cleaned_rows),
        "duplicate_ids_removed": duplicate_ids,
        "fix_count": len(fixes),
        "fixes": [
            {
                "id": fix.row_id,
                "actions": fix.actions,
                "before_pic_count": fix.before_pic_count,
                "before_image_count": fix.before_image_count,
                "after_pic_count": fix.after_pic_count,
                "after_image_count": fix.after_image_count,
            }
            for fix in fixes
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and validate submission CSV answer format.")
    parser.add_argument("--input", default="data/submission.csv")
    parser.add_argument("--output", default="data/submission.cleaned.csv")
    parser.add_argument("--report", default="eval_reports/submission_clean_report.json")
    parser.add_argument("--allowed-images", default="data/手册/插图", help="Image directory, JSON list, or text file. Use '' to disable.")
    args = parser.parse_args()

    allowed = load_allowed_image_ids(Path(args.allowed_images)) if args.allowed_images else None
    report = clean_submission(Path(args.input), Path(args.output), Path(args.report), allowed)
    print(json.dumps({k: v for k, v in report.items() if k != "fixes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

