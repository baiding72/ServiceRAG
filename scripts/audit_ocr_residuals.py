"""Audit likely OCR residuals in structured knowledge.

This is a heuristic audit, not an automatic cleaner. It is intended to surface
high-risk chunks for manual review before deciding whether to rewrite source
manual text.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


PATTERNS = {
    "hyphen_word_break": re.compile(r"\b[A-Za-z]{2,}-\s+[A-Za-z]{2,}\b"),
    "dot_leader_toc": re.compile(r"\.{4,}|…{2,}"),
    "latex_residue": re.compile(r"\$\^|\\boxed|\\begin|\\end|\{array\}|\$\{?\^"),
    "page_ref_noise": re.compile(r"第\s*\d+\s*页|\bpage\s*\d+\b", re.IGNORECASE),
    "joined_english_words": re.compile(
        r"\b(?:OWNER'S|USER|OPERATOR|INSTRUCTION|SAFETY|IMPORTANT|PRODUCT|QUICK|COLOR)"
        r"(?:MANUAL|GUIDE|INSTRUCTIONS|SAFEGUARDS|TELEVISION)\b",
        re.IGNORECASE,
    ),
    "mixed_case_ocr": re.compile(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b"),
    "toc_heading": re.compile(r"#\s*(目录|table of contents|contents)\b", re.IGNORECASE),
}


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def audit(data: List[Dict[str, Any]], sample_limit: int) -> Dict[str, Any]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {name: 0 for name in PATTERNS})
    samples: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(
        lambda: {name: [] for name in PATTERNS}
    )

    for item in data:
        if item.get("level") != "child":
            continue
        product = str(item.get("product") or "")
        content = str(item.get("content") or "")
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(content):
                counts[product][name] += 1
                if len(samples[product][name]) < sample_limit:
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 140)
                    samples[product][name].append(
                        {
                            "chunk_id": str(item.get("chunk_id") or ""),
                            "section_title": str(item.get("section_title") or ""),
                            "text": compact(content[start:end]),
                        }
                    )

    return {"counts": counts, "samples": samples}


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# OCR Residual Audit",
        "",
        "This report lists heuristic OCR/noise signals found in child chunks.",
        "",
        "## Counts",
        "",
        "| product | issue | count |",
        "|---|---:|---:|",
    ]
    for product in sorted(report["counts"]):
        for issue, count in sorted(report["counts"][product].items()):
            if count:
                lines.append(f"| {product} | {issue} | {count} |")

    lines.extend(["", "## Samples", ""])
    for product in sorted(report["samples"]):
        product_has_sample = any(report["samples"][product][issue] for issue in PATTERNS)
        if not product_has_sample:
            continue
        lines.extend([f"### {product}", ""])
        for issue in PATTERNS:
            issue_samples = report["samples"][product][issue]
            if not issue_samples:
                continue
            lines.extend([f"#### {issue}", ""])
            for sample in issue_samples:
                lines.extend(
                    [
                        f"- chunk_id: `{sample['chunk_id']}`",
                        f"- section: `{sample['section_title']}`",
                        "",
                        f"> {sample['text']}",
                        "",
                    ]
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured", default="data/structured_knowledge.json")
    parser.add_argument("--json", default="eval_reports/ocr_residual_audit.json")
    parser.add_argument("--markdown", default="eval_reports/ocr_residual_audit.md")
    parser.add_argument("--sample-limit", type=int, default=3)
    args = parser.parse_args()

    data = json.loads(Path(args.structured).read_text(encoding="utf-8"))
    report = audit(data, args.sample_limit)

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.markdown))

    summary = {}
    for product, product_counts in report["counts"].items():
        non_zero = {key: value for key, value in product_counts.items() if value}
        if non_zero:
            summary[product] = non_zero
    print(json.dumps({"products_with_issues": summary, "json": args.json, "markdown": args.markdown}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
