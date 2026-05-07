"""Export Markdown/JSON reports for manual <PIC> to image-id bindings."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, List, Tuple


PIC = "<PIC>"


def load_parse_manuals() -> Any:
    spec = importlib.util.spec_from_file_location("parse_manuals_module", "parse_manuals.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load parse_manuals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_path_for(image_dir: Path, image_id: str) -> Path | None:
    return next(image_dir.glob(image_id + ".*"), None)


def pic_positions(text: str) -> List[int]:
    positions: List[int] = []
    start = 0
    while True:
        index = text.find(PIC, start)
        if index < 0:
            return positions
        positions.append(index)
        start = index + len(PIC)


def export_one(path: Path, image_dir: Path, out_dir: Path, slug: str) -> Tuple[Path, Path]:
    pm = load_parse_manuals()
    raw = pm.read_file_with_encoding(path)
    if raw is None:
        raise RuntimeError(f"Unable to read {path}")
    records = pm.parse_raw_manual_records(raw, path.name)
    if len(records) != 1:
        raise RuntimeError(f"{path.name} contains {len(records)} records; this helper expects one record")
    text, images = records[0]
    positions = pic_positions(text)
    if len(positions) != len(images):
        raise RuntimeError(f"{path.name}: {len(positions)} PIC vs {len(images)} images")

    rows = []
    for index, position in enumerate(positions):
        image_id = str(images[index])
        image_path = image_path_for(image_dir, image_id)
        before = re.sub(r"\s+", " ", text[max(0, position - 180):position]).strip()
        after = re.sub(r"\s+", " ", text[position + len(PIC):position + len(PIC) + 180]).strip()
        rows.append(
            {
                "pic_order": index + 1,
                "image_id": image_id,
                "image_path": str(image_path) if image_path else "",
                "snippet_before": before,
                "snippet_after": after,
                "snippet_around": f"{before} {PIC} {after}".strip(),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{slug}_pic_order_bindings.json"
    md_path = out_dir / f"{slug}_pic_order_bindings.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {path.name} PIC Order Bindings",
        "",
        f"Count: `{len(positions)}` PIC / `{len(images)}` image ids.",
        "",
    ]
    for item in rows:
        image_path = item["image_path"]
        rel_path = "../" + image_path if image_path else ""
        lines.extend(
            [
                f"## PIC {item['pic_order']:03d} -> {item['image_id']}",
                "",
                f"- image_path: `{image_path}`",
                "",
            ]
        )
        if rel_path:
            lines.extend([f"![{item['image_id']}]({rel_path})", ""])
        lines.extend(
            [
                "**before**",
                "",
                f"> {item['snippet_before']}",
                "",
                "**after**",
                "",
                f"> {item['snippet_after']}",
                "",
                "---",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manuals", nargs="+")
    parser.add_argument("--image-dir", default="data/手册/插图")
    parser.add_argument("--out-dir", default="eval_reports/manual_pic_order_reports")
    args = parser.parse_args()

    for manual in args.manuals:
        path = Path(manual)
        slug = path.stem.replace("手册", "")
        json_path, md_path = export_one(path, Path(args.image_dir), Path(args.out_dir), slug)
        print(f"{path.name}: {json_path} {md_path}")


if __name__ == "__main__":
    main()
