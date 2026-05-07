"""Clean confirmed manual-level <PIC> / image-id alignment errors.

The script backs up original manual files, then rewrites the manual txt files
used by the knowledge base. It only applies narrow, verified fixes.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


PIC = "<PIC>"


def load_record(path: Path) -> Tuple[str, List[str]]:
    raw = path.read_text(encoding="utf-8")
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        data = ast.literal_eval(raw)
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(f"Unsupported manual format: {path}")
    text = str(data[0])
    images = data[1]
    if not isinstance(images, list):
        raise ValueError(f"Image list missing or invalid: {path}")
    return text, [str(item) for item in images]


def write_record(path: Path, text: str, images: List[str]) -> None:
    path.write_text(json.dumps([text, images], ensure_ascii=False), encoding="utf-8")


def backup_once(path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / path.name
    if not target.exists():
        shutil.copy2(path, target)
    return target


def fix_thermostat(text: str, images: List[str]) -> Tuple[str, str]:
    marker = "6 完成所有系统设置编号循环后，显示“完成”。按选择键保存并退出。"
    replacement = f"{marker} {PIC}"
    if text.count(PIC) + 1 == len(images) and "Manual36_40" in images and marker in text and replacement not in text:
        return text.replace(marker, replacement, 1), "insert_missing_pic_for_manual36_40_done_screen"
    return text, ""


def fix_generator(text: str, images: List[str]) -> Tuple[str, str]:
    marker = "5. 发电机存放、搬运及运行时必须保持直立。"
    target = f"{marker}\n{PIC}\nAE00789"
    replacement = f"{marker}\nAE00789"
    if text.count(PIC) == len(images) + 1 and "Manual18_72" in images and target in text:
        return text.replace(target, replacement, 1), "remove_extra_storage_pic_before_specs_and_wiring_diagram"
    return text, ""


def fix_dishwasher(text: str, images: List[str]) -> Tuple[str, str]:
    marker = "向上拉动下层喷淋臂拆下（A、B）。"
    target = f"{marker} # 上层喷淋臂"
    replacement = f"{marker} {PIC} # 上层喷淋臂"
    if text.count(PIC) + 1 == len(images) and "Manual06_24" in images and target in text and replacement not in text:
        return text.replace(target, replacement, 1), "insert_missing_pic_between_lower_and_upper_spray_arm"
    return text, ""


FIXES: Dict[str, Callable[[str, List[str]], Tuple[str, str]]] = {
    "可编程温控器手册.txt": fix_thermostat,
    "发电机手册.txt": fix_generator,
    "洗碗机手册.txt": fix_dishwasher,
}


def clean_manual(path: Path, backup_dir: Path) -> Dict[str, object]:
    text, images = load_record(path)
    before_pic = text.count(PIC)
    before_images = len(images)
    backup_path = backup_once(path, backup_dir)

    fixer = FIXES[path.name]
    cleaned_text, action = fixer(text, images)
    changed = cleaned_text != text
    if changed:
        write_record(path, cleaned_text, images)

    return {
        "manual": path.name,
        "backup": str(backup_path),
        "changed": changed,
        "action": action,
        "before_pic": before_pic,
        "before_images": before_images,
        "after_pic": cleaned_text.count(PIC),
        "after_images": len(images),
        "after_diff_image_minus_pic": len(images) - cleaned_text.count(PIC),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-dir", default="data/手册")
    parser.add_argument("--backup-dir", default="data/original_manuals_before_pic_alignment_cleaning")
    parser.add_argument("--report", default="eval_reports/manual_pic_alignment_cleaning_report.json")
    args = parser.parse_args()

    manual_dir = Path(args.manual_dir)
    backup_dir = Path(args.backup_dir)
    results = []
    for name in FIXES:
        results.append(clean_manual(manual_dir / name, backup_dir))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
