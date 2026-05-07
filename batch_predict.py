"""
批量推理脚本

功能：
1. 读取 question_public.csv 测试集
2. 批量调用本地 FastAPI 服务
3. 生成提交文件

使用前请确保：
- FastAPI 服务正在运行 (uvicorn main:app --port 8000)
"""

import csv
import json
import requests
import time
from datetime import datetime
from pathlib import Path
import re
import unicodedata
import argparse

# 配置
API_URL = "http://localhost:8000/chat"
API_TOKEN = "kafu_test_token_2024"  # 与 main.py 中 API_TOKEN 一致

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

INPUT_FILE = "./data/question_public.csv"
OUTPUT_FILE = "./data/submission.csv"
EXPERIMENTS_DIR = Path("./experiments")
REQUEST_TIMEOUT_SECONDS = 90
MAX_REQUEST_RETRIES = 2


def get_answer(question: str) -> str:
    """调用 API 获取答案"""
    payload = {
        "question": question,
        "images": [],
        "session_id": ""
    }
    last_error = ""
    for attempt in range(MAX_REQUEST_RETRIES + 1):
        try:
            response = requests.post(API_URL, json=payload, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                res_data = response.json()
                return res_data.get("data", {}).get("answer", "接口未返回有效答案")

            body_preview = response.text[:300].replace("\n", " ").strip()
            last_error = f"Error: {response.status_code} | {body_preview}"
        except Exception as e:
            last_error = f"Request Failed: {str(e)}"

        if attempt < MAX_REQUEST_RETRIES:
            time.sleep(1.5 * (attempt + 1))

    return last_error


def normalize_answer(answer: str) -> str:
    """将答案压平成单行，降低评测平台 CSV 解析失败风险。"""
    if answer is None:
        return ""
    text = re.sub(r"\s+", " ", str(answer)).strip()

    cleaned_chars = []
    for ch in text:
        code = ord(ch)
        category = unicodedata.category(ch)

        if code < 32 and ch not in "\t\n\r":
            continue
        if code == 127:
            continue
        if code > 0xFFFF:
            continue
        if category in {"Cf", "Cs"}:
            continue

        cleaned_chars.append(ch)

    cleaned = "".join(cleaned_chars)
    return re.sub(r"\s+", " ", cleaned).strip()


def validate_submission_file(file_path: Path) -> None:
    """校验提交文件是否为合法 UTF-8 且列头符合要求。"""
    raw = file_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw.decode("utf-8-sig")
    else:
        raw.decode("utf-8")

    with file_path.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.reader(infile)
        header = next(reader, [])
        if header != ["id", "ret"]:
            raise ValueError(f"提交文件表头错误: {header}")


def load_existing_results(run_submission_file: Path) -> list[dict]:
    """加载已有实验结果，用于断点续跑。"""
    if not run_submission_file.exists():
        return []

    with run_submission_file.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        results = []
        for row in reader:
            results.append(
                {
                    "id": row.get("id", ""),
                    "question": row.get("question", ""),
                    "ret": row.get("ret", ""),
                }
            )
        return results


def is_retryable_result(answer: str) -> bool:
    """判断是否需要在续跑时重试该结果。"""
    text = (answer or "").strip()
    return text.startswith("Error:") or text.startswith("Request Failed:")


def write_outputs(
    results: list[dict],
    run_submission_file: Path,
    run_metadata_file: Path,
    started_at: datetime,
    run_name: str,
    max_count: int,
    expected_total: int,
) -> None:
    """写出官方提交文件与实验文件。"""
    with open(run_submission_file, mode='w', encoding='utf-8', newline='') as outfile:
        fieldnames = ['id', 'question', 'ret']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    official_output_written = max_count == 0 and len(results) == expected_total
    if official_output_written:
        with open(OUTPUT_FILE, mode='w', encoding='utf-8-sig', newline='') as outfile:
            fieldnames = ['id', 'ret']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{"id": row["id"], "ret": row["ret"]} for row in results])
        validate_submission_file(Path(OUTPUT_FILE))

    elapsed_seconds = time.time() - started_at.timestamp()
    metadata = {
        "run_name": run_name,
        "started_at": started_at.isoformat(timespec="seconds"),
        "input_file": INPUT_FILE,
        "api_url": API_URL,
        "max_count": max_count,
        "expected_total": expected_total,
        "result_count": len(results),
        "output_file": OUTPUT_FILE,
        "official_output_written": official_output_written,
        "run_output_file": str(run_submission_file),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "is_complete": len(results) == expected_total,
    }

    with open(run_metadata_file, mode='w', encoding='utf-8') as outfile:
        json.dump(metadata, outfile, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="批量调用本地 /chat 接口生成提交文件")
    parser.add_argument("max_count", nargs="?", type=int, default=0, help="仅处理前 N 条，默认 0 表示全部")
    parser.add_argument("--resume", type=str, default="", help="从已有实验目录断点续跑，例如 batch_20260419_012114")
    parser.add_argument("--save-every", type=int, default=10, help="每处理多少条落盘一次")
    args = parser.parse_args()
    max_count = args.max_count

    if args.resume:
        run_name = args.resume
        run_dir = EXPERIMENTS_DIR / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"未找到实验目录: {run_dir}")
        run_metadata_file = run_dir / "meta.json"
        if run_metadata_file.exists():
            metadata = json.loads(run_metadata_file.read_text(encoding="utf-8"))
            started_at = datetime.fromisoformat(metadata["started_at"])
        else:
            started_at = datetime.now()
    else:
        started_at = datetime.now()
        run_name = started_at.strftime("batch_%Y%m%d_%H%M%S")
        run_dir = EXPERIMENTS_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

    run_submission_file = run_dir / "submission.csv"
    run_metadata_file = run_dir / "meta.json"
    results = load_existing_results(run_submission_file)
    processed_ids = {row["id"] for row in results if not is_retryable_result(row["ret"])}
    retry_rows = [row for row in results if is_retryable_result(row["ret"])]
    if retry_rows:
        retry_ids = {row["id"] for row in retry_rows}
        results = [row for row in results if row["id"] not in retry_ids]
    else:
        retry_ids = set()

    print("=" * 50)
    print("🚀 开始批量处理测试集...")
    if max_count > 0:
        print(f"   仅处理前 {max_count} 条（测试模式）")
    print(f"   实验目录: {run_dir}")
    if results:
        print(f"   断点续跑: 已有 {len(results)} 条结果，将跳过这些 ID")
    if retry_ids:
        print(f"   将自动重试 {len(retry_ids)} 条异常结果（Error:/Request Failed）")
    print("=" * 50)

    with open(INPUT_FILE, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

        if max_count > 0:
            rows = rows[:max_count]
        total = len(rows)

        for i, row in enumerate(rows, 1):
            q_id = row.get('id', '')
            question = row.get('question', '')
            if q_id in processed_ids:
                continue

            print(f"\n[{i}/{total}] 处理 ID: {q_id}")
            print(f"   问题: {question[:50]}...")

            answer = normalize_answer(get_answer(question))

            print(f"   回答: {answer[:150]}...")

            results.append({
                "id": q_id,
                "question": question,
                "ret": answer
            })
            processed_ids.add(q_id)

            if args.save_every > 0 and len(results) % args.save_every == 0:
                write_outputs(results, run_submission_file, run_metadata_file, started_at, run_name, max_count, total)

            time.sleep(0.1)  # 防止并发过高

    results.sort(key=lambda row: int(row["id"]))
    write_outputs(results, run_submission_file, run_metadata_file, started_at, run_name, max_count, total)

    print("\n" + "=" * 50)
    print(f"✅ 处理完成！")
    print(f"   总计: {len(results)} 条")
    print(f"   输出: {OUTPUT_FILE}")
    print(f"   实验结果: {run_submission_file}")
    print(f"   实验元数据: {run_metadata_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
