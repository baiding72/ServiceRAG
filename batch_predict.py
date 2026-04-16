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


def get_answer(question: str) -> str:
    """调用 API 获取答案"""
    payload = {
        "question": question,
        "images": [],
        "session_id": ""
    }
    try:
        response = requests.post(API_URL, json=payload, headers=HEADERS, timeout=30)
        if response.status_code == 200:
            res_data = response.json()
            return res_data.get("data", {}).get("answer", "接口未返回有效答案")
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Request Failed: {str(e)}"


def normalize_answer(answer: str) -> str:
    """将答案压平成单行，降低评测平台 CSV 解析失败风险。"""
    if answer is None:
        return ""
    return re.sub(r"\s+", " ", str(answer)).strip()


def main():
    import sys
    # 支持命令行参数指定处理数量，默认全部
    max_count = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 表示全部

    started_at = datetime.now()
    run_name = started_at.strftime("batch_%Y%m%d_%H%M%S")
    run_dir = EXPERIMENTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    run_submission_file = run_dir / "submission.csv"
    run_metadata_file = run_dir / "meta.json"

    results = []

    print("=" * 50)
    print("🚀 开始批量处理测试集...")
    if max_count > 0:
        print(f"   仅处理前 {max_count} 条（测试模式）")
    print(f"   实验目录: {run_dir}")
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

            print(f"\n[{i}/{total}] 处理 ID: {q_id}")
            print(f"   问题: {question[:50]}...")

            answer = normalize_answer(get_answer(question))

            print(f"   回答: {answer[:150]}...")

            results.append({
                "id": q_id,
                "question": question,
                "ret": answer
            })

            time.sleep(0.1)  # 防止并发过高

    # 写入提交文件（兼容官方提交流程）
    with open(OUTPUT_FILE, mode='w', encoding='utf-8', newline='') as outfile:
        fieldnames = ['id', 'ret']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{"id": row["id"], "ret": row["ret"]} for row in results])

    # 同时保留本轮实验结果
    with open(run_submission_file, mode='w', encoding='utf-8', newline='') as outfile:
        fieldnames = ['id', 'question', 'ret']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    elapsed_seconds = time.time() - started_at.timestamp()
    metadata = {
        "run_name": run_name,
        "started_at": started_at.isoformat(timespec="seconds"),
        "input_file": INPUT_FILE,
        "api_url": API_URL,
        "max_count": max_count,
        "result_count": len(results),
        "output_file": OUTPUT_FILE,
        "run_output_file": str(run_submission_file),
        "elapsed_seconds": round(elapsed_seconds, 2),
    }

    with open(run_metadata_file, mode='w', encoding='utf-8') as outfile:
        json.dump(metadata, outfile, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"✅ 处理完成！")
    print(f"   总计: {len(results)} 条")
    print(f"   输出: {OUTPUT_FILE}")
    print(f"   实验结果: {run_submission_file}")
    print(f"   实验元数据: {run_metadata_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
