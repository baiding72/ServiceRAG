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
import requests
import time

# 配置
API_URL = "http://localhost:8000/chat"
API_TOKEN = "kafu_test_token_2024"  # 与 main.py 中 API_TOKEN 一致

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

INPUT_FILE = "./data/question_public.csv"
OUTPUT_FILE = "./data/submission.csv"


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


def main():
    import sys
    # 支持命令行参数指定处理数量，默认全部
    max_count = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 表示全部

    results = []

    print("=" * 50)
    print("🚀 开始批量处理测试集...")
    if max_count > 0:
        print(f"   仅处理前 {max_count} 条（测试模式）")
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

            answer = get_answer(question)

            print(f"   回答: {answer[:150]}...")

            results.append({
                "id": q_id,
                "question": question,
                "ret": answer
            })

            time.sleep(0.1)  # 防止并发过高

    # 写入提交文件
    with open(OUTPUT_FILE, mode='w', encoding='utf-8', newline='') as outfile:
        fieldnames = ['id', 'question', 'ret']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 50)
    print(f"✅ 处理完成！")
    print(f"   总计: {len(results)} 条")
    print(f"   输出: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()
