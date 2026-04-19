import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


DEFAULT_INPUT = Path("candidates.json")
DEFAULT_OUTPUT = Path("labeled_gold_set.jsonl")

LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "sk-38eea22c4c3745d08a3961661c64f91d")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL_NAME = os.getenv("AUTO_LABEL_MODEL", "qwen-max")

MAX_RETRIES = 3
REQUEST_TIMEOUT = 60.0

SYSTEM_PROMPT = """你是一个严格的 RAG 检索评判专家，只负责判断候选说明书片段对用户问题的相关性。

你会看到：
1. [用户问题]
2. [候选说明书片段]

你的任务是对“单个候选片段”打分，并且只能输出 JSON。

评分标准如下：
- 2（强相关）：该片段直接包含回答问题所需的核心步骤、关键参数、关键定义、直接关联的 <PIC>、或者足以独立回答问题的核心证据。
- 1（部分相关）：该片段与问题属于同一上下文，提供了有价值的背景、补充说明或局部线索，但单凭该片段无法完整回答问题，通常需要配合其他片段。
- 0（完全无关）：该片段对回答问题没有帮助，或只是噪音、错误产品、错误场景、过于泛泛的内容。

严格要求：
1. 只根据给定问题和候选片段打分，不要脑补额外知识。
2. 优先考虑“是否能支撑回答该具体问题”，而不是泛泛相关。
3. 如果问题是步骤题、界面题、按钮题、参数题，请特别看候选是否包含对应的具体步骤、界面名称、按钮功能、参数值。
4. 如果片段只是同一产品的泛背景说明，但没有触及问题核心，打 1，不打 2。
5. 如果片段明显属于错误产品、错误模块或不同场景，打 0。

输出格式必须是纯 JSON，且只能包含以下两个字段：
{"grade": 0, "reason": "一句简短理由"}

不要输出 markdown，不要输出额外解释。"""

USER_PROMPT_TEMPLATE = """[用户问题]
{query}

[候选说明书片段]
Chunk ID: {chunk_id}
Parent ID: {parent_id}
Product: {product}
Sub Manual: {sub_manual}
Section: {section_title}
Language: {language}
Content Type: {content_type}
Text:
{text}

请输出 JSON：{{"grade": 0, "reason": "..."}}"""


def load_candidates(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_labels(path: Path) -> Dict[tuple, dict]:
    if not path.exists():
        return {}

    existing: Dict[tuple, dict] = {}
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (str(record["query_id"]), str(record["chunk_id"]))
            existing[key] = record
    return existing


def parse_json_response(content: str) -> Dict[str, Any]:
    cleaned = (content or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM 返回不是 JSON object")

    grade = int(data["grade"])
    if grade not in {0, 1, 2}:
        raise ValueError(f"非法 grade: {grade}")

    reason = str(data.get("reason", "")).strip()
    return {"grade": grade, "reason": reason}


async def judge_candidate(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    query_id: str,
    query: str,
    candidate: dict,
) -> dict:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        query=query,
        chunk_id=candidate.get("chunk_id", ""),
        parent_id=candidate.get("parent_id", ""),
        product=candidate.get("product", ""),
        sub_manual=candidate.get("sub_manual", ""),
        section_title=candidate.get("section_title", ""),
        language=candidate.get("language", ""),
        content_type=candidate.get("content_type", ""),
        text=candidate.get("text", ""),
    )

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=200,
                    timeout=REQUEST_TIMEOUT,
                )

            content = response.choices[0].message.content or ""
            parsed = parse_json_response(content)
            return {
                "query_id": query_id,
                "chunk_id": str(candidate.get("chunk_id", "")),
                "grade": parsed["grade"],
                "parent_id": str(candidate.get("parent_id", "") or ""),
            }
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            await asyncio.sleep(min(2 ** attempt, 8))

    raise RuntimeError(
        f"自动标注失败 query_id={query_id} chunk_id={candidate.get('chunk_id', '')}: {last_error}"
    )


async def run_labeling(
    candidates: List[dict],
    output_path: Path,
    concurrency: int,
) -> None:
    if not LLM_API_KEY:
        raise RuntimeError("缺少 LLM_API_KEY 或 OPENAI_API_KEY")

    existing = load_existing_labels(output_path)
    semaphore = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=REQUEST_TIMEOUT)

    tasks = []
    for item in candidates:
        query_id = str(item.get("id", ""))
        query = str(item.get("query", ""))
        for candidate in item.get("candidates", []):
            key = (query_id, str(candidate.get("chunk_id", "")))
            if key in existing:
                continue
            tasks.append(judge_candidate(client, semaphore, query_id, query, candidate))

    print(f"[label] pending={len(tasks)} existing={len(existing)} model={LLM_MODEL_NAME} concurrency={concurrency}", flush=True)

    if not tasks:
        print("[label] no pending tasks", flush=True)
        return

    results = await tqdm_asyncio.gather(*tasks)

    with output_path.open("a", encoding="utf-8") as outfile:
        for record in results:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(results)} new labels to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="用 LLM-as-a-Judge 自动标注 retrieval eval 集")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 candidates.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 labeled_gold_set.jsonl")
    parser.add_argument("--concurrency", type=int, default=5, help="并发数")
    args = parser.parse_args()

    candidates = load_candidates(args.input)
    asyncio.run(run_labeling(candidates, args.output, args.concurrency))


if __name__ == "__main__":
    main()
