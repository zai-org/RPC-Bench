import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from prompt import rewrite_prompt


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "output" / "example_decompose.jsonl"

REWRITE_MODEL = os.getenv("REWRITE_MODEL", "glm").strip().lower()
GLM_MODEL = os.getenv("GLM_MODEL", "glm-4-plus")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
SLEEP_SECONDS = float(os.getenv("REWRITE_SLEEP_SECONDS", "1"))

if REWRITE_MODEL not in {"glm", "deepseek"}:
    raise ValueError("REWRITE_MODEL must be either 'glm' or 'deepseek'.")

OUTPUT_FILE = BASE_DIR / "output" / f"example_rewrite_{REWRITE_MODEL}.jsonl"
RAW_OUTPUT_FILE = BASE_DIR / "output" / f"example_rewrite_{REWRITE_MODEL}_raw.jsonl"
ERROR_OUTPUT_FILE = BASE_DIR / "output" / f"example_rewrite_{REWRITE_MODEL}_errors.jsonl"


def read_jsonl(file_path: Path) -> Iterable[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_rewrite_message(review: str, rebuttal: str) -> str:
    return f"{rewrite_prompt}\nInput:\nreview: {review}\nrebuttal: {rebuttal}\nOutput:\n"


def call_glm(message: str) -> str:
    from zhipuai import ZhipuAI

    api_key = os.getenv("GLM_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GLM_API_KEY or ZHIPUAI_API_KEY before running rewrite.py.")

    client = ZhipuAI(api_key=api_key)
    response = client.chat.completions.create(
        model=GLM_MODEL,
        messages=[{"role": "user", "content": message}],
        max_tokens=4095,
    )
    return response.choices[0].message.content or ""


def call_deepseek(message: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("Set DEEPSEEK_API_KEY before running rewrite.py.")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": message}],
        stream=False,
    )
    return response.choices[0].message.content or ""


def call_selected_model(message: str) -> str:
    if REWRITE_MODEL == "glm":
        return call_glm(message)
    return call_deepseek(message)


def extract_json_payload(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    fenced_plain = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced_plain:
        return fenced_plain.group(1).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def escape_latex(json_str: str) -> str:
    return re.sub(r'(?<!\\)\\(?![bfnrtu\\"\'\\])', r"\\\\", json_str)


def parse_rewrite_output(text: str) -> List[Dict[str, Any]]:
    payload = escape_latex(extract_json_payload(text))
    data = json.loads(payload, strict=False)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("rewrite output is not a JSON object or list")

    parsed: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        parsed.append(
            {
                "review": item.get("review", ""),
                "rebuttal": item.get("rebuttal", ""),
                "Q": item.get("Q", item.get("question", "")),
                "A": item.get("A", item.get("answer", "")),
                "Category": item.get("Category", item.get("category", "")),
            }
        )
    return parsed


def rewrite_with_model() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rewritten_count = 0
    error_count = 0

    with (
        RAW_OUTPUT_FILE.open("w", encoding="utf-8") as raw_writer,
        OUTPUT_FILE.open("w", encoding="utf-8") as output_writer,
        ERROR_OUTPUT_FILE.open("w", encoding="utf-8") as error_writer,
    ):
        for idx, pair in enumerate(read_jsonl(INPUT_FILE), start=1):
            paper_id = pair.get("paper_id", pair.get("id", ""))
            pair_index = idx
            review = pair.get("question", pair.get("review", ""))
            rebuttal = pair.get("answer", pair.get("rebuttal", ""))
            if not review or not rebuttal:
                continue

            print(f"model={REWRITE_MODEL} paper={paper_id} pair={pair_index}")
            message = build_rewrite_message(review, rebuttal)
            model_raw = call_selected_model(message)

            raw_writer.write(
                json.dumps(
                    {
                        "id": paper_id,
                        "pair_index": pair_index,
                        "review": review,
                        "rebuttal": rebuttal,
                        "model": REWRITE_MODEL,
                        "model_output": model_raw,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            raw_writer.flush()

            try:
                rewrite_items = parse_rewrite_output(model_raw)
            except Exception as exc:
                error_count += 1
                error_writer.write(
                    json.dumps(
                        {
                            "id": paper_id,
                            "pair_index": pair_index,
                            "model": REWRITE_MODEL,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                error_writer.flush()
                continue

            for rewrite_item in rewrite_items:
                output_writer.write(
                    json.dumps(
                        {
                            "id": paper_id,
                            "pair_index": pair_index,
                            "model": REWRITE_MODEL,
                            "review": rewrite_item.get("review") or review,
                            "rebuttal": rewrite_item.get("rebuttal") or rebuttal,
                            "Q": rewrite_item.get("Q", ""),
                            "A": rewrite_item.get("A", ""),
                            "Category": rewrite_item.get("Category", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_writer.flush()
                rewritten_count += 1

            if SLEEP_SECONDS > 0:
                time.sleep(SLEEP_SECONDS)

    print(f"wrote raw model outputs to {RAW_OUTPUT_FILE}")
    print(f"wrote {rewritten_count} item-level QA items to {OUTPUT_FILE}")
    if error_count:
        print(f"failed to parse {error_count} model outputs; see {ERROR_OUTPUT_FILE}")


if __name__ == "__main__":
    rewrite_with_model()
