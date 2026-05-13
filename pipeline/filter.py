import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from prompt import quality_filter_prompt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

FILTER_STAGE = os.getenv("FILTER_STAGE", "").strip().lower()
FILTER_MODEL = os.getenv("FILTER_MODEL", "glm").strip().lower()
GLM_MODEL = os.getenv("GLM_MODEL", "glm-4-plus")
SLEEP_SECONDS = float(os.getenv("FILTER_SLEEP_SECONDS", "1"))

if FILTER_STAGE not in {"decompose", "rewrite"}:
    raise ValueError("Set FILTER_STAGE to either 'decompose' or 'rewrite'.")
if FILTER_MODEL not in {"glm", "deepseek"}:
    raise ValueError("FILTER_MODEL must be either 'glm' or 'deepseek'.")


def read_jsonl(file_path: Path) -> Iterable[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def call_glm(message: str) -> str:
    from zhipuai import ZhipuAI

    api_key = os.getenv("GLM_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GLM_API_KEY or ZHIPUAI_API_KEY before running filter.py.")

    client = ZhipuAI(api_key=api_key)
    response = client.chat.completions.create(
        model=GLM_MODEL,
        messages=[{"role": "user", "content": message}],
        max_tokens=4095,
    )
    return response.choices[0].message.content or ""


def extract_json_payload(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    fenced_plain = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced_plain:
        return fenced_plain.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def parse_filter_output(text: str) -> Dict[str, Any]:
    data = json.loads(extract_json_payload(text), strict=False)
    keep = data.get("keep", False)
    if isinstance(keep, str):
        keep = keep.strip().lower() == "true"
    return {
        "keep": bool(keep),
        "reason": data.get("reason", ""),
        "filter_type": data.get("filter_type", "none" if keep else ""),
    }


def build_filter_message(stage: str, item: Dict[str, Any]) -> str:
    return (
        f"{quality_filter_prompt}\n\n"
        f"Input item:\n"
        f"{json.dumps({'stage': stage, **item}, ensure_ascii=False)}\n"
    )


def filter_decompose_stage() -> None:
    input_file = OUTPUT_DIR / "example_decompose.jsonl"
    output_file = OUTPUT_DIR / "example_decompose_filtered.jsonl"
    raw_file = OUTPUT_DIR / "example_decompose_filter_raw.jsonl"
    error_file = OUTPUT_DIR / "example_decompose_filter_errors.jsonl"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    kept_count = 0
    error_count = 0

    with (
        raw_file.open("w", encoding="utf-8") as raw_writer,
        output_file.open("w", encoding="utf-8") as output_writer,
        error_file.open("w", encoding="utf-8") as error_writer,
    ):
        for row_idx, item in enumerate(read_jsonl(input_file), start=1):
            item_id = item.get("id", "")
            pair_index = row_idx
            review = item.get("question", item.get("review", ""))
            rebuttal = item.get("answer", item.get("rebuttal", ""))
            if not review or not rebuttal:
                continue

            print(f"filter=decompose id={item_id} pair={pair_index}")
            logged_item = dict(item)
            logged_item.pop("paper_id", None)
            logged_item["pair_index"] = pair_index
            message = build_filter_message(
                "comment_response",
                {
                    "review": review,
                    "rebuttal": rebuttal,
                },
            )
            raw_decision = call_glm(message)

            raw_writer.write(
                json.dumps(
                    {
                        "id": item_id,
                        "pair_index": pair_index,
                        "stage": "decompose",
                        "item": logged_item,
                        "filter_output": raw_decision,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            raw_writer.flush()

            try:
                decision = parse_filter_output(raw_decision)
            except Exception as exc:
                error_count += 1
                error_writer.write(
                    json.dumps(
                        {
                            "id": item_id,
                            "pair_index": pair_index,
                            "stage": "decompose",
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                error_writer.flush()
                continue

            if decision["keep"]:
                kept = dict(item)
                kept.pop("paper_id", None)
                kept["pair_index"] = pair_index
                kept["filter_reason"] = decision["reason"]
                kept["filter_type"] = decision["filter_type"]
                output_writer.write(json.dumps(kept, ensure_ascii=False) + "\n")
                output_writer.flush()
                kept_count += 1

            if SLEEP_SECONDS > 0:
                time.sleep(SLEEP_SECONDS)

    print(f"wrote raw filter decisions to {raw_file}")
    print(f"wrote {kept_count} kept item-level pairs to {output_file}")
    if error_count:
        print(f"failed to parse {error_count} filter decisions; see {error_file}")


def filter_rewrite_stage() -> None:
    input_file = OUTPUT_DIR / f"example_rewrite_{FILTER_MODEL}.jsonl"
    output_file = OUTPUT_DIR / f"example_rewrite_{FILTER_MODEL}_filtered.jsonl"
    raw_file = OUTPUT_DIR / f"example_rewrite_{FILTER_MODEL}_filter_raw.jsonl"
    error_file = OUTPUT_DIR / f"example_rewrite_{FILTER_MODEL}_filter_errors.jsonl"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    kept_count = 0
    error_count = 0

    with (
        raw_file.open("w", encoding="utf-8") as raw_writer,
        output_file.open("w", encoding="utf-8") as output_writer,
        error_file.open("w", encoding="utf-8") as error_writer,
    ):
        for row_idx, qa_item in enumerate(read_jsonl(input_file), start=1):
            item_id = qa_item.get("id", "")
            pair_index = row_idx
            print(
                f"filter=rewrite model={FILTER_MODEL} "
                f"id={item_id} pair={pair_index}"
            )
            message = build_filter_message("qa_item", qa_item)
            raw_decision = call_glm(message)

            raw_writer.write(
                json.dumps(
                    {
                        "id": item_id,
                        "pair_index": pair_index,
                        "stage": "rewrite",
                        "model": FILTER_MODEL,
                        "item": qa_item,
                        "filter_output": raw_decision,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            raw_writer.flush()

            try:
                decision = parse_filter_output(raw_decision)
            except Exception as exc:
                error_count += 1
                error_writer.write(
                    json.dumps(
                        {
                            "id": item_id,
                            "pair_index": pair_index,
                            "stage": "rewrite",
                            "model": FILTER_MODEL,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                error_writer.flush()
                continue

            if decision["keep"]:
                kept = dict(qa_item)
                kept["filter_reason"] = decision["reason"]
                kept["filter_type"] = decision["filter_type"]
                output_writer.write(json.dumps(kept, ensure_ascii=False) + "\n")
                output_writer.flush()
                kept_count += 1

            if SLEEP_SECONDS > 0:
                time.sleep(SLEEP_SECONDS)

    print(f"wrote raw filter decisions to {raw_file}")
    print(f"wrote {kept_count} kept item-level QA items to {output_file}")
    if error_count:
        print(f"failed to parse {error_count} filter decisions; see {error_file}")


if __name__ == "__main__":
    if FILTER_STAGE == "decompose":
        filter_decompose_stage()
    else:
        filter_rewrite_stage()
