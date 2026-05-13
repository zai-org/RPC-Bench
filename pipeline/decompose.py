import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from prompt import decompose_prompt as QA_PROMPT


BASE_DIR = Path(__file__).resolve().parent
EXAMPLE_FILE = BASE_DIR / "example.json"
OUTPUT_FILE = BASE_DIR / "output" / "example_decompose.jsonl"
RAW_OUTPUT_FILE = BASE_DIR / "output" / "example_decompose_raw.jsonl"
ERROR_OUTPUT_FILE = BASE_DIR / "output" / "example_decompose_errors.jsonl"
MODEL = os.getenv("GPT_MODEL", "gpt-4o")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
SLEEP_SECONDS = float(os.getenv("GPT_SLEEP_SECONDS", "0"))


def read_example_jsonl() -> Iterable[Dict[str, Any]]:
    with EXAMPLE_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_rebuttal_comments(
    comments: List[Dict[str, Any]],
    review_submitter: str,
) -> Tuple[List[str], List[str]]:
    """Return (author_rebuttals, reviewer_followups) from nested comments.
    """

    author_rebuttals: List[str] = []
    reviewer_followups: List[str] = []

    for comment in comments:
        text = comment.get("comment") or comment.get("reply") or comment.get("rebuttal")
        if text:
            submitter = comment.get("submitter")
            if submitter == "Authors":
                author_rebuttals.append(str(text))
            elif submitter == review_submitter:
                reviewer_followups.append(str(text))

        nested = comment.get("rebuttal_comments") or []
        nested_authors, nested_reviewers = split_rebuttal_comments(nested, review_submitter)
        author_rebuttals.extend(nested_authors)
        reviewer_followups.extend(nested_reviewers)

    return author_rebuttals, reviewer_followups


def build_iclr2024_review_text(review: Dict[str, Any], reviewer_followups: List[str]) -> str:
    """Collect the reviewer-side fields used by the ICLR 2024 example."""

    parts = []
    for key in ["weaknesses", "questions", "limitations", "questions_to_address_in_the_rebuttal"]:
        if review.get(key):
            parts.append(str(review[key]))
    parts.extend(reviewer_followups)
    return "\n".join(parts).strip()


def collect_public_rebuttal(paper: Dict[str, Any]) -> str:
    parts = []
    for item in paper.get("public_rebuttal", []):
        text = item.get("rebuttal") or item.get("reply") or item.get("comment")
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def build_prompt(review_text: str, rebuttal_text: str, extra_rebuttal: str) -> str:
    return f"{QA_PROMPT}\nInput:\nreview: {review_text}\nrebuttal: {rebuttal_text}\nextra_rebuttal: {extra_rebuttal}\nOutput:"


def extract_example_review_rebuttals() -> List[Dict[str, str]]:
    """Extract prompt-ready review-rebuttal records from `example.json`."""

    records: List[Dict[str, str]] = []
    for paper in read_example_jsonl():
        paper_id = str(paper["id"])
        extra_rebuttal = collect_public_rebuttal(paper)
        reviews = list(paper.get("review", [])) + list(paper.get("public_review", []))

        for review in reviews:
            if "submitter" not in review or "rebuttal_comments" not in review:
                continue

            author_rebuttals, reviewer_followups = split_rebuttal_comments(
                review["rebuttal_comments"],
                review_submitter=review["submitter"],
            )
            review_text = build_iclr2024_review_text(review, reviewer_followups)
            rebuttal_text = "\n".join(author_rebuttals).strip()
            if not review_text or not rebuttal_text:
                continue

            records.append(
                {
                    "paper_id": paper_id,
                    "prompt": build_prompt(review_text, rebuttal_text, extra_rebuttal),
                }
            )

    return records


def call_gpt(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=OPENAI_BASE_URL)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def parse_gpt_json(text: str) -> List[Dict[str, Any]]:
    """Parse plain JSON or Markdown-fenced JSON returned by GPT."""

    text = text.strip()
    fenced = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    text = re.sub(r'(?<!\\)\\(?![bfnrtu\\"\'\\])', r"\\\\", text)
    data = json.loads(text, strict=False)
    if isinstance(data, dict):
        data = [data]
    return [
        {
            "question": item.get("question", item.get("Q", "")),
            "answer": item.get("answer", item.get("A", "")),
            "is_multimodal_related": item.get(
                "is_multimodal_related",
                item.get("is_multimodal", False),
            ),
        }
        for item in data
        if isinstance(item, dict)
    ]


def convert_example_to_qa() -> None:
    records = extract_example_review_rebuttals()
    parsed_count = 0
    error_count = 0

    RAW_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with (
        RAW_OUTPUT_FILE.open("w", encoding="utf-8") as raw_writer,
        OUTPUT_FILE.open("w", encoding="utf-8") as output_writer,
        ERROR_OUTPUT_FILE.open("w", encoding="utf-8") as error_writer,
    ):
        for idx, record in enumerate(records, start=1):
            print(f"[{idx}/{len(records)}] paper={record['paper_id']}")
            gpt_output = call_gpt(record["prompt"])
            raw_writer.write(
                json.dumps(
                    {
                        "source_call_index": idx,
                        "paper_id": record["paper_id"],
                        "llm_output": gpt_output,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            raw_writer.flush()

            try:
                qa_items = parse_gpt_json(gpt_output)
            except Exception as exc:
                error_count += 1
                error_writer.write(
                    json.dumps(
                        {
                            "source_call_index": idx,
                            "paper_id": record["paper_id"],
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                error_writer.flush()
                continue

            for pair_idx, item in enumerate(qa_items, start=1):
                output_writer.write(
                    json.dumps(
                        {
                            "id": record["paper_id"],
                            "source_call_index": idx,
                            **item,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_writer.flush()
                parsed_count += 1

            if SLEEP_SECONDS > 0:
                time.sleep(SLEEP_SECONDS)

    print(f"extracted {len(records)} review-rebuttal records")
    print(f"wrote raw LLM outputs to {RAW_OUTPUT_FILE}")
    print(f"wrote {parsed_count} item-level pairs to {OUTPUT_FILE}")
    if error_count:
        print(f"failed to parse {error_count} LLM outputs; see {ERROR_OUTPUT_FILE}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY before running pipeline/decompose.py.")
    convert_example_to_qa()
