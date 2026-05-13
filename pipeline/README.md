# OpenReview Review-Rebuttal QA Pipeline

This directory provides a simple example of PRC-Bench data processing. It demonstrates how to convert crawled OpenReview review-rebuttal data into QA items through LLM-based decomposition, rewriting, and
filtering.

The example contains four stages:

1. Decompose broad review/rebuttal discussions into comment-response pairs.
2. Filter low-quality comment-response pairs with GLM-4-Plus.
3. Rewrite each pair into taxonomy-based QA with either GLM or DeepSeek.
4. Filter rewritten QA items with GLM-4-Plus.

## Files

- `example.json`: crawled OpenReview example data in JSONL format.
- `prompt.py`: prompts for decomposition, rewriting, and quality filtering.
- `decompose.py`: extracts review/rebuttal fields and calls a GPT model for decomposition.
- `rewrite.py`: rewrites decomposed pairs with one selected GLM or DeepSeek model.
- `filter.py`: filters decomposition-stage pairs or rewrite-stage QA items with a GLM model.
- `output/`: generated intermediate and final JSONL files.

## Stage 1: Decompose Review-Rebuttal

Run:

```bash
OPENAI_API_KEY=your_key python pipeline/decompose.py
```

Optional variables:

```bash
GPT_MODEL=gpt-4o
OPENAI_BASE_URL=https://...
GPT_SLEEP_SECONDS=0
```

Output:

```text
pipeline/output/example_decompose.jsonl
pipeline/output/example_decompose_raw.jsonl
pipeline/output/example_decompose_errors.jsonl
```

## Stage 2: Filter Decomposed Pairs

Run:

```bash
FILTER_STAGE=decompose GLM_API_KEY=your_glm_key python pipeline/filter.py
```

Output:

```text
pipeline/output/example_decompose_filtered.jsonl
pipeline/output/example_decompose_filter_raw.jsonl
pipeline/output/example_decompose_filter_errors.jsonl
```

## Stage 3: Rewrite With GLM Or DeepSeek

Run GLM:

```bash
REWRITE_MODEL=glm GLM_API_KEY=your_glm_key python pipeline/rewrite.py
```

Outputs:

```text
pipeline/output/example_rewrite_glm.jsonl
pipeline/output/example_rewrite_glm_raw.jsonl
pipeline/output/example_rewrite_glm_errors.jsonl
```

Run DeepSeek:

```bash
REWRITE_MODEL=deepseek DEEPSEEK_API_KEY=your_deepseek_key python pipeline/rewrite.py
```

Outputs:

```text
pipeline/output/example_rewrite_deepseek.jsonl
pipeline/output/example_rewrite_deepseek_raw.jsonl
pipeline/output/example_rewrite_deepseek_errors.jsonl
```

Optional variables:

```bash
GLM_MODEL=glm-4-plus
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
REWRITE_SLEEP_SECONDS=1
```

## Stage 4: Filter Rewritten QA Items

Filter GLM rewrites:

```bash
FILTER_STAGE=rewrite FILTER_MODEL=glm GLM_API_KEY=your_glm_key python pipeline/filter.py
```

Outputs:

```text
pipeline/output/example_rewrite_glm_filtered.jsonl
pipeline/output/example_rewrite_glm_filter_raw.jsonl
pipeline/output/example_rewrite_glm_filter_errors.jsonl
```

Filter DeepSeek rewrites:

```bash
FILTER_STAGE=rewrite FILTER_MODEL=deepseek GLM_API_KEY=your_glm_key python pipeline/filter.py
```

Outputs:

```text
pipeline/output/example_rewrite_deepseek_filtered.jsonl
pipeline/output/example_rewrite_deepseek_filter_raw.jsonl
pipeline/output/example_rewrite_deepseek_filter_errors.jsonl
```

Optional variables:

```bash
GLM_MODEL=glm-4-plus
FILTER_SLEEP_SECONDS=1
```

## Output Conventions

- `*_raw.jsonl` files preserve original model outputs and are useful when JSON parsing fails.
- `*_filter_raw.jsonl` files preserve original GLM filter responses.
- `*_errors.jsonl` and `*_filter_errors.jsonl` files record parsing failures without stopping the whole run.
- `*_filtered.jsonl` files contain only item-level rows whose filter decision is `keep=true`.

