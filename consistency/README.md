# Consistency Evaluation

This folder is a self-contained demo for computing consistency between LLM judgments and human assessments in a judge-selection setting.

## Metrics

Consistency between model judgments and human assessments is summarized by two
metric families: BT-based correlation and pairwise AUC. BT-based correlation
includes `Pearson_BT` and `Spearman_BT`, which fit a Bradley-Terry model to
convert pairwise outcomes into scores and then correlate those scores with human
preferences using Pearson and Spearman correlation. Pairwise AUC
(`Pairwise_auc`) directly compares model-predicted pairwise preferences with
human labels. The reported `avg` score is the mean of the displayed normalized
metrics.

All displayed metrics are normalized to the `0-1` range and rounded to four decimals.


## Data Roles

```text
consistency_example/
  consistency.py
  pairwise_demo.json
  Model_A/
    alpha_score.json
    beta_score.json
    gamma_score.json
  Model_B/
    alpha_score.json
    beta_score.json
    gamma_score.json
```

`pairwise_demo.json` is the human-labeled pairwise reference data. Each line
records a pairwise comparison for one question and identifies which answer is
preferred between two evaluated response models, such as `alpha` versus `beta`.

- `id`: the paper identifier.
- `part_idx`: the index of the question part within the paper.
- `file_a` / `file_b`: the two evaluated response models being compared.
- `better`: the preferred answer among the two candidates.

Example:

```json
{
  "id": "q1",
  "part_idx": 0,
  "file_a": "alpha",
  "file_b": "beta",
  "better": "alpha"
}
```

`Model_A/` and `Model_B/` store the outputs of two candidate judges. Each folder
contains one score file for each evaluated response model. For example,
`Model_A / alpha_score.json` contains the scores assigned by `Model_A` as a
judge to the answers produced by model `alpha` on the same set of questions.

- `predicted_answer`: the answer produced by the evaluated response model.
- `score`: the judge's rubric output, stored as a JSON-like string.

Example:

```json
{
  "id": "q1",
  "part_idx": 0,
  "predicted_answer": "answer A1",
  "score": "```json\n{\"Correctness\":{\"rating\":\"4.20\"},\"Completeness\":{\"rating\":\"3.80\"},\"Conciseness\":{\"rating\":\"4.00\"}}\n```"
}
```

## Run

```bash
python consistency/consistency.py
```
