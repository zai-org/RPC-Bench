from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import choix
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
RATING_RE_TEMPLATE = r'"{key}"\s*:\s*\{{[^}}]*?"rating"\s*:\s*"?([-+]?\d+(?:\.\d+)?)"?'
SCRIPT_DIR = Path(__file__).resolve().parent


def escape_latex(json_str: str) -> str:
    """Escape stray backslashes before JSON parsing."""
    return re.sub(r'(?<!\\)\\(?![bfnrtu\\"\'\\])', r"\\\\", json_str)


def _strip_json_block(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = JSON_BLOCK_RE.search(text)
    return match.group(1) if match else text


def _extract_rating(text: str, key: str) -> float | None:
    """Extract one rating value from a score string."""
    if not text:
        return None

    text = escape_latex(_strip_json_block(text))

    try:
        data = json.loads(text)
        block = data.get(key)
        if isinstance(block, dict) and "rating" in block:
            return float(block["rating"])
    except Exception:
        pass

    pattern = RATING_RE_TEMPLATE.format(key=re.escape(key))
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return float(match.group(1))
    return None


def _extract_pair_ratings(content) -> Tuple[float | None, float | None]:
    correctness = None
    completeness = None
    parts = content if isinstance(content, list) else [content]

    for part in parts:
        if not part:
            continue
        c_value = _extract_rating(part, "Correctness")
        if c_value is not None:
            correctness = c_value
        c_value = _extract_rating(part, "Completeness")
        if c_value is not None:
            completeness = c_value

    return correctness, completeness


def load_score_file(file_path: str) -> Dict[str, Dict[str, float]]:
    """Load one automatic score file and return {model: {item_id: f1}}."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    model_name = Path(file_path).name.split("_")[0]
    scores: Dict[str, Dict[str, float]] = {model_name: {}}

    for item in data:
        uid = f"{item['id']}_{item['part_idx']}"
        if item.get("predicted_answer", "") == "":
            scores[model_name][uid] = 0.0
            continue

        correctness, completeness = _extract_pair_ratings(item.get("score"))
        if correctness is None or completeness is None:
            f1_score = 0.0
        elif correctness + completeness > 0:
            f1_score = 2 * correctness * completeness / (correctness + completeness)
        else:
            f1_score = 0.0

        scores[model_name][uid] = float(f1_score)

    return scores


def load_pairwise_file(
    file_path: str,
) -> Tuple[List[Tuple[str, str]], List[dict], List[str]]:
    """Load human pairwise labels and return BT pairs, pairwise items and models."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    pairs_bt: List[Tuple[str, str]] = []
    pairwise_items: List[dict] = []
    models = set()

    for item in data:
        a = item["file_a"]
        b = item["file_b"]
        better = item["better"]
        qa_id = f"{item['id']}_{item['part_idx']}"

        models.update([a, b])
        pairwise_items.append(
            {
                "model_a": a,
                "model_b": b,
                "better": better,
                "qa_id": qa_id,
            }
        )

        if str(better).lower() == "tie":
            continue

        if better not in {a, b}:
            continue

        loser = b if better == a else a
        pairs_bt.append((better, loser))

    return pairs_bt, pairwise_items, sorted(models)


def fit_bt_scores(pairs_bt: Sequence[Tuple[str, str]], models: Sequence[str]) -> pd.Series:
    """Fit Bradley-Terry scores for the human pairwise data."""
    models = list(models)
    if not models:
        return pd.Series(dtype=float)
    if len(models) == 1:
        return pd.Series([0.0], index=models, dtype=float)

    model_to_idx = {model: idx for idx, model in enumerate(models)}
    bt_data = [
        (model_to_idx[winner], model_to_idx[loser])
        for winner, loser in pairs_bt
        if winner in model_to_idx and loser in model_to_idx
    ]
    if not bt_data:
        return pd.Series(0.0, index=models, dtype=float)

    try:
        scores = choix.ilsr_pairwise(len(models), bt_data, alpha=0.01)
    except Exception:
        scores = np.zeros(len(models), dtype=float)

    return pd.Series(scores, index=models, dtype=float)


def _safe_corr(func, x, y) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    try:
        value = func(x, y)[0]
    except Exception:
        return np.nan
    if value is None:
        return np.nan
    value = float(value)
    return value if np.isfinite(value) else np.nan


def pairwise_auc(
    pairwise_items: Sequence[dict],
    model_item_scores: Dict[str, Dict[str, float]],
) -> float:
    """Compute pairwise AUC from item-level score differences."""
    y_true: List[int] = []
    y_score: List[float] = []

    for item in pairwise_items:
        if str(item["better"]).lower() == "tie":
            continue

        model_a = item["model_a"]
        model_b = item["model_b"]
        qa_id = item["qa_id"]
        scores_a = model_item_scores.get(model_a, {})
        scores_b = model_item_scores.get(model_b, {})
        if qa_id not in scores_a or qa_id not in scores_b:
            continue

        y_true.append(1 if item["better"] == model_a else 0)
        y_score.append(scores_a[qa_id] - scores_b[qa_id])

    if len(y_true) == 0 or len(set(y_true)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def _normalized_corr(value: float) -> float:
    if value is None or not np.isfinite(value):
        return np.nan
    return (value + 1.0) / 2.0


def compare_versions(
    score_files_list: Dict[str, List[str]],
    pairwise_file: str,
) -> Tuple[pd.DataFrame, pd.Series | None]:
    """Compare settings with Pearson_BT, Spearman_BT, Pairwise_auc and avg."""
    pairs_bt, pairwise_items, models = load_pairwise_file(pairwise_file)
    human_bt_scores = fit_bt_scores(pairs_bt, models)

    results = []

    for setting_name, files in score_files_list.items():
        model_item_scores: Dict[str, Dict[str, float]] = {}
        for file_path in files:
            model_item_scores.update(load_score_file(file_path))

        avg_scores = pd.Series(
            {
                model: float(np.mean(list(scores.values())))
                for model, scores in model_item_scores.items()
                if scores
            },
            dtype=float,
        )

        common_models = [m for m in models if m in avg_scores.index and m in human_bt_scores.index]
        if len(common_models) >= 2:
            auto_vals = avg_scores.loc[common_models].to_numpy(dtype=float)
            human_vals = human_bt_scores.loc[common_models].to_numpy(dtype=float)
            pearson_bt = _safe_corr(pearsonr, auto_vals, human_vals)
            spearman_bt_raw = _safe_corr(spearmanr, auto_vals, human_vals)
        else:
            pearson_bt = np.nan
            spearman_bt_raw = np.nan

        pairwise_auc_score = pairwise_auc(pairwise_items, model_item_scores)

        pearson_bt = _normalized_corr(pearson_bt)
        spearman_bt = _normalized_corr(spearman_bt_raw)

        normalized_scores = [pearson_bt, spearman_bt, pairwise_auc_score]
        normalized_scores = [value for value in normalized_scores if np.isfinite(value)]
        avg_score = float(np.mean(normalized_scores)) if normalized_scores else np.nan

        results.append(
            {
                "setting": setting_name,
                "Pearson_BT": pearson_bt,
                "Spearman_BT": spearman_bt,
                "Pairwise_auc": pairwise_auc_score,
                "avg": avg_score,
            }
        )

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df[["setting", "Pearson_BT", "Spearman_BT", "Pairwise_auc", "avg"]]
        results_df[["Pearson_BT", "Spearman_BT", "Pairwise_auc", "avg"]] = results_df[
            ["Pearson_BT", "Spearman_BT", "Pairwise_auc", "avg"]
        ].round(4)

    if results_df.empty:
        return results_df, None

    if results_df["avg"].notna().any():
        best_setting_row = results_df.loc[results_df["avg"].idxmax()]
    else:
        best_setting_row = results_df.iloc[0]

    return results_df, best_setting_row


def build_score_files_list(base_dir: str) -> Dict[str, List[str]]:
    """Collect score files under each setting directory."""
    score_files_list: Dict[str, List[str]] = {}

    base_path = Path(base_dir)
    if not base_path.is_dir():
        return score_files_list

    for setting_path in sorted(base_path.iterdir()):
        if not setting_path.is_dir():
            continue

        json_files = sorted(str(path) for path in setting_path.glob("*.json"))
        if json_files:
            score_files_list[setting_path.name] = json_files

    return score_files_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute consistency metrics.")
    parser.add_argument(
        "--score-dir",
        default=str(SCRIPT_DIR),
        help="Directory with setting subfolders that contain score JSON files.",
    )
    parser.add_argument(
        "--pairwise-file",
        default=str(SCRIPT_DIR / "pairwise_demo.json"),
        help="Human pairwise file in line-delimited JSON format.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV path for saving the result table.",
    )
    args = parser.parse_args()

    score_files_list = build_score_files_list(args.score_dir)
    if not score_files_list:
        raise SystemExit(f"No score files found under: {args.score_dir}")
    if not Path(args.pairwise_file).is_file():
        raise SystemExit(f"Pairwise file not found: {args.pairwise_file}")

    results_df, best_setting_row = compare_versions(score_files_list, args.pairwise_file)

    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if best_setting_row is not None:
        print("\nBest setting:")
        for key, value in best_setting_row.items():
            if isinstance(value, (float, np.floating)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    if args.output_csv:
        results_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig", float_format="%.4f")


if __name__ == "__main__":
    main()
