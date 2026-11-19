"""Compare fixed-budget remeasurement rules on the validation split."""

import numpy as np
import pandas as pd

from . import config as cfg
from .data import load_measurements, prediction_inputs, split_ids


def review_scores(train: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Lower scores receive priority. Percentiles use the training empirical CDF."""
    scores = candidates.loc[:, ["f_08", "f_12"]].copy()
    counts = np.zeros(len(scores), dtype=int)
    for col in scores:
        if train[col].isna().any() or scores[col].isna().any():
            raise ValueError(f"Missing measurement for review rule: {col}")
        counts += np.searchsorted(np.sort(train[col]), scores[col], side="right")
    scores["mean_percentile"] = counts / (2 * len(train))
    return scores


def select_for_review(scores: pd.Series, budget: float = cfg.REVIEW_BUDGET) -> pd.Index:
    """Select floor(budget * rows) IDs, using a common shuffled order only for ties."""
    if not scores.index.is_unique or scores.isna().any() or not 0 <= budget <= 1:
        raise ValueError("Expected unique IDs, complete scores, and a budget in [0, 1].")
    shuffled = np.random.default_rng(cfg.SEED).permutation(scores.index.sort_values())
    ranked = scores.loc[shuffled].sort_values(kind="stable")
    return ranked.index[:int(len(scores) * budget)]


def compare_rules(scores: pd.DataFrame, high: pd.Series) -> pd.DataFrame:
    records = []
    positives = int(high.sum())
    for rule, values in scores.items():
        selected = select_for_review(values)
        caught = int(high.loc[selected].sum())
        precision = caught / len(selected) if len(selected) else np.nan
        records.append({
            "rule": rule, "rows": len(high), "rechecks": len(selected),
            "high_readings": positives, "caught": caught,
            "recall": caught / positives if positives else np.nan,
            "precision": precision, "lift": precision / high.mean() if positives else np.nan,
        })
    return pd.DataFrame(records)


def main() -> None:
    frame = load_measurements()
    membership = split_ids(frame.index)
    if cfg.SPLIT_FILE.exists() and not pd.read_csv(cfg.SPLIT_FILE).equals(membership):
        raise ValueError("Split IDs differ from Milestone 1; check the original CSV order.")
    train, y_train = prediction_inputs(frame.loc[membership.loc[membership["split"] == "train", cfg.ROW_ID]])
    val, y_val = prediction_inputs(frame.loc[membership.loc[membership["split"] == "validation", cfg.ROW_ID]])

    # Spearman correlation is Pearson correlation of the average ranks (ties included).
    relevance = train.rank().corrwith(y_train.rank()).rename("spearman")
    relevance = relevance.sort_values(key=abs, ascending=False).rename_axis("attribute")
    comparison = compare_rules(review_scores(train, val), y_val)
    tables = cfg.REPORTS / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    relevance.to_csv(tables / "feature_relevance.csv")
    comparison.to_csv(tables / "baseline_comparison.csv", index=False)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Tables: {tables}")


if __name__ == "__main__":
    main()
