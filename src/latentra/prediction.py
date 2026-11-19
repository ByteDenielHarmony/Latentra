"""Tune, select, and evaluate fixed-budget classifiers for high readings."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from . import config as cfg
from .baseline import review_scores, select_for_review
from .data import load_measurements, prediction_inputs, split_ids


def candidates():
    return {
        "logistic": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
            {"logisticregression__C": [0.01, 0.1, 1.0]},
        ),
        "tree": (
            DecisionTreeClassifier(random_state=cfg.SEED),
            {"max_depth": [4, 8], "min_samples_leaf": [20, 100]},
        ),
        "forest": (
            RandomForestClassifier(n_estimators=96, random_state=cfg.SEED, n_jobs=6),
            {"max_features": ["sqrt", 0.5], "min_samples_leaf": [5, 25]},
        ),
    }


def risk_scores(model, x):
    return pd.Series(model.predict_proba(x)[:, 1], index=x.index)


def budget_recall(model, x, y):
    selected = select_for_review(-risk_scores(model, x))
    return y.loc[selected].sum() / y.sum()


def metrics(risk, y):
    """Evaluate rankings and review decisions; higher scores mean greater risk."""
    selected = select_for_review(-risk)
    tn, fp, fn, tp = confusion_matrix(y, y.index.isin(selected), labels=[False, True]).ravel()
    precision = tp / (tp + fp)
    return dict(rows=len(y), rechecks=len(selected), high_readings=tp + fn,
                tn=tn, fp=fp, fn=fn, tp=tp, accuracy=(tp + tn) / len(y),
                recall=tp / (tp + fn), specificity=tn / (tn + fp),
                precision=precision, lift=precision / y.mean(),
                roc_auc=roc_auc_score(y, risk.loc[y.index]))


def main():
    frame = load_measurements()
    membership = split_ids(frame.index)
    if cfg.SPLIT_FILE.exists() and not pd.read_csv(cfg.SPLIT_FILE).equals(membership):
        raise ValueError("Split IDs differ from Milestone 1; check the original CSV order.")

    def partition(name):
        return prediction_inputs(frame.loc[membership.loc[membership["split"] == name, cfg.ROW_ID]])

    train, y = partition("train")
    tables = cfg.REPORTS / "tables"
    artifacts = cfg.ROOT / "artifacts/milestone-3"
    for path in [tables, artifacts]:
        path.mkdir(parents=True, exist_ok=True)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.SEED)
    models, settings, trials = {}, {}, []
    for name, (model, grid) in candidates().items():
        print(f"Tuning {name}...", flush=True)
        search = GridSearchCV(model, grid, scoring=budget_recall, cv=folds, error_score="raise")
        with threadpool_limits(limits=1, user_api="blas"):
            search.fit(train, y)
        models[name] = search.best_estimator_
        settings[name] = search.best_params_
        cv = search.cv_results_
        for i, params in enumerate(cv["params"]):
            trials.append(dict(model=name, params=json.dumps(params, sort_keys=True),
                               mean_recall=cv["mean_test_score"][i], std_recall=cv["std_test_score"][i],
                               selected=i == search.best_index_))
    pd.DataFrame(trials).to_csv(tables / "classifier_tuning.csv", index=False)

    def evaluate(split, x, high):
        scores = {"baseline": -review_scores(train, x)["mean_percentile"]}
        scores.update({name: risk_scores(model, x) for name, model in models.items()})
        return [dict(split=split, model=name, params=json.dumps(settings.get(name, {})),
                     **metrics(risk, high)) for name, risk in scores.items()]

    rows = evaluate("train", train, y)
    validation = evaluate("validation", *partition("validation"))
    rows.extend(validation)
    # In a tie, prefer the earlier (simpler) rule. Freeze before opening test labels.
    winner = max(validation, key=lambda row: row["tp"])["model"]
    decision = dict(winner=winner, criterion="validation true positives at fixed budget",
                    budget=cfg.REVIEW_BUDGET, settings=settings)
    (artifacts / "selection.json").write_text(json.dumps(decision, indent=2) + "\n")
    if winner != "baseline":
        joblib.dump(models[winner], artifacts / "selected_model.joblib", compress=3)
    rows.extend(evaluate("test", *partition("test")))
    results = pd.DataFrame(rows)
    results.to_csv(tables / "classifier_comparison.csv", index=False)
    print(results[["split", "model", "rechecks", "tp", "recall", "roc_auc"]].to_string(index=False))
    print(f"Selected on validation: {winner}")


if __name__ == "__main__":
    main()
