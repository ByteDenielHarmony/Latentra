"""Four-profile consensus with abstention and a decision-tree surrogate."""

import argparse
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from . import config as cfg
from .clustering import ARTIFACTS as PREVIOUS, estimator
from .data import load_measurements, split_ids

ARTIFACTS = cfg.ROOT / "artifacts/milestone-5"
LABELS = [-1, 0, 1, 2, 3]  # -1 is an explicit fifth outcome: Uncertain


def align(km, gm):
    counts = np.zeros((4, 4), dtype=int)
    np.add.at(counts, (km, gm), 1)
    rows, cols = linear_sum_assignment(counts, maximize=True)
    mapping = np.empty(4, dtype=int)
    mapping[rows] = cols
    return counts, mapping


def vote(km, gm):
    return np.where(km == gm, gm, -1)


def fidelity(truth, pred):
    disputed = truth == -1
    uncertain_recall = (pred[disputed] == -1).mean() if disputed.any() else np.nan
    return dict(agreement=(truth == pred).mean(),
                macro_f1=f1_score(truth, pred, labels=LABELS, average="macro", zero_division=0),
                uncertain_recall=uncertain_recall, disputed_assigned=1 - uncertain_recall)


def benchmark(function, x):
    function(x)  # warm-up; include preprocessing, exclude disk I/O and model loading
    times = []
    for _ in range(9):
        start = perf_counter()
        function(x)
        times.append((perf_counter() - start) * 1000)
    return np.median(times)


def analyze():
    if not (PREVIOUS / "models.joblib").is_file():
        raise SystemExit("Run `uv run python -m latentra.clustering --no-show` first.")
    bundle = joblib.load(PREVIOUS / "models.joblib")
    scaler, gm = bundle["scaler"], bundle["models"]["gmm"]
    if gm.n_components != 4:
        raise ValueError("This experiment requires the four-component GMM from Milestone 4.")
    x = load_measurements().loc[:, cfg.PREDICTORS]
    ids = split_ids(x.index)
    if cfg.SPLIT_FILE.exists() and not pd.read_csv(cfg.SPLIT_FILE).equals(ids):
        raise ValueError("Unexpected split membership.")

    def part(name):
        return x.loc[ids.loc[ids.split == name, cfg.ROW_ID]]

    train, val = part("train"), part("validation")
    z = scaler.transform(train)
    km = estimator("kmeans", 4, cfg.SEED).fit(z)
    gm_train = gm.predict(z)
    overlap, mapping = align(km.labels_, gm_train)
    y = vote(mapping[km.labels_], gm_train)

    def voters(raw):
        z = scaler.transform(raw)
        return mapping[km.predict(z)], gm.predict(z)

    val_km, val_gm = voters(val)
    val_vote = vote(val_km, val_gm)
    saved_gm = pd.read_csv(PREVIOUS / "validation.csv", index_col=cfg.ROW_ID)["cluster_gmm"]
    if not np.array_equal(saved_gm.loc[val.index], val_gm):
        raise ValueError("GMM assignments differ from the saved validation projection.")

    trials = []
    best, tree = -1, None
    for depth in (4, 8, 12, 16):
        for leaf in (100, 20):
            model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=cfg.SEED)
            model.fit(train, y)
            scores = fidelity(val_vote, model.predict(val))
            trials.append(dict(max_depth=depth, min_samples_leaf=leaf, **scores))
            if scores["macro_f1"] > best:
                best, tree = scores["macro_f1"], model

    tables = cfg.REPORTS / "tables"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(scaler=scaler, kmeans=km, gmm=gm, mapping=mapping, tree=tree),
                ARTIFACTS / "models.joblib", compress=3)
    tab = pd.DataFrame(overlap, columns=[f"gmm_{i}" for i in range(4)])
    tab["aligned_gmm"] = mapping
    tab.rename_axis("original_kmeans").to_csv(tables / "consensus_overlap.csv")
    pd.DataFrame(trials).assign(selected=lambda t: (t.max_depth == tree.max_depth) &
                                (t.min_samples_leaf == tree.min_samples_leaf)).to_csv(
                                    tables / "surrogate_tuning.csv", index=False)
    pd.DataFrame(dict(reference_gmm=val_gm, cluster_kmeans4=val_km, cluster_consensus=val_vote),
                 index=val.index).to_csv(ARTIFACTS / "validation_votes.csv")
    test = part("test")
    test_km, test_gm = voters(test)
    coverage, comparisons, matrices = [], [], []
    for split, raw, k, g in (("validation", val, val_km, val_gm), ("test", test, test_km, test_gm)):
        truth = vote(k, g)
        for profile in ["all", *range(4)]:
            mask = np.ones(len(g), dtype=bool) if profile == "all" else g == profile
            assigned = (truth[mask] != -1).sum()
            coverage.append(dict(split=split, gmm_profile=profile, rows=mask.sum(), agreed=assigned,
                                 coverage=assigned / mask.sum()))
        for name, pred in (("aligned_kmeans", k), ("tree", tree.predict(raw))):
            comparisons.append(dict(split=split, method=name, **fidelity(truth, pred)))
            cm = confusion_matrix(truth, pred, labels=LABELS)
            for label, row in zip(LABELS, cm):
                matrices.append(dict(split=split, method=name, consensus=label,
                                     **{f"pred_{c}": n for c, n in zip(LABELS, row)}))
    timings = [dict(method=name, rows=len(test), median_ms=benchmark(fn, test))
               for name, fn in (("ensemble", lambda raw: vote(*voters(raw))),
                                ("aligned_kmeans", lambda raw: mapping[km.predict(scaler.transform(raw))]),
                                ("tree", tree.predict))]
    for name, records in (("consensus_coverage", coverage), ("surrogate_fidelity", comparisons),
                          ("surrogate_confusion", matrices), ("consensus_timing", timings)):
        pd.DataFrame(records).to_csv(tables / f"{name}.csv", index=False)
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(pd.DataFrame(timings).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        analyze()
    from .cluster_viewer import view_results
    view_results(show=not args.no_show)
