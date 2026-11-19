"""Compare operating profiles by reproducibility on common validation observations."""

import argparse
import json
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import config as cfg
from .data import load_measurements, split_ids

ARTIFACTS = cfg.ROOT / "artifacts/milestone-4"
METHODS = ("kmeans", "gmm")
SEEDS = (cfg.SEED, cfg.SEED + 1, cfg.SEED + 2)


def estimator(method, k, seed):
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=5, random_state=seed)
    return GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-4,
                           max_iter=150, random_state=seed)


def uncertain(probabilities):
    leading = np.sort(probabilities, axis=1)[:, -2:]
    return leading[:, 1] - leading[:, 0] < .1


def screen(train, val):
    subsets = [np.random.default_rng(s).choice(len(train), 8192, replace=False) for s in SEEDS]
    sample = np.random.default_rng(cfg.SEED).choice(len(val), 1800, replace=False)
    distances = pairwise_distances(val[sample])
    np.fill_diagonal(distances, 0)
    records = []
    for method in METHODS:
        for k in range(2, 11):
            labels, separation, min_sizes, ambiguity, converged = [], [], [], [], []
            for seed, subset in zip(SEEDS, subsets):
                model = estimator(method, k, seed).fit(train[subset])
                assigned = model.predict(val)
                labels.append(assigned)
                min_sizes.append(np.bincount(assigned, minlength=k).min() / len(val))
                if len(np.unique(assigned[sample])) > 1:
                    separation.append(silhouette_score(distances, assigned[sample], metric="precomputed"))
                if method == "gmm":
                    ambiguity.append(uncertain(model.predict_proba(val)).mean())
                    converged.append(model.converged_)
                else:
                    converged.append(model.n_iter_ < model.max_iter)
            ari = [adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]
            records.append(dict(method=method, k=k, stability=np.mean(ari), min_ari=min(ari),
                                silhouette=np.mean(separation) if len(separation) == 3 else np.nan,
                                smallest_share=min(min_sizes), converged=all(converged),
                                ambiguous_share=np.mean(ambiguity) if ambiguity else np.nan))
    return pd.DataFrame(records)


def select_count(group):
    eligible = group[group.converged & (group.smallest_share >= .01) & group.silhouette.notna()]
    if eligible.empty:
        raise ValueError(f"No eligible configuration for {group.method.iloc[0]}; inspect the screen.")
    stable = eligible[eligible.stability >= eligible.stability.max() - .05]
    return int(stable.loc[stable.silhouette >= stable.silhouette.max() - .01, "k"].min())


def heatmap(method, centers, limit, path):
    fig = Figure(figsize=(11, max(3.2, len(centers) * .45 + 1.8)), layout="constrained")
    ax = fig.subplots()
    image = ax.imshow(centers, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(centers.columns)), centers.columns, rotation=90)
    ax.set_yticks(range(len(centers)), centers.index)
    ax.set(ylabel="Profile ID", title=method.upper())
    fig.colorbar(image, ax=ax, label="Mean (training SD units)")
    fig.savefig(path, dpi=150)


def analyze():
    frame = load_measurements()
    membership = split_ids(frame.index)
    if cfg.SPLIT_FILE.exists() and not pd.read_csv(cfg.SPLIT_FILE).equals(membership):
        raise ValueError("Split IDs differ from Milestone 1.")
    train_ids, val_ids = [membership.loc[membership.split == part, cfg.ROW_ID] for part in ("train", "validation")]
    scaler = StandardScaler().fit(frame.loc[train_ids, cfg.PREDICTORS])
    train, val = [scaler.transform(frame.loc[ids, cfg.PREDICTORS]) for ids in (train_ids, val_ids)]
    tables, figures = cfg.REPORTS / "tables", cfg.REPORTS / "figures"
    for directory in (tables, figures, ARTIFACTS):
        directory.mkdir(parents=True, exist_ok=True)
    comparison = screen(train, val)
    comparison.to_csv(tables / "cluster_screen.csv", index=False)
    chosen = {m: select_count(comparison[comparison.method == m]) for m in METHODS}
    print("Selected k:", chosen, flush=True)
    comparison["selected"] = comparison.k == comparison.method.map(chosen)
    comparison.to_csv(tables / "cluster_screen.csv", index=False)

    pca = PCA(n_components=3, svd_solver="full").fit(train)
    points = pd.DataFrame(pca.transform(val), index=pd.Index(val_ids), columns=["PC1", "PC2", "PC3"])
    high = frame.loc[val_ids, cfg.TARGET].gt(cfg.HIGH_ABOVE).to_numpy()
    profiles, means, models, diagnostics = [], {}, {}, {}
    for method, k in chosen.items():
        model = estimator(method, k, cfg.SEED).fit(train)
        models[method] = model
        labels = model.predict(val)
        points[f"cluster_{method}"] = labels
        center = pd.DataFrame(val, columns=cfg.PREDICTORS).groupby(labels).mean()
        means[method] = center
        if method == "gmm":
            diagnostics = dict(converged=bool(model.converged_),
                               ambiguous_share=float(uncertain(model.predict_proba(val)).mean()))
        for label, row in center.iterrows():
            mask = labels == label
            top = row.abs().nlargest(3).index
            profiles.append(dict(method=method, cluster=label, rows=mask.sum(), high_count=high[mask].sum(),
                                 high_rate=high[mask].mean(),
                                 distinguishing=", ".join(f"{col} ({row[col]:+.2f})" for col in top)))
    pd.DataFrame(profiles).to_csv(tables / "cluster_profiles.csv", index=False)
    limit = max(np.abs(c.to_numpy()).max() for c in means.values())
    for method, center in means.items():
        heatmap(method, center, limit, figures / f"cluster_profile_{method}.png")
    points.to_csv(ARTIFACTS / "validation.csv")
    metadata = dict(chosen=chosen, variance=pca.explained_variance_ratio_.tolist(), gmm=diagnostics,
                    agreement=adjusted_rand_score(points.cluster_kmeans, points.cluster_gmm))
    (ARTIFACTS / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    joblib.dump(dict(scaler=scaler, pca=pca, models=models), ARTIFACTS / "models.joblib", compress=3)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        analyze()
    from .cluster_viewer import view_results
    view_results(show=not args.no_show)
