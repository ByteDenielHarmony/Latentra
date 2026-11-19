"""Generate the training-set evidence for the data-understanding reports."""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.ticker import PercentFormatter

from . import config as cfg
from .data import load_measurements, split_ids


def attribute_summary(train: pd.DataFrame) -> pd.DataFrame:
    stats = train.describe().T.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
    stats["count"] = stats["count"].astype(int)
    stats["dtype"] = train.dtypes.astype(str)
    stats["distinct"] = train.nunique()
    stats["missing"] = train.isna().sum()
    spread = stats.q3 - stats.q1
    stats["outside_iqr_fences"] = (
        train.lt(stats.q1 - 1.5 * spread) | train.gt(stats.q3 + 1.5 * spread)
    ).sum()
    return stats.rename_axis("attribute")


def relationships(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson = train.corr()
    spearman = train.corr(method="spearman")
    i, j = np.triu_indices(len(pearson), k=1)
    pairs = pd.DataFrame({
        "attribute_a": pearson.columns[i],
        "attribute_b": pearson.columns[j],
        "pearson": pearson.to_numpy()[i, j],
        "spearman": spearman.to_numpy()[i, j],
    })
    return pearson, pairs.sort_values("pearson", key=abs, ascending=False)


def plot_correlations(matrix: pd.DataFrame, path) -> None:
    fig = Figure(figsize=(8, 7), layout="constrained")
    ax = fig.subplots()
    # Let between-attribute relationships determine the contrast, not self-correlations.
    values = np.ma.array(matrix.to_numpy(), mask=np.eye(len(matrix), dtype=bool))
    limit = max(0.1, np.ceil(np.abs(values).max() * 10) / 10)
    cmap = colormaps["RdBu_r"].with_extremes(bad="#cccccc")
    image = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
    ticks = range(len(matrix))
    ax.set_xticks(ticks, matrix.columns, rotation=90, fontsize=8)
    ax.set_yticks(ticks, matrix.index, fontsize=8)
    ax.set_title(r"$r_{ii}=1$  (↘)", loc="left", color="#555555", fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.85, label="Pearson r")
    fig.savefig(path, dpi=150)


def plot_high_rates(train: pd.DataFrame, path) -> pd.DataFrame:
    # Quartiles of the integer-valued f_08 on the original training split.
    bands = pd.cut(train["f_08"], [-np.inf, 4, 6, 9, np.inf],
                   labels=["≤ 4", "5–6", "7–9", "≥ 10"])
    high = train[cfg.TARGET].gt(cfg.HIGH_ABOVE)
    groups = high.groupby(bands, observed=True).agg(["size", "mean"])
    fig = Figure(figsize=(6.5, 4), layout="constrained")
    ax = fig.subplots()
    bars = ax.bar(groups.index.astype(str), groups["mean"], color="#327490")
    ax.bar_label(bars, labels=[
        f"{rate:.1%}\nn = {count}" for count, rate in groups.itertuples(index=False)
    ], padding=3)
    ax.set_ylim(0, groups["mean"].max() * 1.35)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set(xlabel="f_08", ylabel=f"P({cfg.TARGET} > {cfg.HIGH_ABOVE})")
    fig.savefig(path, dpi=150)
    return groups


def main() -> None:
    frame = load_measurements()
    membership = split_ids(frame.index)
    cfg.SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    membership.to_csv(cfg.SPLIT_FILE, index=False)
    train = frame.loc[membership.loc[membership["split"] == "train", cfg.ROW_ID]]

    tables, figures = cfg.REPORTS / "tables", cfg.REPORTS / "figures"
    for directory in (tables, figures):
        directory.mkdir(parents=True, exist_ok=True)
    attribute_summary(train).to_csv(tables / "attribute_summary.csv")
    matrix, pairs = relationships(train)
    pairs.to_csv(tables / "attribute_relationships.csv", index=False)
    plot_correlations(matrix, figures / "attribute_correlations.png")
    groups = plot_high_rates(train, figures / "f09_high_rate_by_f08.png")

    print(f"{len(frame)} rows; duplicate IDs: {frame.index.duplicated().sum()}")
    print(f"Training: {len(train)} rows; duplicate measurement rows: {train.duplicated().sum()}; "
          f"missing values: {train.isna().sum().sum()}")
    print(f"{cfg.TARGET} training 90th percentile: {train[cfg.TARGET].quantile(.9):g}")
    print(groups.to_string())
    print(f"Reports: {cfg.REPORTS}\nSplit IDs: {cfg.SPLIT_FILE}")


if __name__ == "__main__":
    main()
