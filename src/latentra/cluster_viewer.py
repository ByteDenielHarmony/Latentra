"""Compare cluster assignments in linked 2D and 3D views of one PCA projection."""

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
from scipy.optimize import linear_sum_assignment

from . import config as cfg


def color_slots(labels, reference):
    """Match equal-sized partitions by maximum overlap, keeping the original IDs."""
    ids = sorted(set(labels) - {-1})
    base = sorted(set(reference) - {-1})
    mapping = {label: i for i, label in enumerate(ids)}
    if len(ids) == len(base) and ids:
        counts = pd.crosstab(labels, reference).reindex(index=ids, columns=base, fill_value=0)
        rr, cc = linear_sum_assignment(counts.to_numpy(), maximize=True)
        mapping = {ids[i]: int(j) for i, j in zip(rr, cc)}
    return mapping


class ClusterView:
    def __init__(self, fig, points, variance):
        self.fig = fig
        self.methods = [col for col in points if col.startswith("cluster_")]
        self.slots = {col: color_slots(points[col], points[self.methods[0]]) for col in self.methods}
        # Voter and consensus IDs already use the training-aligned GMM namespace.
        for col in ("cluster_kmeans4", "cluster_consensus"):
            if col in self.methods:
                self.slots[col] = self.slots["cluster_gmm"]
        self.coverage = (points.cluster_consensus != -1).mean() if "cluster_consensus" in points else None
        self.points = points.sample(n=min(2200, len(points)), random_state=cfg.SEED)
        self.plane = fig.add_subplot(121)
        self.ax = fig.add_subplot(122, projection="3d")
        fig.subplots_adjust(left=.08, right=.87, bottom=.17, top=.88, wspace=.2)
        self.flat = self.plane.scatter(self.points.PC1, self.points.PC2, s=8, alpha=.75,
                                       edgecolors="none")
        self.plane.set(xlabel=f"PC1 ({variance[0]:.1%})", ylabel=f"PC2 ({variance[1]:.1%})",
                       title="2D · PC1–PC2", aspect="equal")
        self.ax.set_title("3D · PC1–PC3")
        self.scatter = self.ax.scatter(*[self.points[f"PC{i}"] for i in (1, 2, 3)],
                                       s=8, alpha=.75, depthshade=False, edgecolors="none")
        self.crosses = []
        if self.coverage is not None:
            tied = self.points[self.points.cluster_consensus == -1]
            style = dict(marker="x", c="#888888", s=12, linewidths=.7)
            self.crosses = [self.plane.scatter(tied.PC1, tied.PC2, **style),
                            self.ax.scatter(tied.PC1, tied.PC2, tied.PC3, depthshade=False, **style)]
        for i, axis in enumerate([self.ax.set_xlabel, self.ax.set_ylabel, self.ax.set_zlabel]):
            axis(f"PC{i + 1} ({variance[i]:.1%})")
        self.index = 0
        self.buttons = []
        self.display(0)

    def display(self, index):
        self.index = index % len(self.methods)
        method = self.methods[self.index]
        slots = self.slots[method]
        # Charcoal replaces tab10 green so red and green never distinguish profiles.
        palette = list(matplotlib.colormaps["tab10"].colors)
        palette[2] = "#333333"
        colors = {label: palette[slot % len(palette)] for label, slot in slots.items()}
        consensus = method == "cluster_consensus"
        if consensus:
            colors[-1] = "#888888"
        for artist in self.crosses:
            artist.set_visible(consensus)
        sizes = np.where(self.points[method] == -1, 0, 8) if consensus else [8]
        self.flat.set_sizes(sizes)
        self.scatter.set_sizes(sizes)
        facecolors = [colors[label] for label in self.points[method]]
        self.flat.set_facecolor(facecolors)
        self.scatter.set_facecolor(facecolors)
        name = {"cluster_kmeans": "K-Means", "cluster_gmm": "GMM",
                "cluster_kmeans4": "K-Means (ensemble voter)", "cluster_consensus": "Consensus"
                }.get(method, method.removeprefix("cluster_").upper())
        title = f"{name} · {len(slots)} profiles · n = {len(self.points):,}"
        if consensus:
            title += f" · full validation coverage {self.coverage:.1%}"
        self.fig.suptitle(title)
        handles = [Line2D([], [], linestyle="", marker="x" if label == -1 else "o", color=color,
                          label=f"Profile {label}" if label != -1 else "Uncertain")
                   for label, color in colors.items()]
        self.ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, .5))
        self.fig.canvas.draw_idle()

    def key(self, event):
        if event.key in ("left", "right"):
            self.display(self.index + (1 if event.key == "right" else -1))

    def controls(self):
        for i, (text, step) in enumerate([("Previous", -1), ("Next", 1)]):
            button = Button(self.fig.add_axes([.32 + .17*i, .03, .15, .065]), text)
            button.on_clicked(lambda _, d=step: self.display(self.index + d))
            self.buttons.append(button)
        self.fig.canvas.mpl_connect("key_press_event", self.key)


def view_results(*, show=True):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = cfg.ROOT / "artifacts/milestone-4"
    if not (artifacts / "summary.json").exists():
        raise SystemExit("Run `uv run python -m latentra.clustering` first.")
    points = pd.read_csv(artifacts / "validation.csv", index_col=cfg.ROW_ID)
    extra = cfg.ROOT / "artifacts/milestone-5/validation_votes.csv"
    if extra.is_file():
        votes = pd.read_csv(extra, index_col=cfg.ROW_ID).reindex(points.index)
        if not votes.reference_gmm.equals(points.cluster_gmm):
            raise ValueError("Saved consensus votes do not match this projection; rerun latentra.consensus.")
        points = points.join(votes.drop(columns="reference_gmm"), validate="one_to_one")
    variance = json.loads((artifacts / "summary.json").read_text())["variance"]
    fig = plt.figure(figsize=(12, 5.8))
    view = ClusterView(fig, points, variance)
    for i, method in enumerate(view.methods):
        view.display(i)
        fig.savefig(cfg.REPORTS / "figures" / f"pca_{method}.png", dpi=150)
    if show and fig.canvas.required_interactive_framework:
        view.display(0)
        view.controls()
        plt.show(block=True)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    view_results(show=not parser.parse_args().no_show)
