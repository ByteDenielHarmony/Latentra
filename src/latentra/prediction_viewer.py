"""Explore saved fixed-budget classifier results without retraining."""

import argparse

import matplotlib
import pandas as pd
from matplotlib.widgets import Button

from . import config as cfg

LABELS = {"baseline": "Percentile baseline", "logistic": "Logistic regression",
          "tree": "Decision tree", "forest": "Random forest"}
SPLITS = ("train", "validation", "test")


class ComparisonView:
    def __init__(self, fig, results):
        self.fig = fig
        self.results = results.set_index(["split", "model"])
        if not self.results.index.is_unique:
            raise ValueError("Expected one comparison row per split and model.")
        self.split = "test"
        self.model = "forest"
        self.bars_ax, self.matrix_ax = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.4, 1]})
        fig.subplots_adjust(left=.17, right=.97, bottom=.22, top=.76, wspace=.30)
        self.heading = fig.suptitle("", fontsize=16)
        self.caption = fig.text(.52, .85, "", ha="center")
        self.buttons = []
        self.draw()

    def draw(self):
        rows = self.results.loc[self.split]
        row = rows.loc[self.model]
        self.heading.set_text(f"Remeasurement results · {self.split.title()}")
        self.caption.set_text(f"{row.rechecks:,} rechecks  ·  {row.high_readings:,} high readings")

        ax = self.bars_ax
        ax.clear()
        self.bars = ax.barh([LABELS[m] for m in rows.index], rows.tp,
                            color=["#186b79" if m == self.model else "#a4bbc4" for m in rows.index])
        self.models = list(rows.index)
        ax.invert_yaxis()
        ax.bar_label(self.bars, labels=[f"{v:,}" for v in rows.tp], padding=4)
        ax.set(xlabel="High readings caught", xlim=(0, rows.tp.max() * 1.22))

        ax = self.matrix_ax
        ax.clear()
        counts = ((row.tn, row.fp), (row.fn, row.tp))
        ax.imshow(counts, cmap="Blues", vmin=0)
        ax.set(xticks=[0, 1], xticklabels=["Skip", "Recheck"],
               yticks=[0, 1], yticklabels=["Not high", "High"],
               xlabel="Decision", ylabel="Observed", title=LABELS[self.model])
        for i, line in enumerate(counts):
            for j, n in enumerate(line):
                ax.text(j, i, f"{n:,}", ha="center", va="center", fontsize=14,
                        color="white" if n > max(row.tn, row.fp, row.fn, row.tp) * .65 else "black")
        self.fig.canvas.draw_idle()

    def select_split(self, split):
        self.split = split
        self.draw()

    def on_click(self, event):
        if event.inaxes is self.bars_ax:
            for bar, model in zip(self.bars, self.models):
                if bar.contains(event)[0]:
                    self.model = model
                    self.draw()
                    break

    def enable_controls(self):
        for i, split in enumerate(SPLITS):
            button = Button(self.fig.add_axes([.16 + i * .12, .04, .11, .06]), split.title())
            button.on_clicked(lambda _, name=split: self.select_split(name))
            self.buttons.append(button)
        self.fig.text(.60, .07, "Click a bar to inspect a method.", va="center")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)


def view_results(*, show=True):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = cfg.REPORTS / "tables/classifier_comparison.csv"
    if not source.is_file():
        raise SystemExit("Run `uv run python -m latentra.prediction` to generate the comparison first.")
    fig = plt.figure(figsize=(11, 5.4))
    view = ComparisonView(fig, pd.read_csv(source))
    output = cfg.REPORTS / "figures/classifier_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")
    if show and fig.canvas.required_interactive_framework:
        view.enable_controls()
        plt.show(block=True)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="export the PNG without opening a window")
    view_results(show=not parser.parse_args().no_show)
