# Latentra

Exploring simulated manufacturing control measurements from Kaggle’s
[Tabular Playground Series – July 2022](https://www.kaggle.com/competitions/tabular-playground-series-jul-2022/data).

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root:

```sh
uv sync --locked
uv run python --version
```

This creates `.venv` with Python 3.13 and the dependencies in `uv.lock`.

## Competition data

Sign in to Kaggle and accept the [competition rules](https://www.kaggle.com/competitions/tabular-playground-series-jul-2022/rules).
For authenticated access, generate a token in [Kaggle settings](https://www.kaggle.com/settings/api)
and save it outside the repository in `~/.kaggle/access_token`, or set the
`KAGGLE_API_TOKEN` environment variable. Then run:

```sh
uv run python scripts/download_data.py
```

The downloader checks `data/tabular-playground-series-jul-2022/` for `data.csv`
and `sample_submission.csv`. It skips valid files and downloads missing or
incomplete files to temporary storage before replacing them. Both local CSVs
are ignored by Git. Use `data.csv` for analysis; `sample_submission.csv` shows
the expected output format and does not contain observed control states.

## Milestone 1: reproduce the exploratory analysis

Once the CSVs are present, run from the repository root (no token required):

```sh
uv run --locked python -m latentra.explore
```

The command writes two training-set tables to `reports/tables/`, two PNG figures
to `reports/figures/`, and the ordered train/validation/test IDs to the local,
Git-ignored `artifacts/milestone-1/split_ids.csv`. Use the original CSV order to
reproduce the split. The [English report](reports/report.en.md) and its
[Slovak translation](reports/report.sk.md) reference the same outputs.

## Milestone 2: reproduce the review-budget baseline

```sh
uv run --locked python -m latentra.baseline
uv run --locked python -m unittest discover -s tests
```

The first command reconstructs the same split (and verifies the saved IDs if present).
It writes `reports/tables/feature_relevance.csv` from training data and
`reports/tables/baseline_comparison.csv` from validation data. Each rule selects
exactly 1,568 validation IDs for remeasurement; score ties use the same seeded
shuffle of sorted IDs. The final test set is reserved for model evaluation.

## Milestone 3: reproduce the classifier comparison

```sh
uv run --locked python -m latentra.prediction
```

The command tunes logistic regression, a decision tree and a random forest on
the original training rows with three stratified folds. Validation selects the
method; all tuned classifiers and the fixed baseline are then reported on test.
`reports/tables/classifier_tuning.csv` contains the tried parameter settings and
fold-averaged recall at the 10% budget. `classifier_comparison.csv` contains the
train/validation/test metrics and confusion counts (`tn`, `fp`, `fn`, `tp`).
The selection record and winning fitted classifier are saved locally under
`artifacts/milestone-3/`.

After saving the results, training exports `reports/figures/classifier_comparison.png`
and opens a Matplotlib window on graphical desktops. Switch between Train,
Validation and Test with the buttons; click a method's bar to inspect its
confusion matrix. To reopen the viewer using the saved CSV:

```sh
uv run --locked python -m latentra.prediction_viewer
```

Append `--no-show` to either command for export without a window.

## Milestone 4: operating profiles

```sh
uv run --locked python -m latentra.clustering
uv run --locked python -m latentra.cluster_viewer
```

The first command screens K-Means and Gaussian mixtures for 2–10 profiles,
refits the selected models, exports report tables and figures, and opens the
viewer. The second reopens the saved comparison without refitting. Use left/right
arrows or Previous/Next to switch methods in both panels; the 2D PC1–PC2 plot
and rotatable 3D plot show the same points with the same colors. Both commands accept
`--no-show` for headless figure export.

`reports/tables/cluster_screen.csv` records resampling diagnostics, and
`cluster_profiles.csv` describes the final validation groups. Local models, the
scaler, PCA, variance ratios and ID-keyed validation assignments are under
`artifacts/milestone-4/`. The viewer discovers `cluster_*` assignment columns
in `validation.csv`; IDs in the legend belong to the indicated method.
