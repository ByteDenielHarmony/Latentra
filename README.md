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
