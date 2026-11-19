"""Read the original CSV and reconstruct the exploratory split by row position."""

import numpy as np
import pandas as pd

from . import config as cfg


def load_measurements() -> pd.DataFrame:
    if not cfg.DATA_FILE.is_file():
        raise FileNotFoundError("Run `uv run python scripts/download_data.py` first.")
    frame = pd.read_csv(cfg.DATA_FILE)
    if set(frame.columns) != {cfg.ROW_ID, *cfg.MEASUREMENTS}:
        raise ValueError("Unexpected competition data schema.")
    if frame[cfg.ROW_ID].isna().any() or not frame[cfg.ROW_ID].is_unique:
        raise ValueError("Split membership requires non-missing, unique row IDs.")
    return frame.set_index(cfg.ROW_ID)


def split_ids(index: pd.Index) -> pd.DataFrame:
    sizes = (cfg.TRAIN_SIZE, cfg.VALIDATION_SIZE, cfg.TEST_SIZE)
    if len(index) != sum(sizes) or not index.is_unique:
        raise ValueError("Expected one unique ID per competition observation.")
    order = np.random.default_rng(cfg.SEED).permutation(len(index))
    return pd.DataFrame({
        cfg.ROW_ID: index.take(order),
        "split": np.repeat(["train", "validation", "test"], sizes),
    })


def prediction_inputs(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Keep the observed outcome separate from the measurements available at review time."""
    if rows[cfg.TARGET].isna().any():
        raise ValueError("A missing target measurement cannot label a high reading.")
    return rows.loc[:, cfg.PREDICTORS], rows[cfg.TARGET].gt(cfg.HIGH_ABOVE).rename("high")
