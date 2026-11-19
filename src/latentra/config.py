"""Working study design; the target remains provisional."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/tabular-playground-series-jul-2022/data.csv"
REPORTS = ROOT / "reports"
SPLIT_FILE = ROOT / "artifacts/milestone-1/split_ids.csv"

ROW_ID = "id"
MEASUREMENTS = tuple(f"f_{i:02}" for i in range(29))
TARGET = "f_09"
HIGH_ABOVE = 16
REVIEW_BUDGET = 0.10
PREDICTORS = tuple(name for name in MEASUREMENTS if name != TARGET)

SEED = 42
TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE = 62_720, 15_680, 19_600
