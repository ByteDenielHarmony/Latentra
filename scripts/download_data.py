"""Fetch the July 2022 competition files into the project's data directory."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import kagglehub
from kagglehub.exceptions import UnauthenticatedError

COMPETITION = "tabular-playground-series-jul-2022"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / COMPETITION
FILES = ("data.csv", "sample_submission.csv")
EXPECTED_ROWS = 98_000


def valid_csv(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.reader(stream, strict=True)
            header = [col.lower() for col in next(rows, [])]
            if path.name == "sample_submission.csv":
                if header != ["id", "predicted"]:
                    return False
            elif "id" not in header or len(header) < 2:
                return False
            count = 0
            for row in rows:
                if len(row) != len(header):
                    return False
                count += 1
            return count == EXPECTED_ROWS
    except (csv.Error, UnicodeError):
        return False


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    missing = [name for name in FILES if not valid_csv(DESTINATION / name)]
    if not missing:
        print(f"Dataset already present: {DESTINATION}")
        return

    with TemporaryDirectory(dir=DESTINATION) as staging:
        for name in missing:
            print(f"Downloading {name}...", flush=True)
            try:
                candidate = Path(kagglehub.competition_download(COMPETITION, path=name, output_dir=staging))
            except UnauthenticatedError:
                raise SystemExit(
                    "Kaggle credentials required. Accept the competition rules and "
                    "configure a token as described in README.md, then rerun."
                ) from None
            if not valid_csv(candidate):
                raise SystemExit(f"Downloaded {name} failed CSV validation.")
            candidate.replace(DESTINATION / name)
    print(f"Dataset ready: {DESTINATION}")


if __name__ == "__main__":
    main()
