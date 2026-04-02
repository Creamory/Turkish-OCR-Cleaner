"""
merge_clean.py — Merge cleaned CSV datasets and create a reproducible sample.
"""

from pathlib import Path

import pandas as pd


INPUT_FILES = [
    Path("data/clean50k.csv"),
    Path("data/clean140k.csv"),
    Path("data/cleanheavysiyasi.csv"),
]
OUTPUT_FILE = Path("data/datasetclean.csv")
SAMPLE_FILE = Path("data/cleantest.csv")
CSV_ENCODING = "utf-8-sig"
SAMPLE_SIZE = 3000
RANDOM_STATE = 42


def load_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0, encoding=CSV_ENCODING).columns)


def count_rows(path: Path) -> int:
    return sum(1 for _ in path.open(encoding=CSV_ENCODING, newline="")) - 1


def main() -> None:
    missing_files = [str(path) for path in INPUT_FILES if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing_files)}")

    headers = [load_header(path) for path in INPUT_FILES]
    expected_header = headers[0]
    for path, header in zip(INPUT_FILES[1:], headers[1:]):
        if header != expected_header:
            raise ValueError(
                f"Column mismatch in {path}. Expected {expected_header}, found {header}"
            )

    row_counts = {path: count_rows(path) for path in INPUT_FILES}
    expected_total = sum(row_counts.values())

    merged = pd.concat(
        [pd.read_csv(path, encoding=CSV_ENCODING) for path in INPUT_FILES],
        ignore_index=True,
    )
    merged.to_csv(OUTPUT_FILE, index=False, encoding=CSV_ENCODING)

    merged_count = len(merged)
    if merged_count != expected_total:
        raise ValueError(
            f"Merged row count mismatch. Expected {expected_total}, found {merged_count}"
        )

    if merged_count < SAMPLE_SIZE:
        raise ValueError(
            f"Merged dataset has {merged_count} rows, fewer than sample size {SAMPLE_SIZE}"
        )

    sample = merged.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    sample.to_csv(SAMPLE_FILE, index=False, encoding=CSV_ENCODING)

    sample_check = pd.read_csv(SAMPLE_FILE, encoding=CSV_ENCODING)
    if len(sample_check) != SAMPLE_SIZE:
        raise ValueError(
            f"Sample row count mismatch. Expected {SAMPLE_SIZE}, found {len(sample_check)}"
        )
    if list(sample_check.columns) != list(merged.columns):
        raise ValueError("Sample columns do not match merged dataset columns")

    print(f"Merged {expected_total} rows into {OUTPUT_FILE}")
    print(f"Wrote {SAMPLE_SIZE} sampled rows to {SAMPLE_FILE}")


if __name__ == "__main__":
    main()
