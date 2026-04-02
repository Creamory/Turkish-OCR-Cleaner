"""
category_counts.py — Print category counts for the merged clean dataset.
"""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/datasetclean.csv")
CSV_ENCODING = "utf-8-sig"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, usecols=["category"], encoding=CSV_ENCODING)
    categories = df["category"].fillna("").astype(str).str.strip()

    total_rows = len(categories)
    non_empty = categories[categories.ne("")]
    empty_rows = int(categories.eq("").sum())
    counts = non_empty.value_counts()

    print(f"Dataset: {INPUT_FILE}")
    print(f"Total rows: {total_rows}")
    print(f"Non-empty category rows: {len(non_empty)}")
    print(f"Empty category rows: {empty_rows}")
    print(f"Distinct non-empty categories: {counts.size}")
    print("")
    print("Category counts:")

    for category, count in counts.items():
        print(f"{category}: {count}")


if __name__ == "__main__":
    main()
