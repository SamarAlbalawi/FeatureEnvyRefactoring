"""Load and validate the manually annotated C# Feature Envy dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_METHOD_COUNT = 220
EXPECTED_POSITIVE_COUNT = 36
REQUIRED_COLUMNS = {
    "Code Snippet ID",
    "Code Smell",
    "Final annotation",
}


def _normalise_column_name(value: Any) -> str:
    """Convert an Excel header value into a stable string name."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _find_header_row(raw: pd.DataFrame) -> int:
    """Find the row containing the method-record column headers."""
    for row_index in range(len(raw)):
        names = {_normalise_column_name(value) for value in raw.iloc[row_index]}
        if REQUIRED_COLUMNS.issubset(names):
            return row_index
    raise ValueError(
        "Could not find a dataset header row containing: "
        + ", ".join(sorted(REQUIRED_COLUMNS))
    )


def _make_unique_column_names(values: list[Any]) -> list[str]:
    """Create non-empty, unique column names from raw Excel headers."""
    names: list[str] = []
    counts: dict[str, int] = {}
    for value in values:
        base = _normalise_column_name(value) or "Unnamed"
        occurrence = counts.get(base, 0)
        counts[base] = occurrence + 1
        names.append(base if occurrence == 0 else f"{base}.{occurrence}")
    return names


def load_dataset(
    path: str | Path,
    *,
    validate_expected_counts: bool = True,
) -> pd.DataFrame:
    """Load method records from the annotated Feature Envy workbook.

    The workbook contains a grouped-label row before the actual headers.
    Rows without a method identifier are excluded, while all source columns
    and annotation/metric values are preserved.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    raw = pd.read_excel(dataset_path, header=None, engine="openpyxl")
    header_row = _find_header_row(raw)
    columns = _make_unique_column_names(raw.iloc[header_row].tolist())
    records = raw.iloc[header_row + 1 :].copy()
    records.columns = columns
    records = records[records["Code Snippet ID"].notna()].reset_index(drop=True)

    records["Final annotation"] = pd.to_numeric(
        records["Final annotation"], errors="raise"
    )
    method_ids = records["Code Snippet ID"].astype(str).str.strip()
    if method_ids.eq("").any():
        raise ValueError("Dataset contains an empty method identifier.")
    if method_ids.duplicated().any():
        raise ValueError("Dataset contains duplicate method identifiers.")

    positive_count = int((records["Final annotation"] > 0).sum())
    if validate_expected_counts:
        if len(records) != EXPECTED_METHOD_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_METHOD_COUNT} method records, found {len(records)}."
            )
        if positive_count != EXPECTED_POSITIVE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_POSITIVE_COUNT} positive cases, "
                f"found {positive_count}."
            )

    records.attrs["header_row"] = header_row
    records.attrs["positive_count"] = positive_count
    return records


def positive_cases(records: pd.DataFrame) -> pd.DataFrame:
    """Return records whose dataset-derived final severity is positive."""
    return records.loc[records["Final annotation"] > 0].copy()


def _verification_summary(path: Path) -> None:
    records = load_dataset(path)
    positives = positive_cases(records)
    severity_distribution = records["Final annotation"].value_counts().sort_index()

    print(f"TOTAL METHODS: {len(records)}")
    print(f"UNIQUE METHODS: {records['Code Snippet ID'].nunique()}")
    print(f"POSITIVE FEATURE ENVY CASES: {len(positives)}")
    print("SEVERITY DISTRIBUTION:")
    for severity in (0, 1, 2):
        print(f"{severity} = {int(severity_distribution.get(severity, 0))}")
    print("IDENTIFIED COLUMNS:")
    print(", ".join(records.columns))
    print("FIRST POSITIVE CASE:")
    print(positives.iloc[0].to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    _verification_summary(args.dataset)


if __name__ == "__main__":
    main()
