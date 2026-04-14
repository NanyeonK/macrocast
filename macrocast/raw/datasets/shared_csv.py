from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_fred_csv(filepath: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    path = Path(filepath)
    raw = pd.read_csv(path, header=None, dtype=str, na_values=["", ".", " "])
    if raw.shape[0] < 3 or raw.shape[1] < 2:
        raise ValueError(f"file does not look like a FRED CSV: {path}")

    first_cell = str(raw.iloc[0, 0]).strip().lower()
    second_cell = str(raw.iloc[1, 0]).strip().lower()

    if first_cell in {"sasdate", "sasqdate"} or second_cell == "transform:":
        header_row = raw.iloc[0].tolist()
        tcodes_row = raw.iloc[1].tolist()
        data_start = 2
    else:
        tcodes_row = raw.iloc[0].tolist()
        header_row = raw.iloc[1].tolist()
        data_start = 2

    columns = [str(x).strip() for x in header_row]
    if not columns or columns[0].lower() not in {"sasdate", "sasqdate"}:
        raise ValueError(f"missing sasdate/sasqdate header row in {path}")

    tcodes: dict[str, int] = {}
    for name, value in zip(columns[1:], tcodes_row[1:], strict=False):
        try:
            tcodes[name] = int(float(str(value)))
        except (TypeError, ValueError):
            tcodes[name] = 1

    data = raw.iloc[data_start:].copy()
    data.columns = columns
    date_col = columns[0]
    data[date_col] = pd.to_datetime(data[date_col], errors="raise")
    data.set_index(date_col, inplace=True)

    for col in columns[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data.index.name = "date"
    data.sort_index(inplace=True)
    return data, tcodes
