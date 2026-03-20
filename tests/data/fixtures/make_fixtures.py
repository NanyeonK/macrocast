"""Script to generate minimal test fixture files.

Run once: uv run python tests/data/fixtures/make_fixtures.py
"""

from pathlib import Path

import openpyxl
import pandas as pd

HERE = Path(__file__).parent


def make_fred_md_sample() -> None:
    """Create a minimal FRED-MD style CSV fixture."""
    lines = [
        # Row 0: tcode row (first cell blank, then codes)
        ",5,5,2,6\n",
        # Row 1: header (sasdate + variable names)
        "sasdate,INDPRO,RPI,UNRATE,CPIAUCSL\n",
        # Data rows
        "1/1/2000,100.0,1000.0,4.0,170.0\n",
        "2/1/2000,101.0,1010.0,4.1,170.5\n",
        "3/1/2000,102.0,1020.0,4.0,171.0\n",
        "4/1/2000,103.0,1030.0,3.9,171.5\n",
        "5/1/2000,104.0,1040.0,3.8,172.0\n",
        "6/1/2000,105.0,1050.0,3.9,172.0\n",
    ]
    out = HERE / "fred_md_sample.csv"
    with open(out, "w") as fh:
        fh.writelines(lines)
    print(f"Wrote {out}")


def make_fred_qd_sample() -> None:
    """Create a minimal FRED-QD style CSV fixture."""
    lines = [
        ",5,6,2\n",
        "sasqdate,GDPC1,CPIAUCSL,FEDFUNDS\n",
        "1/1/2000,10000.0,170.0,5.5\n",
        "4/1/2000,10100.0,171.0,5.25\n",
        "7/1/2000,10200.0,172.0,5.0\n",
        "10/1/2000,10150.0,172.5,4.75\n",
        "1/1/2001,10050.0,173.0,5.0\n",
        "4/1/2001,9950.0,174.0,4.5\n",
    ]
    out = HERE / "fred_qd_sample.csv"
    with open(out, "w") as fh:
        fh.writelines(lines)
    print(f"Wrote {out}")


def make_fred_sd_sample() -> None:
    """Create a minimal FRED-SD style XLSX fixture."""
    wb = openpyxl.Workbook()

    # Sheet 1: UR (Unemployment Rate)
    ws_ur = wb.active
    ws_ur.title = "UR"
    states = ["CA", "TX", "NY"]
    ws_ur.append([""] + states)
    data = [
        ["2000-01-01", 4.5, 4.2, 5.0],
        ["2000-02-01", 4.6, 4.3, 5.1],
        ["2000-03-01", 4.4, 4.1, 4.9],
        ["2000-04-01", 4.3, 4.0, 4.8],
    ]
    for row in data:
        ws_ur.append(row)

    # Sheet 2: EMPL (Employment)
    ws_empl = wb.create_sheet("EMPL")
    ws_empl.append([""] + states)
    empl_data = [
        ["2000-01-01", 15000, 11000, 9000],
        ["2000-02-01", 15100, 11050, 9020],
        ["2000-03-01", 15200, 11100, 9050],
        ["2000-04-01", 15300, 11150, 9080],
    ]
    for row in empl_data:
        ws_empl.append(row)

    out = HERE / "fred_sd_sample.xlsx"
    wb.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    make_fred_md_sample()
    make_fred_qd_sample()
    make_fred_sd_sample()
