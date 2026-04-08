"""
Convert OCT label abbreviations in labels-cul.xlsx (column B) to full disease names,
and save the result to octlabel.csv.

Mapping:
  AMRD   -> age-related macular degeneration
  CSR    -> central serous retinopathy
  MH     -> macular hole
  DR     -> diabetic retinopathy
  NORMAL -> normal
"""

import csv
from pathlib import Path

try:
    import openpyxl
except ImportError:
    raise ImportError("Please run: pip install openpyxl")

XLSX_PATH   = Path("/root/labels-cul.xlsx")
OUTPUT_PATH = Path("/root/octlabel.csv")

ABBREV_MAP = {
    "amrd":   "age-related macular degeneration",
    "csr":    "central serous retinopathy",
    "mh":     "macular hole",
    "dr":     "diabetic retinopathy",
    "normal": "normal",
}


def convert():
    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)
    ws = wb.active

    rows_out = []
    unknown  = set()

    for row in ws.iter_rows(values_only=True):
        col_a = str(row[0]).strip() if row[0] is not None else ""
        col_b = str(row[1]).strip() if len(row) >= 2 and row[1] is not None else ""

        abbrev = col_b.lower()
        full   = ABBREV_MAP.get(abbrev)

        if col_b and full is None:
            unknown.add(col_b)

        rows_out.append((col_a, full if full is not None else col_b))

    wb.close()

    if unknown:
        print(f"[WARNING] Unrecognised abbreviations (kept as-is): {unknown}")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows_out)

    print(f"Done. {len(rows_out)} rows written -> {OUTPUT_PATH}")


if __name__ == "__main__":
    convert()
