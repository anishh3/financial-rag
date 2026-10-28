#!/usr/bin/env python3
"""Validate normalized Infosys quarterly financial data.

Expected project layout:
financial-retrieval/
  artifacts/normalized/
    q1-2026/records.jsonl
    q2-2026/records.jsonl
    q3-2026/records.jsonl
    additional-information-2025-26/records.jsonl

Run from anywhere:
    python financial-retrieval/src/validate.py

The script checks Q1 + Q2 against the reported six-month figures, Q1 + Q2 +
Q3 against reported nine-month figures, and derives FY2026 Q4 revenue/operating
profit from annual figures minus nine-month figures. Derived figures are clearly
labelled and are not presented as directly reported Q4 figures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_TOLERANCE_CRORE = 1.0  # Allow for rounding in reported ₹ crore figures.


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    # If saved in financial-retrieval/src/validate.py, use the project's own
    # artifacts folder regardless of the current PowerShell directory.
    default_normalized_dir = script_path.parents[1] / "artifacts" / "normalized"

    parser = argparse.ArgumentParser(description="Validate normalized financial records.")
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=default_normalized_dir,
        help=f"Folder containing report subfolders (default: {default_normalized_dir})",
    )
    parser.add_argument(
        "--tolerance-crore",
        type=float,
        default=DEFAULT_TOLERANCE_CRORE,
        help="Allowed absolute difference in ₹ crore for cumulative reconciliation (default: 1)",
    )
    return parser.parse_args()


def parse_number(value: Any) -> float | None:
    """Parse Indian-formatted values such as '1,32,248' into 132248.0."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"-", "—", "–", "n/a", "na", "none", "null"}:
        return None

    # Ignore a trailing percent sign; financial checks below use ₹ crore rows.
    text = text.replace(",", "").replace("₹", "").replace("Rs.", "")
    text = text.replace(" crore", "").replace("INR", "").strip()
    if text.endswith("%"):
        text = text[:-1].strip()

    # Parentheses are commonly used for negative financial values.
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()

    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def load_records(normalized_dir: Path, doc_name: str) -> list[dict[str, Any]]:
    path = normalized_dir / doc_name / "records.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing records file: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_no}: {exc}") from exc
            if isinstance(item, dict):
                records.append(item)
    return records


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def statement_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer the INR-crore statement table, not the parallel USD table."""
    matches = []
    for record in records:
        if record.get("record_type") != "table":
            continue
        if clean_text(record.get("title")) != "statement of comprehensive income":
            continue
        context = " ".join(str(part) for part in (record.get("nearby_context") or []))
        context_norm = clean_text(context)
        rows = record.get("data_rows") or []
        has_inr_rows = any(
            clean_text(row.get("normalized_unit")) == "inr crore"
            for row in rows
            if isinstance(row, dict)
        )
        if has_inr_rows or "₹ crore" in context_norm or "inr crore" in context_norm:
            matches.append(record)
    return matches


def find_row(record: dict[str, Any], label: str) -> dict[str, Any] | None:
    wanted = clean_text(label)
    for row in record.get("data_rows") or []:
        if isinstance(row, dict) and clean_text(row.get("label")) == wanted:
            return row
    return None


def find_period_value(row: dict[str, Any], required_terms: tuple[str, ...]) -> float | None:
    """Find a numeric value whose header contains each requested phrase."""
    values = row.get("values") or {}
    matches: list[tuple[str, Any]] = []
    for header, value in values.items():
        header_norm = clean_text(header)
        if all(clean_text(term) in header_norm for term in required_terms):
            matches.append((str(header), value))

    if len(matches) == 1:
        return parse_number(matches[0][1])
    if len(matches) > 1:
        # Prefer keys explicitly denoting the current calendar year (2025), and
        # avoid selecting growth columns. A remaining ambiguity is an error.
        non_growth = [m for m in matches if "growth" not in clean_text(m[0])]
        if len(non_growth) == 1:
            return parse_number(non_growth[0][1])
        raise ValueError(
            f"Ambiguous period columns for row {row.get('label')!r}: "
            f"{[header for header, _ in matches]}"
        )
    return None


def get_quarter_statement_value(
    records: list[dict[str, Any]],
    metric: str,
    period_terms: tuple[str, ...],
) -> float:
    candidates: list[float] = []
    for record in statement_records(records):
        row = find_row(record, metric)
        if row is None:
            continue
        value = find_period_value(row, period_terms)
        if value is not None:
            candidates.append(value)

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one INR-crore value for {metric!r} and "
            f"period terms {period_terms}; found {candidates}"
        )
    return candidates[0]


def get_cumulative_statement_value(
    records: list[dict[str, Any]],
    metric: str,
    period_terms: tuple[str, ...],
) -> float:
    return get_quarter_statement_value(records, metric, period_terms)


def get_annual_historical_value(
    records: list[dict[str, Any]], metric: str, year: str = "2026"
) -> float:
    matches: list[float] = []
    for record in records:
        if record.get("record_type") != "table":
            continue
        if clean_text(record.get("title")) != "historical data":
            continue
        row = find_row(record, metric)
        if row is None:
            continue
        value = parse_number((row.get("values") or {}).get(year))
        if value is not None:
            matches.append(value)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one FY{year} Historical Data value for {metric!r}; "
            f"found {matches}"
        )
    return matches[0]


def fmt(value: float | None) -> str:
    if value is None:
        return "MISSING"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def check_close(label: str, calculated: float, reported: float, tolerance: float) -> bool:
    difference = calculated - reported
    ok = abs(difference) <= tolerance
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] {label}: calculated {fmt(calculated)}; reported {fmt(reported)}; "
        f"difference {fmt(difference)} ₹ crore (tolerance ±{fmt(tolerance)})"
    )
    return ok


def run_validation(normalized_dir: Path, tolerance: float) -> int:
    print("QuarterLens financial validation")
    print(f"Normalized folder: {normalized_dir}")
    print(f"Reconciliation tolerance: ±{fmt(tolerance)} ₹ crore\n")

    docs = {
        "q1": load_records(normalized_dir, "q1-2026"),
        "q2": load_records(normalized_dir, "q2-2026"),
        "q3": load_records(normalized_dir, "q3-2026"),
        "annual": load_records(normalized_dir, "additional-information-2025-26"),
    }

    metric_specs = {
        "Revenues": {
            "q1_period": ("three months ended june 30", "2025"),
            "q2_period": ("three months ended september 30", "2025"),
            "q2_cumulative": ("six months ended september 30", "2025"),
            "q3_period": ("three months ended december 31", "2025"),
            "q3_cumulative": ("nine months ended december 31", "2025"),
        },
        "Operating profit": {
            "q1_period": ("three months ended june 30", "2025"),
            "q2_period": ("three months ended september 30", "2025"),
            "q2_cumulative": ("six months ended september 30", "2025"),
            "q3_period": ("three months ended december 31", "2025"),
            "q3_cumulative": ("nine months ended december 31", "2025"),
        },
    }

    values: dict[str, dict[str, float]] = {}
    errors = 0
    for metric, spec in metric_specs.items():
        try:
            q1 = get_quarter_statement_value(docs["q1"], metric, spec["q1_period"])
            q2 = get_quarter_statement_value(docs["q2"], metric, spec["q2_period"])
            q2_cum = get_cumulative_statement_value(docs["q2"], metric, spec["q2_cumulative"])
            q3 = get_quarter_statement_value(docs["q3"], metric, spec["q3_period"])
            q3_cum = get_cumulative_statement_value(docs["q3"], metric, spec["q3_cumulative"])
            annual = get_annual_historical_value(docs["annual"], metric)
            values[metric] = {
                "q1": q1,
                "q2": q2,
                "q2_cumulative": q2_cum,
                "q3": q3,
                "q3_cumulative": q3_cum,
                "annual": annual,
            }
        except (ValueError, KeyError) as exc:
            errors += 1
            print(f"[ERROR] Could not validate {metric}: {exc}")

    if errors:
        print("\nValidation could not complete because one or more expected values were missing or ambiguous.")
        return 1

    print("SOURCE VALUES (₹ crore)")
    for metric, v in values.items():
        print(
            f"{metric}: Q1={fmt(v['q1'])}; Q2={fmt(v['q2'])}; "
            f"Q2 reported 6M={fmt(v['q2_cumulative'])}; Q3={fmt(v['q3'])}; "
            f"Q3 reported 9M={fmt(v['q3_cumulative'])}; FY2026 annual={fmt(v['annual'])}"
        )

    print("\nCUMULATIVE RECONCILIATION CHECKS")
    all_passed = True
    for metric, v in values.items():
        all_passed &= check_close(
            f"{metric}: Q1 + Q2 vs reported six-month total",
            v["q1"] + v["q2"], v["q2_cumulative"], tolerance
        )
        all_passed &= check_close(
            f"{metric}: Q1 + Q2 + Q3 vs reported nine-month total",
            v["q1"] + v["q2"] + v["q3"], v["q3_cumulative"], tolerance
        )

    print("\nDERIVED Q4 FY2026 VALUES — NOT DIRECTLY REPORTED Q4 FIGURES")
    for metric, v in values.items():
        q4 = v["annual"] - v["q3_cumulative"]
        print(
            f"{metric}: FY2026 annual {fmt(v['annual'])} - nine-month total "
            f"{fmt(v['q3_cumulative'])} = derived Q4 {fmt(q4)} ₹ crore"
        )

    print("\nRESULT:", "ALL RECONCILIATION CHECKS PASSED" if all_passed else "CHECK FAILED — inspect values and source tables")
    return 0 if all_passed else 2


def main() -> int:
    args = parse_args()
    try:
        return run_validation(args.normalized_dir.resolve(), args.tolerance_crore)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
