from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def clean_text(value: Any) -> str:
    """Normalize whitespace without altering the source text stored elsewhere."""
    if not isinstance(value, str):
        return ""
    value = value.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value).strip()


def get_ref(value: Any) -> str | None:
    """Read a Docling reference, e.g. {'$ref': '#/texts/12'}."""
    if isinstance(value, dict):
        value = value.get("$ref")
    return value if isinstance(value, str) else None


def stable_id(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def get_pages(item: dict) -> list[int]:
    pages: set[int] = set()
    for provenance in item.get("prov") or []:
        if not isinstance(provenance, dict):
            continue
        page = provenance.get("page_no")
        if page is not None:
            try:
                pages.add(int(page))
            except (TypeError, ValueError):
                pass
    return sorted(pages)


def get_bbox(item: dict) -> dict | None:
    provenance = item.get("prov") or []
    if provenance and isinstance(provenance[0], dict):
        return provenance[0].get("bbox")
    return None


def index_objects(document: dict) -> dict[str, dict]:
    """Index Docling objects so references can be resolved efficiently."""
    indexed: dict[str, dict] = {}
    for collection in ("texts", "tables", "pictures", "groups"):
        for item in document.get(collection) or []:
            self_ref = item.get("self_ref")
            if self_ref:
                indexed[self_ref] = item
    return indexed


def resolve_text(reference: Any, objects: dict[str, dict]) -> str:
    key = get_ref(reference)
    if key:
        return clean_text(objects.get(key, {}).get("text", ""))
    return clean_text(reference) if isinstance(reference, str) else ""


def get_caption(item: dict, objects: dict[str, dict]) -> str:
    captions = [resolve_text(ref, objects) for ref in item.get("captions") or []]
    return " ".join(text for text in captions if text)


PERIOD_START_RE = re.compile(
    r"^(?:quarter ended|year ended|three months ended|six months ended|"
    r"nine months ended|period ended)\b",
    re.IGNORECASE,
)


def bbox_numbers(item: dict) -> tuple[float, float, float, float] | None:
    """Return (left, right, top, bottom) when usable provenance exists."""
    bbox = get_bbox(item)
    if not isinstance(bbox, dict):
        return None
    try:
        return (
            float(bbox["l"]),
            float(bbox["r"]),
            float(bbox["t"]),
            float(bbox["b"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def bbox_center_y(item: dict) -> float | None:
    coords = bbox_numbers(item)
    if coords is None:
        return None
    _, _, top, bottom = coords
    return (top + bottom) / 2.0


def is_period_label(text: str) -> bool:
    """Recognize source labels that describe a reporting period."""
    return bool(PERIOD_START_RE.match(clean_text(text)))


def parse_numeric_label(text: str) -> float | None:
    """Parse a simple numeric chart label while retaining its raw text too."""
    value = clean_text(text).replace(",", "")
    value = value.replace("%", "").strip()
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_metric_unit(
    metric_label: str,
    raw_unit_label: str | None = None,
    context_text: str = "",
) -> tuple[str | None, str | None]:
    """Return (normalized unit, method) for well-defined financial metrics.

    The extracted source label is never overwritten. This function only adds a
    standardized unit when the metric label/source context supports it.
    """
    metric = clean_text(metric_label).lower()
    raw_unit = clean_text(raw_unit_label).lower()
    context = clean_text(context_text).lower()
    combined = f"{metric} {raw_unit} {context}"

    # EPS is a per-share amount, not a percentage or an ordinary currency total.
    if "basic eps" in metric or "diluted eps" in metric or re.search(r"\beps\b", metric):
        if "$" in metric or "usd" in raw_unit or "us $" in raw_unit:
            return "USD per share", "metric_label_or_explicit_unit"
        if "₹" in metric or "inr" in raw_unit or "rupee" in raw_unit:
            return "INR per share", "metric_label_or_explicit_unit"
        # In this Infosys report the rupee glyph in the chart's '(in ₹)' unit
        # was extracted as a backtick. Only repair this known glyph issue for EPS.
        if "`" in raw_unit and ("basic eps" in metric or "diluted eps" in metric):
            return "INR per share", "verified_source_glyph_repair"
        if "in us $" in context or "usd" in context:
            return "USD per share", "table_unit_context"
        if "₹" in context or "inr" in context:
            return "INR per share", "table_unit_context"
        return None, None

    # These ratios are expressed as multiples ("number of times" in the source).
    if (
        "price / earnings" in metric
        or "price-to-earnings" in metric
        or re.search(r"\bp\s*/\s*e\b", metric)
        or "market capitalization / revenue" in metric
    ):
        if "represents number of times" in context:
            return "multiple (times)", "metric_and_source_footnote"
        return "multiple (times)", "metric_definition"

    if "day's sales outstanding" in metric or "days sales outstanding" in metric:
        return "days", "metric_label"

    # Ratio metrics are percentages; prioritize multiple-valued metrics above.
    percentage_terms = (
        "%" in metric
        or "%" in raw_unit
        or "/ revenue" in metric
        or "return on equity" in metric
        or re.search(r"\broe\b", metric) is not None
        or "margin" in metric
        or "tax / pbt" in metric
        or "in %" in raw_unit
    )
    if percentage_terms:
        return "percent", "metric_label_or_explicit_unit"

    # Common financial statement amounts inherit currency and scale from the
    # nearby table heading, e.g. '(In ₹ crore)' or '(in US $ millions)'.
    if "us $" in context or "usd" in context or ("$" in context and "million" in context):
        if "million" in context:
            return "USD million", "table_unit_context"
        return "USD", "table_unit_context"

    if "₹" in metric or "inr" in context or ("₹" in context and "crore" in context):
        if "book value" in metric:
            return "INR per share", "metric_label_and_source_context"
        if "crore" in context:
            return "INR crore", "table_unit_context"
        return "INR", "metric_label_or_table_unit_context"

    if "book value" in metric and ("₹" in raw_unit or "₹" in context or "inr" in context):
        return "INR per share", "metric_label_and_source_context"

    return None, None


def find_figure_unit_label(
    figure: dict,
    figure_bbox: tuple[float, float, float, float] | None,
    all_text_items: list[dict],
) -> str | None:
    """Find an explicitly printed unit line directly above the figure.

    This only preserves an extracted unit label; it does not infer a unit
    from the metric name.
    """
    if figure_bbox is None:
        return None

    figure_left, _, figure_top, _ = figure_bbox
    figure_pages = set(get_pages(figure))
    candidates = []

    for text_item in all_text_items:
        text = clean_text(text_item.get("text", ""))
        if not text or not re.match(r"^\(?\s*in\b", text, re.IGNORECASE):
            continue
        if figure_pages and not (figure_pages & set(get_pages(text_item))):
            continue

        coords = bbox_numbers(text_item)
        center_y = bbox_center_y(text_item)
        if coords is None or center_y is None:
            continue

        _, right, _, _ = coords
        # The unit should be immediately to the left of this figure. The old
        # test accepted any unit anywhere further left on the page, which could
        # incorrectly attach the Basic EPS rupee unit to the P/E chart.
        horizontal_gap = figure_left - right
        vertical_distance = center_y - figure_top
        if -5 <= horizontal_gap <= 180 and -2 <= vertical_distance <= 38:
            candidates.append((abs(vertical_distance), horizontal_gap, text))

    if not candidates:
        return None

    candidates.sort()
    return candidates[0][2]


def map_figure_values_to_periods(
    figure: dict,
    labels: list[dict],
    all_text_items: list[dict],
) -> tuple[list[dict], str]:
    """Match chart numeric labels to nearest period labels by page geometry.

    We require same-page provenance, a period label to the left of the chart
    value, and a close vertical alignment. If that evidence is missing, we do
    not guess a period association.
    """
    figure_bbox = bbox_numbers(figure)
    figure_pages = set(get_pages(figure))
    if not figure_pages:
        return [], "unmatched_missing_figure_page"

    periods = []
    for item in all_text_items:
        text = clean_text(item.get("text", ""))
        if not is_period_label(text):
            continue
        if not (figure_pages & set(get_pages(item))):
            continue
        coords = bbox_numbers(item)
        center_y = bbox_center_y(item)
        if coords is None or center_y is None:
            continue

        periods.append({
            "text": text,
            "source_ref": item.get("self_ref"),
            "page_numbers": get_pages(item),
            "bbox": get_bbox(item),
            "_left": coords[0],
            "_right": coords[1],
            "_center_y": center_y,
        })

    if not labels:
        return [], "unmatched_no_numeric_labels"
    if not periods:
        return [], "unmatched_no_period_labels"

    # Top-to-bottom ordering is used only as a deterministic tie-breaker.
    value_items = []
    for label in labels:
        text = clean_text(label.get("text", ""))
        coords = label.get("bbox")
        if not isinstance(coords, dict):
            continue
        try:
            left = float(coords["l"])
            top = float(coords["t"])
            bottom = float(coords["b"])
            center_y = (top + bottom) / 2.0
        except (KeyError, TypeError, ValueError):
            continue

        value_items.append({
            **label,
            "_left": left,
            "_center_y": center_y,
            "_numeric_value": parse_numeric_label(text),
        })

    if not value_items:
        return [], "unmatched_missing_value_coordinates"

    unused_period_refs = {p["source_ref"] for p in periods}
    observations = []

    # In Docling's page coordinates, higher y centers are closer to the top.
    value_items.sort(key=lambda x: -x["_center_y"])

    for value_item in value_items:
        candidates = []
        for period in periods:
            if period["source_ref"] not in unused_period_refs:
                continue
            # The period should appear to the left of this numeric label.
            if period["_right"] > value_item["_left"] + 5:
                continue
            vertical_gap = abs(period["_center_y"] - value_item["_center_y"])
            if vertical_gap > 8.0:
                continue
            horizontal_gap = max(0.0, value_item["_left"] - period["_right"])
            candidates.append((vertical_gap, horizontal_gap, period))

        if not candidates:
            continue

        candidates.sort(key=lambda row: (row[0], row[1]))
        vertical_gap, horizontal_gap, period = candidates[0]
        unused_period_refs.discard(period["source_ref"])

        observations.append({
            "period": period["text"],
            "raw_value": clean_text(value_item["text"]),
            "numeric_value": value_item["_numeric_value"],
            "unit_label": None,  # Filled by caller when an explicit unit exists.
            "source_document": None,  # Filled by caller.
            "source_page": (value_item.get("page_numbers") or figure_pages and sorted(figure_pages))[0],
            "figure_source_ref": figure.get("self_ref"),
            "value_source_ref": value_item.get("source_ref"),
            "period_source_ref": period.get("source_ref"),
            "alignment": {
                "vertical_gap_points": round(vertical_gap, 3),
                "horizontal_gap_points": round(horizontal_gap, 3),
                "method": "nearest_same_page_period_by_bbox",
            },
        })

    if not observations:
        return [], "unmatched_no_aligned_pairs"

    status = "matched_all_values" if len(observations) == len(value_items) else "partially_matched"
    return observations, status


# ---------------------------------------------------------------------------
# Walk the document body in reading order and preserve heading context.
# Text labels belonging to a picture are stored with that picture, rather than
# emitted as standalone numeric text records without context.
# ---------------------------------------------------------------------------


def ordered_body_items(document: dict, objects: dict[str, dict]):
    body = document.get("body") or {}
    visited: set[str] = set()

    def walk(references):
        for reference in references or []:
            key = get_ref(reference)
            if not key or key in visited:
                continue
            item = objects.get(key)
            if item is None:
                continue
            visited.add(key)

            if key.startswith("#/groups/"):
                yield from walk(item.get("children") or [])
                continue

            # Picture-child text (often chart labels) is handled as part of
            # the parent figure record. Do not create context-free text chunks.
            if key.startswith("#/texts/"):
                parent = get_ref(item.get("parent"))
                if parent and parent.startswith("#/pictures/"):
                    continue

            yield item

    yield from walk(body.get("children") or [])


# ---------------------------------------------------------------------------
# Table handling
# ---------------------------------------------------------------------------


def get_table_cells(item: dict) -> list[dict]:
    data = item.get("data") or {}
    cells = data.get("table_cells") or []
    # Support a nested row-list shape as well as Docling's normal flat list.
    if cells and isinstance(cells[0], list):
        return [cell for row in cells for cell in row if isinstance(cell, dict)]
    return [cell for cell in cells if isinstance(cell, dict)]


def cell_int(cell: dict, key: str, default: int = 0) -> int:
    try:
        return int(cell.get(key, default))
    except (TypeError, ValueError):
        return default


def expand_table(item: dict) -> tuple[list[list[str]], list[list[bool]], list[dict]]:
    """Reconstruct a table grid and preserve column-header flags.

    Merged cells are repeated over their covered grid positions, but the raw
    cell list is also returned so header spans and section rows can be handled
    without losing the original span metadata.
    """
    data = item.get("data") or {}
    cells = get_table_cells(item)
    if not cells:
        return [], [], []

    row_count = cell_int(data, "num_rows", 0)
    col_count = cell_int(data, "num_cols", 0)
    for cell in cells:
        row = cell_int(cell, "start_row_offset_idx")
        col = cell_int(cell, "start_col_offset_idx")
        row_span = max(1, cell_int(cell, "row_span", 1))
        col_span = max(1, cell_int(cell, "col_span", 1))
        row_count = max(row_count, row + row_span)
        col_count = max(col_count, col + col_span)

    if row_count <= 0 or col_count <= 0:
        return [], [], cells

    grid = [["" for _ in range(col_count)] for _ in range(row_count)]
    header_grid = [[False for _ in range(col_count)] for _ in range(row_count)]

    # Populate by cell start positions; if cells overlap, the later source cell
    # wins in the grid, while the unmodified raw cells remain available.
    for cell in cells:
        row = cell_int(cell, "start_row_offset_idx")
        col = cell_int(cell, "start_col_offset_idx")
        row_span = max(1, cell_int(cell, "row_span", 1))
        col_span = max(1, cell_int(cell, "col_span", 1))
        text = clean_text(cell.get("text", ""))
        is_column_header = bool(cell.get("column_header", False))

        for r in range(row, min(row + row_span, row_count)):
            for c in range(col, min(col + col_span, col_count)):
                grid[r][c] = text
                header_grid[r][c] = is_column_header

    return grid, header_grid, cells


def build_column_headers(
    grid: list[list[str]], cells: list[dict]
) -> tuple[list[str], list[int]]:
    """Build column names from the actual header cells and their spans.

    This supports common two-level headers such as:
      Three months ended June 30, (spans two columns)
        2025 | 2024

    We keep raw header cell information in the normalized record as well,
    because some complex source tables need manual or table-specific handling.
    """
    if not grid:
        return [], []

    column_count = max((len(row) for row in grid), default=0)
    header_cells = [
        cell for cell in cells
        if bool(cell.get("column_header", False)) and clean_text(cell.get("text", ""))
    ]
    header_rows = sorted({cell_int(cell, "start_row_offset_idx") for cell in header_cells})

    # If Docling did not label any header cells, do not invent header meaning.
    # Keep generic names and preserve every source row as data below.
    if not header_cells:
        return [f"Column {i + 1}" for i in range(column_count)], []

    headers: list[str] = []
    for col in range(column_count):
        parts: list[str] = []
        for cell in sorted(
            header_cells,
            key=lambda c: (
                cell_int(c, "start_row_offset_idx"),
                cell_int(c, "start_col_offset_idx"),
            ),
        ):
            start_col = cell_int(cell, "start_col_offset_idx")
            col_span = max(1, cell_int(cell, "col_span", 1))
            if not (start_col <= col < start_col + col_span):
                continue
            value = clean_text(cell.get("text", ""))
            # Avoid adding the same merged-cell header twice for one column.
            if value and value not in parts:
                parts.append(value)

        headers.append(" / ".join(parts) if parts else f"Column {col + 1}")

    # Make exact duplicate names unambiguous for dictionaries and JSON output.
    used: Counter[str] = Counter()
    unique_headers: list[str] = []
    for header in headers:
        used[header] += 1
        if used[header] == 1:
            unique_headers.append(header)
        else:
            unique_headers.append(f"{header} [{used[header]}]")

    return unique_headers, header_rows


def classify_table(title: str, section_path: list[str], grid: list[list[str]]) -> str:
    """Assign a coarse table category; do not infer a financial metric schema."""
    text = " ".join(
        [title, *section_path]
        + [cell for row in grid[:5] for cell in row[:8] if cell]
    ).lower()
    title_text = title.lower()

    # Specific types first to avoid mistaking stock-market tables for statements.
    if (
        "ratio analysis" in title_text
        or "ratio analysis" in text
        or "return on equity" in title_text and "price / earnings" in text
    ):
        return "financial_ratio"

    if (
        "stock market data" in text
        or "american depositary shares" in text
        or "volume at nyse" in text
        or ("bse" in text and "nse" in text and any(x in text for x in ("volume", "high (rs.)", "low (rs.)")))
    ):
        return "market_data"

    if any(
        phrase in text
        for phrase in (
            "statement of comprehensive income",
            "income statement",
            "balance sheet data",
            "statement of financial position",
            "cash flow statement",
            "statement of cash flows",
        )
    ):
        return "financial_statement"

    return "other_table"


def make_table_search_text(
    title: str,
    section_path: list[str],
    category: str,
    headers: list[str],
    section_rows: list[dict],
    data_rows: list[dict],
    nearby_context: list[str],
) -> str:
    lines = [
        f"Table title: {title}",
        f"Table category: {category}",
        f"Section: {' > '.join(section_path) if section_path else 'Unknown'}",
    ]
    if nearby_context:
        lines.append("Nearby source context: " + " | ".join(nearby_context))
    if headers:
        lines.append("Columns: " + " | ".join(headers))

    for section in section_rows:
        lines.append(f"Table subsection: {section['label']}")

    for row in data_rows:
        prefix = f"Row: {row['label']}"
        if row.get("section_context"):
            prefix += f" (subsection: {row['section_context']})"
        if row.get("normalized_unit"):
            prefix += f" (unit: {row['normalized_unit']})"
        values = row.get("values") or {}
        value_text = " | ".join(f"{key}: {value}" for key, value in values.items())
        lines.append(prefix + (f" | {value_text}" if value_text else ""))

    return "\n".join(line for line in lines if line.strip())


def build_table_record(
    item: dict,
    source: str,
    section_path: list[str],
    objects: dict[str, dict],
    nearby_context: list[str],
) -> dict:
    grid, header_grid, cells = expand_table(item)
    caption = get_caption(item, objects)
    title = caption or (section_path[-1] if section_path else "Untitled table")
    category = classify_table(title, section_path, grid)
    headers, header_rows = build_column_headers(grid, cells)
    header_row_set = set(header_rows)

    # Count original non-empty cell origins by row. This avoids treating a
    # section heading that spans several columns as a numeric data row merely
    # because its value was expanded across the grid.
    nonempty_origins_by_row: dict[int, list[dict]] = {}
    for cell in cells:
        value = clean_text(cell.get("text", ""))
        if not value:
            continue
        row = cell_int(cell, "start_row_offset_idx")
        nonempty_origins_by_row.setdefault(row, []).append(cell)

    section_rows: list[dict] = []
    data_rows: list[dict] = []
    active_subsection: str | None = None

    for row_index, row in enumerate(grid):
        values = [clean_text(value) for value in row]
        if not any(values) or row_index in header_row_set:
            continue

        origins = nonempty_origins_by_row.get(row_index, [])
        nonempty_origins = [cell for cell in origins if clean_text(cell.get("text", ""))]

        # A single source cell in a non-header row is generally a subsection
        # label (e.g. Financial performance (%), Balance sheet, Return, Per share).
        # Preserve it as context, not as a financial observation.
        if len(nonempty_origins) == 1:
            label = clean_text(nonempty_origins[0].get("text", ""))
            section_row = {
                "row_index": row_index,
                "label": label,
                "source_cell": {
                    "start_row": cell_int(nonempty_origins[0], "start_row_offset_idx"),
                    "start_col": cell_int(nonempty_origins[0], "start_col_offset_idx"),
                    "row_span": max(1, cell_int(nonempty_origins[0], "row_span", 1)),
                    "col_span": max(1, cell_int(nonempty_origins[0], "col_span", 1)),
                },
            }
            section_rows.append(section_row)
            active_subsection = label
            continue

        # The first column is the row label (e.g. Particulars/Metric). It is
        # deliberately excluded from `values` to avoid duplicating the label.
        label = values[0] if values else ""
        if not label:
            label = next((value for value in values if value), "")

        mapped_values: dict[str, str] = {}
        for col in range(1, len(values)):
            value = values[col]
            if not value:
                continue
            header = headers[col] if col < len(headers) else f"Column {col + 1}"
            mapped_values[header] = value

        unit_context = " ".join(
            part for part in [title, " ".join(section_path), active_subsection or "", *nearby_context]
            if part
        )
        normalized_unit, unit_method = normalize_metric_unit(
            label, raw_unit_label=None, context_text=unit_context
        )

        data_rows.append({
            "row_index": row_index,
            "label": label,
            "values": mapped_values,
            "cells": values,
            "section_context": active_subsection,
            "normalized_unit": normalized_unit,
            "unit_normalization_method": unit_method,
        })

    pages = get_pages(item)
    searchable_text = make_table_search_text(
        title, section_path, category, headers,
        section_rows, data_rows, nearby_context,
    )

    # Preserve source header-cell metadata so callers can audit complicated
    # multi-level headings instead of relying only on the flattened labels.
    raw_header_cells = []
    for cell in cells:
        if not bool(cell.get("column_header", False)):
            continue
        raw_header_cells.append({
            "text": clean_text(cell.get("text", "")),
            "start_row": cell_int(cell, "start_row_offset_idx"),
            "start_col": cell_int(cell, "start_col_offset_idx"),
            "row_span": max(1, cell_int(cell, "row_span", 1)),
            "col_span": max(1, cell_int(cell, "col_span", 1)),
        })

    return {
        "record_id": stable_id(source, item.get("self_ref", ""), "table"),
        "record_type": "table",
        "table_category": category,
        "content": searchable_text,
        "title": title,
        "caption": caption or None,
        "section_path": section_path,
        "headers": headers,
        "header_rows": header_rows,
        "raw_header_cells": raw_header_cells,
        "rows": grid,
        "section_rows": section_rows,
        "data_rows": data_rows,
        "nearby_context": nearby_context,
        "source_document": source,
        "source_ref": item.get("self_ref"),
        "page_numbers": pages,
        "source_page": pages[0] if pages else None,
        "bbox": get_bbox(item),
    }


# ---------------------------------------------------------------------------
# Figure handling
# ---------------------------------------------------------------------------


def save_figure(item: dict, figure_dir: Path, number: int) -> str | None:
    image = item.get("image") or {}
    uri = image.get("uri") if isinstance(image, dict) else None
    if not isinstance(uri, str):
        return None

    match = re.match(r"^data:image/([^;]+);base64,(.*)$", uri, re.DOTALL)
    if not match:
        return None

    image_format, encoded = match.groups()
    try:
        binary = base64.b64decode(re.sub(r"\s+", "", encoded), validate=True)
    except (ValueError, binascii.Error):
        return None

    extension = {"jpeg": "jpg", "svg+xml": "svg"}.get(image_format.lower(), image_format.lower())
    figure_dir.mkdir(parents=True, exist_ok=True)
    filename = f"figure_{number:04d}.{extension}"
    (figure_dir / filename).write_bytes(binary)
    return f"figures/{filename}"


def build_figure_record(
    item: dict,
    source: str,
    section_path: list[str],
    objects: dict[str, dict],
    figure_dir: Path,
    number: int,
    all_text_items: list[dict],
) -> dict:
    labels = []
    for child_ref in item.get("children") or []:
        key = get_ref(child_ref)
        child = objects.get(key, {}) if key else {}
        if not key or not key.startswith("#/texts/"):
            continue
        value = clean_text(child.get("text", ""))
        if value:
            labels.append({
                "text": value,
                "source_ref": key,
                "page_numbers": get_pages(child),
                "bbox": get_bbox(child),
            })

    caption = get_caption(item, objects)
    pages = get_pages(item)
    figure_bbox = bbox_numbers(item)
    image_path = save_figure(item, figure_dir, number)
    section = " > ".join(section_path) if section_path else "Unknown section"
    label_string = "; ".join(label["text"] for label in labels)

    observations, mapping_status = map_figure_values_to_periods(
        item, labels, all_text_items
    )
    explicit_unit_label = find_figure_unit_label(item, figure_bbox, all_text_items)
    metric_label = section_path[-1] if section_path else (caption or "")
    page_text = " ".join(
        clean_text(text_item.get("text", ""))
        for text_item in all_text_items
        if set(pages) & set(get_pages(text_item))
    )
    normalized_unit, unit_method = normalize_metric_unit(
        metric_label,
        raw_unit_label=explicit_unit_label,
        context_text=page_text,
    )
    display_unit = normalized_unit or explicit_unit_label

    for observation in observations:
        observation["explicit_unit_label"] = explicit_unit_label
        observation["unit_label"] = display_unit
        observation["normalized_unit"] = normalized_unit
        observation["unit_normalization_method"] = unit_method
        observation["source_document"] = source

    observation_lines = [
        f"{obs['period']}: {obs['raw_value']}"
        + (f" {display_unit}" if display_unit else "")
        for obs in observations
    ]

    # Searchable text includes metric heading and validated period/value pairs.
    # It deliberately keeps unpaired labels visible without assigning periods.
    content_parts = [
        f"Figure section: {section}",
        f"Caption: {caption or 'No caption extracted'}",
        f"Extracted labels: {label_string or 'None'}",
    ]
    if observation_lines:
        content_parts.append("Period/value observations: " + "; ".join(observation_lines))
    if explicit_unit_label:
        content_parts.append(f"Extracted unit label: {explicit_unit_label}")
    if normalized_unit:
        content_parts.append(f"Normalized unit: {normalized_unit}")

    return {
        "record_id": stable_id(source, item.get("self_ref", ""), "figure"),
        "record_type": "figure",
        "content": "\n".join(content_parts),
        "caption": caption or None,
        "section_path": section_path,
        "labels": labels,
        "observations": observations,
        "period_value_mapping_status": mapping_status,
        "explicit_unit_label": explicit_unit_label,
        "unit_label": display_unit,
        "normalized_unit": normalized_unit,
        "unit_normalization_method": unit_method,
        "image_path": image_path,
        "source_document": source,
        "source_ref": item.get("self_ref"),
        "page_numbers": pages,
        "source_page": pages[0] if pages else None,
        "bbox": get_bbox(item),
    }


# ---------------------------------------------------------------------------
# Main normalization pass
# ---------------------------------------------------------------------------


def normalize_document(document: dict, source: str, output_dir: Path) -> list[dict]:
    objects = index_objects(document)
    records: list[dict] = []
    heading_stack: dict[int, str] = {}
    recent_text_by_page: dict[int, list[str]] = {}
    figure_dir = output_dir / "figures"
    figure_number = 0

    for item in ordered_body_items(document, objects):
        self_ref = item.get("self_ref", "")
        label = str(item.get("label", "")).lower()
        content_layer = item.get("content_layer", "body")
        pages = get_pages(item)

        # Repeated page furniture harms retrieval and is omitted.
        if label in {"page_header", "page_footer", "page_number"} or content_layer == "furniture":
            continue

        if self_ref.startswith("#/texts/"):
            original = item.get("text", "")
            content = clean_text(original)
            if not content:
                continue

            if label == "section_header":
                try:
                    level = max(1, int(item.get("level", 1)))
                except (TypeError, ValueError):
                    level = 1
                heading_stack = {k: v for k, v in heading_stack.items() if k < level}
                heading_stack[level] = content

            section_path = [heading_stack[k] for k in sorted(heading_stack)]
            record = {
                "record_id": stable_id(source, self_ref, "text"),
                "record_type": "text",
                "content": content,
                "original_text": original,
                "label": item.get("label"),
                "section_path": section_path,
                "source_document": source,
                "source_ref": self_ref,
                "parent_ref": get_ref(item.get("parent")),
                "page_numbers": pages,
                "source_page": pages[0] if pages else None,
                "bbox": get_bbox(item),
            }
            records.append(record)

            # Keep a small amount of text context to attach unit lines such as
            # '(In ₹ crore except per share data)' to a following table.
            for page in pages:
                recent_text_by_page.setdefault(page, []).append(content)
                recent_text_by_page[page] = recent_text_by_page[page][-5:]

        elif self_ref.startswith("#/tables/"):
            section_path = [heading_stack[k] for k in sorted(heading_stack)]
            # Use preceding text from the same page, but don't try to guess
            # which line is the unit. We preserve it as nearby context.
            page = pages[0] if pages else None
            nearby = recent_text_by_page.get(page, [])[-4:] if page is not None else []
            records.append(
                build_table_record(item, source, section_path, objects, nearby)
            )

        elif self_ref.startswith("#/pictures/"):
            section_path = [heading_stack[k] for k in sorted(heading_stack)]
            figure_number += 1
            records.append(
                build_figure_record(
                    item, source, section_path, objects, figure_dir, figure_number,
                    document.get("texts") or [],
                )
            )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize structured Docling JSON for financial retrieval."
    )
    parser.add_argument("input_json", type=Path, help="Docling JSON input path")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.input_json.is_file():
        raise FileNotFoundError(f"Docling JSON not found: {args.input_json}")

    with args.input_json.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    document = payload.get("document", payload)
    if not isinstance(document, dict) or "body" not in document:
        raise ValueError("Input does not look like a supported DoclingDocument JSON.")

    origin = document.get("origin") or {}
    source = origin.get("filename") or args.input_json.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = normalize_document(document, source, args.output_dir)
    records_path = args.output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = Counter(record["record_type"] for record in records)
    table_categories = Counter(
        record["table_category"]
        for record in records
        if record["record_type"] == "table"
    )
    summary = {
        "source_document": source,
        "total_records": len(records),
        "records_by_type": dict(counts),
        "tables_by_category": dict(table_categories),
        "records_file": str(records_path),
        "figures_directory": str(args.output_dir / "figures"),
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
