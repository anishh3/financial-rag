from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load normalized records from a JSONL file."""
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            if isinstance(record, dict):
                records.append(record)

    return records


def clean(value: Any) -> str:
    """Normalize whitespace without modifying the underlying source data."""
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return str(value).strip()


def make_id(*parts: Any) -> str:
    """Generate reproducible IDs for chunks."""
    raw = "::".join(str(part or "") for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"chunk_{digest}"


def get_source_metadata(record: dict[str, Any]) -> dict[str, Any]:
    pages = record.get("page_numbers")

    if not isinstance(pages, list) or not pages:
        page = record.get("source_page")
        pages = [page] if page is not None else []

    return {
        "source_document": record.get("source_document"),
        "source_page": record.get("source_page"),
        "page_numbers": pages,
        "source_ref": record.get("source_ref"),
        "section_path": record.get("section_path", []),
    }


def format_values(values: Any) -> list[str]:
    """Keep each financial value connected to its reporting period."""
    result = []

    if isinstance(values, dict):
        for period, value in values.items():
            period = clean(period)
            value = clean(value)

            if period and value:
                result.append(f"{period}: {value}")

    elif isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                period = clean(item.get("period"))
                value = clean(
                    item.get("raw_value", item.get("value"))
                )
                if period and value:
                    result.append(f"{period}: {value}")

    return result


def chunk_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Create one searchable chunk per table row.

    Keeping rows separate helps retrieval find a specific metric
    without losing its reporting periods or units.
    """
    title = clean(record.get("title")) or "Untitled table"
    category = record.get("table_category")
    rows = record.get("data_rows")

    if not isinstance(rows, list) or not rows:
        content = clean(record.get("content"))

        if not content:
            return []

        return [{
            "chunk_id": make_id(
                record.get("source_document"),
                record.get("record_id"),
                "whole-table",
            ),
            "record_type": "table",
            "title": title,
            "table_title": title,
            "table_category": category,
            "content": content,
            "source_record_id": record.get("record_id"),
            "source_record_type": "table",
            "headers": record.get("headers", []),
            **get_source_metadata(record),
        }]

    headers = record.get("headers", [])
    header_text = " | ".join(
        clean(header) for header in headers if clean(header)
    )

    nearby = record.get("nearby_context")
    if isinstance(nearby, list):
        nearby_text = " > ".join(
            clean(item) for item in nearby if clean(item)
        )
    else:
        nearby_text = clean(nearby)

    chunks = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue

        label = clean(row.get("label"))
        if not label:
            continue

        values = row.get("values", {})
        unit = clean(
            row.get("normalized_unit") or row.get("unit")
        )
        section = clean(row.get("section_context"))

        parts = [f"Table: {title}"]

        if category:
            parts.append(f"Table category: {category}")

        if nearby_text:
            parts.append(f"Document context: {nearby_text}")

        if header_text:
            parts.append(f"Column headers: {header_text}")

        if section:
            parts.append(f"Section: {section}")

        parts.append(f"Metric: {label}")

        if unit:
            parts.append(f"Unit: {unit}")

        parts.extend(format_values(values))

        chunks.append({
            "chunk_id": make_id(
                record.get("source_document"),
                record.get("record_id"),
                row.get("row_index", index),
                label,
            ),
            "record_type": "table_row",
            "title": label,
            "table_title": title,
            "table_category": category,
            "section_context": section or None,
            "normalized_unit": unit or None,
            "content": "\n".join(parts),
            "values": values if isinstance(values, dict) else {},
            "headers": headers,
            "row_index": row.get("row_index", index),
            "source_record_id": record.get("record_id"),
            "source_record_type": "table",
            **get_source_metadata(record),
        })

    return chunks


def chunk_figure(record: dict[str, Any]) -> dict[str, Any] | None:
    """Keep a figure and its period/value observations together."""
    content = clean(record.get("content"))

    if not content:
        return None

    section_path = record.get("section_path", [])
    section = (
        " > ".join(clean(part) for part in section_path if clean(part))
        if isinstance(section_path, list)
        else ""
    )

    title = clean(record.get("caption")) or section or "Figure"

    return {
        "chunk_id": make_id(
            record.get("source_document"),
            record.get("record_id"),
            "figure",
        ),
        "record_type": "figure",
        "title": title,
        "content": content,
        "caption": record.get("caption"),
        "normalized_unit": record.get("normalized_unit"),
        "period_value_mapping_status": record.get(
            "period_value_mapping_status"
        ),
        "source_record_id": record.get("record_id"),
        "source_record_type": "figure",
        **get_source_metadata(record),
    }


def chunk_text(
    records: list[dict[str, Any]],
    max_chars: int,
) -> list[dict[str, Any]]:
    """Group nearby prose from the same page and section."""
    chunks = []
    buffer = []
    buffer_records = []
    current_key = None

    def flush():
        nonlocal buffer, buffer_records

        if not buffer:
            return

        first = buffer_records[0]
        last = buffer_records[-1]
        content = "\n".join(buffer).strip()

        if content:
            section_path = first.get("section_path", [])
            page_numbers = sorted({
                page
                for item in buffer_records
                for page in (
                    item.get("page_numbers", [])
                    if isinstance(item.get("page_numbers"), list)
                    else []
                )
                if page is not None
            })

            if not page_numbers and first.get("source_page") is not None:
                page_numbers = [first["source_page"]]

            chunks.append({
                "chunk_id": make_id(
                    first.get("source_document"),
                    first.get("record_id"),
                    last.get("record_id"),
                    "text",
                ),
                "record_type": "text",
                "title": (
                    " > ".join(clean(p) for p in section_path if clean(p))
                    if isinstance(section_path, list) and section_path
                    else "Document text"
                ),
                "content": content,
                "source_record_ids": [
                    item.get("record_id")
                    for item in buffer_records
                    if item.get("record_id")
                ],
                "source_record_type": "text",
                "source_document": first.get("source_document"),
                "source_page": first.get("source_page"),
                "page_numbers": page_numbers,
                "source_ref": first.get("source_ref"),
                "section_path": section_path,
            })

        buffer = []
        buffer_records = []

    for record in records:
        if record.get("record_type") != "text":
            continue

        text = clean(record.get("content"))

        if not text:
            continue

        # Chart periods and unit-only labels already belong to figure chunks.
        if re.fullmatch(
            r"(?i)(?:(?:quarter|three months|six months|nine months|year)"
            r"\s+ended\b.*\b20\d{2}|\(?in\s+(?:%|₹|`|\$|rupees?|days?)\)?)",
            text,
        ):
            continue

        section_path = record.get("section_path", [])
        page = record.get("source_page")

        key = (
            record.get("source_document"),
            page,
            tuple(str(part) for part in section_path)
            if isinstance(section_path, list)
            else (),
        )

        proposed_length = (
            sum(len(part) for part in buffer)
            + len(text)
            + len(buffer)
        )

        if buffer and (
            key != current_key
            or proposed_length > max_chars
        ):
            flush()

        current_key = key
        buffer.append(text)
        buffer_records.append(record)

    flush()
    return chunks


def create_chunks(
    records: list[dict[str, Any]],
    max_text_chars: int,
) -> list[dict[str, Any]]:
    chunks = []

    # Financial tables are split by row; figures remain grouped.
    for record in records:
        record_type = record.get("record_type")

        if record_type == "table":
            chunks.extend(chunk_table(record))

        elif record_type == "figure":
            figure = chunk_figure(record)
            if figure:
                chunks.append(figure)

    # Prose is grouped separately by page and section.
    chunks.extend(chunk_text(records, max_text_chars))

    # Guard against accidental duplicate IDs.
    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare normalized QuarterLens records for retrieval."
    )

    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path("financial-retrieval/artifacts/normalized"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("financial-retrieval/artifacts/chunks"),
    )

    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=1200,
    )

    args = parser.parse_args()

    if args.max_text_chars < 200:
        parser.error("--max-text-chars must be at least 200")

    if not args.normalized_dir.exists():
        print(
            f"Error: directory not found: {args.normalized_dir}",
            file=sys.stderr,
        )
        return 1

    input_files = sorted(
        args.normalized_dir.rglob("records.jsonl")
    )

    if not input_files:
        print(
            f"Error: no records.jsonl files found under "
            f"{args.normalized_dir}",
            file=sys.stderr,
        )
        return 1

    total_chunks = 0
    processed_documents = 0

    for input_file in input_files:
        try:
            records = read_jsonl(input_file)
            chunks = create_chunks(records, args.max_text_chars)
        except (OSError, ValueError) as exc:
            print(f"Error processing {input_file}: {exc}", file=sys.stderr)
            return 1

        if not chunks:
            print(f"Warning: no chunks created from {input_file}")
            continue

        document_name = input_file.parent.name
        output_folder = args.output_dir / document_name
        output_folder.mkdir(parents=True, exist_ok=True)

        output_file = output_folder / "chunks.jsonl"

        with output_file.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )

        counts = {}
        for chunk in chunks:
            kind = chunk["record_type"]
            counts[kind] = counts.get(kind, 0) + 1

        print(
            f"{document_name}: {len(records)} records -> "
            f"{len(chunks)} chunks"
        )
        print(f"  Chunk types: {counts}")
        print(f"  Output: {output_file}")

        total_chunks += len(chunks)
        processed_documents += 1

    print(f"\nDocuments processed: {processed_documents}")
    print(f"Total chunks created: {total_chunks}")
    print("Original normalized records were not modified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())