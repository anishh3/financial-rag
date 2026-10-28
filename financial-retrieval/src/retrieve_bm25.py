from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi


ROOT = Path.cwd()
CHUNKS_DIR = ROOT / "financial-retrieval" / "artifacts" / "chunks"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "did", "do", "for", "from", "how", "in", "is", "it",
    "of", "on", "the", "to", "was", "what", "were", "with",
}


def tokenize(text: str) -> list[str]:
    """Normalize text into searchable word and numeric tokens."""
    tokens = re.findall(
        r"[a-z]+|\d+(?:[,.]\d+)*",
        text.lower(),
    )
    return [token for token in tokens if token not in STOPWORDS]


def load_chunks() -> list[dict]:
    """Load all prepared chunks from all source documents."""
    files = sorted(CHUNKS_DIR.rglob("chunks.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"No chunks.jsonl files found under {CHUNKS_DIR}. "
            "Run chunk_records.py first."
        )

    chunks = []

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {file_path}, "
                        f"line {line_number}"
                    ) from exc

                if not chunk.get("chunk_id") or not chunk.get("content"):
                    continue

                chunks.append(chunk)

    ids = [chunk["chunk_id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    if not chunks:
        raise ValueError("No usable chunks found.")

    return chunks


def searchable_text(chunk: dict) -> str:
    """
    Put extra emphasis on the chunk title.

    For financial table rows, title is usually the metric name,
    such as 'Operating profit' or 'Revenue'.
    """
    title = str(chunk.get("title") or "")
    table_title = str(chunk.get("table_title") or "")
    content = str(chunk.get("content") or "")

    return f"{title} {title} {table_title} {content}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search QuarterLens chunks with BM25."
    )

    parser.add_argument("query", help="Your financial search query.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to display.",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    try:
        chunks = load_chunks()
    except (OSError, ValueError) as exc:
        print(f"Error loading chunks: {exc}", file=sys.stderr)
        return 1

    corpus = [
        tokenize(searchable_text(chunk))
        for chunk in chunks
    ]

    # BM25 needs tokenized text, not embeddings.
    index = BM25Okapi(corpus)
    query_tokens = tokenize(args.query)

    if not query_tokens:
        print("The query contains no searchable terms.")
        return 1

    scores = index.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:args.top_k]

    print(f"\nQuery: {args.query}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Query tokens: {query_tokens}")
    print(f"Results returned: {len(ranked_indices)}")

    for rank, index_number in enumerate(ranked_indices, start=1):
        chunk = chunks[index_number]

        print("\n" + "=" * 65)
        print(f"Rank: {rank}")
        print(f"BM25 score: {scores[index_number]:.4f}")
        print(f"Chunk ID: {chunk.get('chunk_id')}")
        print(f"Document: {chunk.get('source_document')}")
        print(f"Page: {chunk.get('source_page')}")
        print(f"Type: {chunk.get('record_type')}")
        print(f"Title: {chunk.get('title')}")
        print(f"\nEvidence:\n{chunk.get('content')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())