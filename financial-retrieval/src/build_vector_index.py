from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from retrieval import retrieval_text


# Run this script from the outer financial-rag directory.
PROJECT_ROOT = Path.cwd()

CHUNKS_DIR = (
    PROJECT_ROOT
    / "financial-retrieval"
    / "artifacts"
    / "chunks"
)

VECTOR_STORE_DIR = (
    PROJECT_ROOT
    / "financial-retrieval"
    / "artifacts"
    / "vector_store"
    / "chroma"
)

COLLECTION_NAME = "quarterlens_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_chunks() -> list[dict[str, Any]]:
    """Load the JSONL chunks created by chunk_records.py."""
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
                        f"line {line_number}: {exc}"
                    ) from exc

                content = str(chunk.get("content") or "").strip()
                chunk_id = str(chunk.get("chunk_id") or "").strip()

                if not content or not chunk_id:
                    continue

                # Avoid indexing a text chunk consisting only of its title.
                title = str(chunk.get("title") or "").strip()
                if (
                    chunk.get("record_type") == "text"
                    and title
                    and content.casefold() == title.casefold()
                ):
                    continue

                chunks.append(chunk)

    if not chunks:
        raise ValueError("No usable chunks were found.")

    ids = [chunk["chunk_id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate chunk IDs detected. Fix these before indexing."
        )

    return chunks


def to_chroma_metadata(
    chunk: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    """Convert chunk metadata to Chroma-compatible scalar fields."""
    metadata: dict[str, str | int | float | bool] = {
        "record_type": str(chunk.get("record_type") or "unknown"),
    }

    scalar_fields = [
        "source_document",
        "title",
        "table_title",
        "table_category",
        "normalized_unit",
        "source_ref",
        "source_record_id",
        "source_record_type",
    ]

    for field in scalar_fields:
        value = chunk.get(field)

        if isinstance(value, (str, int, float, bool)):
            if str(value).strip():
                metadata[field] = value

    page = chunk.get("source_page")

    if isinstance(page, int):
        metadata["source_page"] = page
    elif isinstance(page, str) and page.strip().isdigit():
        metadata["source_page"] = int(page.strip())

    section = chunk.get("section_path")

    if isinstance(section, list):
        section = " > ".join(str(item) for item in section if item)

    if isinstance(section, str) and section.strip():
        metadata["section_path"] = section.strip()

    return metadata


def main() -> int:
    if not CHUNKS_DIR.exists():
        print(f"Chunks directory not found: {CHUNKS_DIR}")
        return 1

    try:
        chunks = load_chunks()
    except (OSError, ValueError) as exc:
        print(f"Failed to load chunks: {exc}")
        return 1

    documents = [retrieval_text(chunk) for chunk in chunks]
    ids = [str(chunk["chunk_id"]) for chunk in chunks]
    metadatas = [to_chroma_metadata(chunk) for chunk in chunks]

    source_documents = sorted({
        str(metadata["source_document"])
        for metadata in metadatas
        if metadata.get("source_document")
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Chunks loaded: {len(chunks)}")
    print(f"Embedding model: {MODEL_NAME}")
    print(f"Device: {device}")
    print("Loading embedding model...")

    try:
        model = SentenceTransformer(MODEL_NAME, device=device)

        print("Generating embeddings...")
        embeddings = model.encode(
            documents,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Each embedding is a numerical vector representing a chunk.
        embedding_rows = embeddings.tolist()

        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_DIR)
        )

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

        # Remove older entries for these source documents before rebuilding.
        # This prevents stale chunks from surviving after a rebuild.
        if collection.count() > 0 and source_documents:
            collection.delete(
                where={
                    "source_document": {
                        "$in": source_documents
                    }
                }
            )

        # Insert in batches to avoid oversized database requests.
        for start in range(0, len(chunks), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(chunks))

            collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embedding_rows[start:end],
            )

            print(f"Indexed {end}/{len(chunks)} chunks")

        print("\nVector index built successfully.")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Stored vectors: {collection.count()}")
        print(f"Vector store location: {VECTOR_STORE_DIR}")

    except Exception as exc:
        print(f"\nVector indexing failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())