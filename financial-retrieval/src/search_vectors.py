from pathlib import Path
import argparse

import chromadb
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "quarterlens_chunks"

ROOT = Path.cwd()
VECTOR_DB = (
    ROOT
    / "financial-retrieval"
    / "artifacts"
    / "vector_store"
    / "chroma"
)


def main():
    parser = argparse.ArgumentParser(
        description="Search the QuarterLens vector index."
    )
    parser.add_argument(
        "query",
        help="Financial question to search for."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return."
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    if not VECTOR_DB.exists():
        raise FileNotFoundError(
            f"Vector store not found: {VECTOR_DB}"
        )

    client = chromadb.PersistentClient(path=str(VECTOR_DB))

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    if collection.count() == 0:
        raise RuntimeError("The vector collection is empty.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    query_embedding = model.encode(
        args.query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(args.top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nQuery: {args.query}")
    print(f"Results returned: {len(results['ids'][0])}")

    for rank, chunk_id in enumerate(results["ids"][0], start=1):
        metadata = results["metadatas"][0][rank - 1] or {}
        content = results["documents"][0][rank - 1] or ""
        distance = results["distances"][0][rank - 1]

        print(f"\n{'=' * 65}")
        print(f"Rank: {rank}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Document: {metadata.get('source_document', 'Unknown')}")
        print(f"Page: {metadata.get('source_page', 'Unknown')}")
        print(f"Type: {metadata.get('record_type', 'Unknown')}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"\nEvidence:\n{content}")

    print("\nLower cosine distance generally means greater vector similarity.")
    print("Similarity alone does not establish factual correctness.")


if __name__ == "__main__":
    main()