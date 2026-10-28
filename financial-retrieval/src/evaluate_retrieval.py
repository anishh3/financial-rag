
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from retrieval import (
    CANDIDATE_POOL,
    RERANK_CANDIDATE_POOL,
    retrieval_text,
    retrieve,
)


ROOT = Path.cwd()

CHUNKS_DIR = ROOT / "financial-retrieval" / "artifacts" / "chunks"
VECTOR_DB = (
    ROOT
    / "financial-retrieval"
    / "artifacts"
    / "vector_store"
    / "chroma"
)
OUTPUT_DIR = ROOT / "financial-retrieval" / "artifacts" / "evaluation"

COLLECTION_NAME = "quarterlens_chunks"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60
FUSION_POOL = CANDIDATE_POOL

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "did", "do", "for", "from", "how", "in", "is", "it",
    "of", "on", "the", "to", "was", "what", "were", "with",
}

ALIASES = {
    "revenues": "revenue",
    "expenses": "expense",
    "earnings": "earning",
    "taxes": "tax",
    "quarters": "quarter",
    "quarterly": "quarter",
    "months": "month",
    "years": "year",
    "shares": "share",
    "assets": "asset",
    "liabilities": "liability",
    "investments": "investment",
    "ratios": "ratio",
    "returns": "return",
    "statements": "statement",
    "operations": "operation",
}

# Each question has one expected source row.
# Verify these source rows against the original reports before
# treating the benchmark as representative.
QUESTIONS = [
    {
        "id": "Q01",
        "query": "What was Infosys revenue in Q1 FY2026?",
        "document": "q1-2026",
        "page": 3,
        "title": "Revenues",
    },
    {
        "id": "Q02",
        "query": "How much operating profit was reported for the June 2025 quarter?",
        "document": "q1-2026",
        "page": 3,
        "title": "Operating profit",
    },
    {
        "id": "Q03",
        "query": "What was revenue in the September 2025 quarter?",
        "document": "q2-2026",
        "page": 3,
        "title": "Revenues",
    },
    {
        "id": "Q04",
        "query": "Operating profit in Q2 FY2026 in INR crore",
        "document": "q2-2026",
        "page": 3,
        "title": "Operating profit",
    },
    {
        "id": "Q05",
        "query": "What was revenue for the December 2025 quarter?",
        "document": "q3-2026",
        "page": 3,
        "title": "Revenues",
    },
    {
        "id": "Q06",
        "query": (
            "Infosys operating profit in INR crore for the three months "
            "ended December 31 2025"
        ),
        "document": "q3-2026",
        "page": 3,
        "title": "Operating profit",
    },
    {
        "id": "Q07",
        "query": "What was Infosys gross profit in the December quarter?",
        "document": "q3-2026",
        "page": 3,
        "title": "Gross profit",
    },
    {
        "id": "Q08",
        "query": "What was profit before income taxes in Q3 FY2026?",
        "document": "q3-2026",
        "page": 3,
        "title": "Profit before income taxes",
    },
    {
        "id": "Q09",
        "query": "What was basic earnings per share in rupees for Q3 FY2026?",
        "document": "q3-2026",
        "page": 3,
        "title": "Basic EPS (₹)",
    },
    {
        "id": "Q10",
        "query": "What was Infosys annual revenue for fiscal year 2026?",
        "document": "additional-information-2025-26",
        "page": 4,
        "title": "Revenues",
    },
    {
        "id": "Q11",
        "query": "How much operating profit did Infosys report for the full year 2026?",
        "document": "additional-information-2025-26",
        "page": 4,
        "title": "Operating profit",
    },
    {
        "id": "Q12",
        "query": "What was the full-year profit after tax in FY2026?",
        "document": "additional-information-2025-26",
        "page": 4,
        "title": "Profit after tax",
    },
    {
        "id": "Q13",
        "query": (
            "What percentage of revenue was operating profit "
            "in the December 2025 quarter?"
        ),
        "document": "q3-2026",
        "page": 1,
        "title": "Operating profit / revenue",
    },
    {
        "id": "Q14",
        "query": (
            "What was the operating profit margin "
            "in the September 2025 quarter?"
        ),
        "document": "q2-2026",
        "page": 1,
        "title": "Operating profit / revenue",
    },
    {
        "id": "Q15",
        "query": "What proportion of revenue came from exports in FY2026?",
        "document": "additional-information-2025-26",
        "page": 1,
        "title": "Export revenue / revenue",
    },
]


def normalize_key(value: object) -> str:
    """Normalize labels for metric-title comparison."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def document_key(value: object) -> str:
    """Treat filenames with or without .pdf as the same document."""
    return Path(str(value or "")).stem.casefold()


def tokenize(text: str) -> list[str]:
    """Normalize common financial terms for BM25."""
    tokens = re.findall(r"[a-z]+|\d+(?:[,.]\d+)*", text.casefold())

    normalized = []

    for token in tokens:
        if token in STOPWORDS:
            continue
        normalized.append(ALIASES.get(token, token))

    return normalized


def searchable_text(chunk: dict) -> str:
    """Use the same row-focused representation as shared retrieval."""
    return retrieval_text(chunk)


def load_chunks() -> list[dict]:
    """Load the shared retrieval corpus from all chunk files."""
    files = sorted(CHUNKS_DIR.rglob("chunks.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"No chunks.jsonl files found under {CHUNKS_DIR}"
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
                        f"Invalid JSON in {file_path}, line {line_number}"
                    ) from exc

                content = str(chunk.get("content") or "").strip()
                chunk_id = str(chunk.get("chunk_id") or "").strip()
                title = str(chunk.get("title") or "").strip()

                if not content or not chunk_id:
                    continue

                # Match the exclusion rule used by the vector index.
                if (
                    chunk.get("record_type") == "text"
                    and title
                    and content.casefold() == title.casefold()
                ):
                    continue

                chunks.append(chunk)

    ids = [str(chunk["chunk_id"]) for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    return sorted(chunks, key=lambda chunk: str(chunk["chunk_id"]))


def get_expected_chunk(question: dict, chunks: list[dict]) -> dict:
    """Locate the expected row by document, page, and metric label."""
    expected_title = normalize_key(question["title"])
    expected_document = document_key(question["document"])
    expected_page = str(question["page"])

    matches = [
        chunk
        for chunk in chunks
        if document_key(chunk.get("source_document")) == expected_document
        and str(chunk.get("source_page")) == expected_page
        and normalize_key(chunk.get("title")) == expected_title
        and chunk.get("record_type") == "table_row"
    ]

    if len(matches) != 1:
        raise ValueError(
            f"{question['id']}: expected exactly one matching row, "
            f"but found {len(matches)}. "
            f"Document={expected_document}, page={expected_page}, "
            f"title={question['title']!r}"
        )

    return matches[0]


def reciprocal_rank_fusion(
    bm25_ranks: dict[str, int],
    vector_ranks: dict[str, int],
) -> list[str]:
    """Fuse the top BM25 and vector candidates using RRF."""
    candidates = {
        chunk_id
        for chunk_id, rank in bm25_ranks.items()
        if rank <= FUSION_POOL
    }
    candidates.update(
        chunk_id
        for chunk_id, rank in vector_ranks.items()
        if rank <= FUSION_POOL
    )

    def score(chunk_id: str) -> float:
        value = 0.0

        if chunk_id in bm25_ranks and bm25_ranks[chunk_id] <= FUSION_POOL:
            value += 1.0 / (RRF_K + bm25_ranks[chunk_id])

        if chunk_id in vector_ranks and vector_ranks[chunk_id] <= FUSION_POOL:
            value += 1.0 / (RRF_K + vector_ranks[chunk_id])

        return value

    return sorted(
        candidates,
        key=lambda chunk_id: (-score(chunk_id), chunk_id),
    )


def rerank_with_cross_encoder(
    query: str,
    candidate_ids: list[str],
    chunk_map: dict[str, dict],
    model: CrossEncoder,
) -> list[str]:
    """Rerank a shortlist of chunks using query-document relevance scores."""
    candidates = candidate_ids[:RERANK_CANDIDATE_POOL]

    if not candidates:
        return []

    pairs = [
        (query, searchable_text(chunk_map[chunk_id]))
        for chunk_id in candidates
    ]

    scores = model.predict(pairs, show_progress_bar=False)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: (-float(item[1]), item[0]),
    )

    return [chunk_id for chunk_id, _ in ranked]


def find_rank(
    ranked_ids: list[str],
    expected_id: str,
) -> int | None:
    """Return the one-based rank, or None if the row is absent."""
    try:
        return ranked_ids.index(expected_id) + 1
    except ValueError:
        return None


def hit_at(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0

    hits = sum(rank is not None and rank <= k for rank in ranks)
    return hits / len(ranks)


def mrr_at(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0

    reciprocal_sum = sum(
        1.0 / rank
        for rank in ranks
        if rank is not None and rank <= k
    )

    return reciprocal_sum / len(ranks)


def main() -> int:
    if not CHUNKS_DIR.exists():
        print(f"Chunks directory not found: {CHUNKS_DIR}")
        return 1

    if not VECTOR_DB.exists():
        print(f"Vector store not found: {VECTOR_DB}")
        return 1

    try:
        chunks = load_chunks()

        expected_chunks = {
            question["id"]: get_expected_chunk(question, chunks)
            for question in QUESTIONS
        }

        client = chromadb.PersistentClient(path=str(VECTOR_DB))
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
        )

        stored = collection.get(include=["metadatas"])
        vector_ids = set(stored["ids"])
        chunk_ids = {str(chunk["chunk_id"]) for chunk in chunks}

        if vector_ids != chunk_ids:
            raise ValueError(
                "Chunk corpus and vector index IDs do not match. "
                "Check the chunks and rebuild the vector index if "
                "the source chunks have changed."
            )

    except (OSError, ValueError, KeyError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1

    chunk_map = {
        str(chunk["chunk_id"]): chunk
        for chunk in chunks
    }

    # 1. BM25 retrieval.
    corpus = [
        tokenize(
            "Metric: " + str(chunk.get("title") or "") + "\n"
            + searchable_text(chunk)
        )
        for chunk in chunks
    ]
    bm25 = BM25Okapi(corpus)

    bm25_rankings: dict[str, list[str]] = {}

    for question in QUESTIONS:
        query_tokens = tokenize(question["query"])
        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(chunks)),
            key=lambda index: (
                -float(scores[index]),
                str(chunks[index]["chunk_id"]),
            ),
        )

        bm25_rankings[question["id"]] = [
            str(chunks[index]["chunk_id"])
            for index in ranked_indices
        ]

    # 2. Vector search using the existing Chroma index.
    query_texts = [question["query"] for question in QUESTIONS]

    print(f"Evaluation questions: {len(QUESTIONS)}")
    print(f"Shared corpus: {len(chunks)} chunks")
    print("Loading embedding model and embedding test queries...")

    try:
        embedding_model = SentenceTransformer(MODEL_NAME)

        query_embeddings = embedding_model.encode(
            query_texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        vector_results = collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=len(chunks),
            include=["distances"],
        )

    except Exception as exc:
        print(f"Vector search failed: {exc}", file=sys.stderr)
        return 1

    vector_rankings: dict[str, list[str]] = {}
    hybrid_rankings: dict[str, list[str]] = {}

    for index, question in enumerate(QUESTIONS):
        question_id = question["id"]
        vector_ids_for_query = vector_results["ids"][index]

        vector_rankings[question_id] = vector_ids_for_query

        bm25_ranks = {
            chunk_id: rank
            for rank, chunk_id in enumerate(
                bm25_rankings[question_id],
                start=1,
            )
        }

        vector_ranks = {
            chunk_id: rank
            for rank, chunk_id in enumerate(
                vector_ids_for_query,
                start=1,
            )
        }

        hybrid_rankings[question_id] = reciprocal_rank_fusion(
            bm25_ranks,
            vector_ranks,
        )

    # 3. Cross-encoder reranking from the shared production pipeline.
    print("\nRunning shared retrieval pipeline...")

    try:
        cross_encoder_rankings = {
            question["id"]: [
                str(result["chunk"]["chunk_id"])
                for result in retrieve(question["query"], rerank=True)
            ]
            for question in QUESTIONS
        }
    except Exception as exc:
        print(f"Shared retrieval failed: {exc}", file=sys.stderr)
        return 1

    # 4. Evaluate every method against the same expected source row.
    results = []

    print("\nPER-QUESTION RESULTS")
    print("-" * 130)
    print(
        f"{'ID':<5} "
        f"{'BM25':>8} "
        f"{'Vector':>8} "
        f"{'Hybrid RRF':>12} "
        f"{'Cross-enc.':>12}  "
        "Query"
    )
    print("-" * 130)

    for question in QUESTIONS:
        question_id = question["id"]
        expected_id = str(expected_chunks[question_id]["chunk_id"])

        bm25_rank = find_rank(bm25_rankings[question_id], expected_id)
        vector_rank = find_rank(vector_rankings[question_id], expected_id)
        hybrid_rank = find_rank(hybrid_rankings[question_id], expected_id)
        cross_encoder_rank = find_rank(
            cross_encoder_rankings[question_id],
            expected_id,
        )

        def format_rank(rank: int | None) -> str:
            return str(rank) if rank is not None else "not in pool"

        print(
            f"{question_id:<5} "
            f"{format_rank(bm25_rank):>8} "
            f"{format_rank(vector_rank):>8} "
            f"{format_rank(hybrid_rank):>12} "
            f"{format_rank(cross_encoder_rank):>12}  "
            f"{question['query']}"
        )

        results.append({
            "id": question_id,
            "query": question["query"],
            "expected_document": question["document"],
            "expected_page": question["page"],
            "expected_title": question["title"],
            "expected_chunk_id": expected_id,
            "bm25_rank": bm25_rank,
            "vector_rank": vector_rank,
            "hybrid_rank": hybrid_rank,
            "cross_encoder_rank": cross_encoder_rank,
        })

    # 5. Aggregate metrics.
    print("\nSUMMARY METRICS")
    print("-" * 70)
    print(
        f"{'Method':<18}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'MRR@10':>10}"
    )

    methods = ("bm25", "vector", "hybrid", "cross_encoder")

    for method in methods:
        ranks = [row[f"{method}_rank"] for row in results]

        print(
            f"{method:<18}"
            f"{hit_at(ranks, 1):>10.3f}"
            f"{hit_at(ranks, 3):>10.3f}"
            f"{hit_at(ranks, 5):>10.3f}"
            f"{hit_at(ranks, 10):>10.3f}"
            f"{mrr_at(ranks, 10):>10.3f}"
        )

    # 6. Save detailed rankings for later comparisons.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "retrieval_eval.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDetailed results saved to: {csv_path}")
    print(
        "\nBM25 uses financial vocabulary aliases. "
        "Vector search uses the existing embeddings."
    )
    print(
        f"Cross-encoder reranks at most {RERANK_CANDIDATE_POOL} "
        "candidates from the RRF ranking."
    )
    print(
        "These 15 questions are a preliminary benchmark. "
        "Verify the expected rows and expand the benchmark before "
        "treating the results as representative."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
