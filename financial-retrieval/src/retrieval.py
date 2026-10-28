
from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


ROOT = Path.cwd()

CHUNKS_DIR = ROOT / "financial-retrieval" / "artifacts" / "chunks"
VECTOR_DB = (
    ROOT
    / "financial-retrieval"
    / "artifacts"
    / "vector_store"
    / "chroma"
)

COLLECTION_NAME = "quarterlens_chunks"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60
CANDIDATE_POOL = 100
RERANK_CANDIDATE_POOL = 150

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


def tokenize(text: str) -> list[str]:
    """Normalize common financial terms for BM25."""
    tokens = re.findall(
        r"[a-z]+|\d+(?:[,.]\d+)*",
        text.casefold(),
    )

    return [
        ALIASES.get(token, token)
        for token in tokens
        if token not in STOPWORDS
    ]


def source_identity(source_document: object) -> str:
    """Turn quarterly filenames into searchable fiscal-source identity."""
    stem = Path(str(source_document or "")).stem.casefold()

    quarter_match = re.fullmatch(r"q([1-4])-20(\d{2})", stem)
    if quarter_match:
        quarter, short_year = quarter_match.groups()
        return (
            f"{stem} q{quarter} fy20{short_year} "
            f"quarter {quarter} fiscal year 20{short_year}"
        )

    if stem == "additional-information-2025-26":
        return (
            "additional information 2025 26 annual fy2026 fiscal year 2026 "
            "financial year 2026 full year 2026"
        )

    return stem.replace("-", " ")


def retrieval_text(chunk: dict) -> str:
    """Build the consistent field-focused representation used for ranking."""
    fields = []

    def add(label: str, value: object) -> None:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if text:
            fields.append(f"{label}: {text}" if label else text)

    title = str(chunk.get("title") or "")
    add("Metric", title)
    add("Company", "Infosys Limited")
    add("Table", chunk.get("table_title"))

    section_path = chunk.get("section_path")
    if isinstance(section_path, list):
        add("Section", " > ".join(str(part) for part in section_path if part))
    else:
        add("Section", chunk.get("section_context") or section_path)

    add("Source", source_identity(chunk.get("source_document")))

    headers = chunk.get("headers")
    if isinstance(headers, list):
        add("Columns", " | ".join(str(header) for header in headers if header))

    add("Unit", chunk.get("normalized_unit") or chunk.get("unit"))

    values = chunk.get("values")
    if isinstance(values, dict):
        for period, value in values.items():
            add("Value", f"{period}: {value}")

    # Prose chunks have no structured row fields to extract, so retain concise
    # content while avoiding table rows' broad Document context block.
    if chunk.get("record_type") != "table_row":
        add("", chunk.get("content"))

    return "\n".join(fields)


# Backwards-compatible internal name used by existing ranking code.
searchable_text = retrieval_text


def load_chunks() -> list[dict]:
    """Load the same chunks used to build the vector index."""
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
                        f"Invalid JSON in {file_path}, "
                        f"line {line_number}"
                    ) from exc

                chunk_id = str(chunk.get("chunk_id") or "").strip()
                content = str(chunk.get("content") or "").strip()
                title = str(chunk.get("title") or "").strip()

                if not chunk_id or not content:
                    continue

                # Match the existing vector-index exclusion rule.
                if (
                    chunk.get("record_type") == "text"
                    and title
                    and content.casefold() == title.casefold()
                ):
                    continue

                chunks.append(chunk)

    ids = [str(chunk["chunk_id"]) for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs found.")

    return sorted(
        chunks,
        key=lambda chunk: str(chunk["chunk_id"]),
    )


def reciprocal_rank_fusion(
    bm25_ranks: dict[str, int],
    vector_ranks: dict[str, int],
    chunk_map: dict[str, dict],
    bm25_scores: dict[str, float],
) -> list[dict]:
    """Combine BM25 and vector rankings using RRF."""
    candidate_ids = {
        chunk_id
        for chunk_id, rank in bm25_ranks.items()
        if rank <= CANDIDATE_POOL
    }

    candidate_ids.update(
        chunk_id
        for chunk_id, rank in vector_ranks.items()
        if rank <= CANDIDATE_POOL
    )

    fused = []

    for chunk_id in candidate_ids:
        bm25_rank = bm25_ranks.get(chunk_id)
        vector_rank = vector_ranks.get(chunk_id)

        rrf_score = 0.0

        if bm25_rank is not None and bm25_rank <= CANDIDATE_POOL:
            rrf_score += 1.0 / (RRF_K + bm25_rank)

        if vector_rank is not None and vector_rank <= CANDIDATE_POOL:
            rrf_score += 1.0 / (RRF_K + vector_rank)

        fused.append({
            "chunk": chunk_map[chunk_id],
            "rrf_score": rrf_score,
            "bm25_rank": bm25_rank,
            "vector_rank": vector_rank,
            "bm25_score": bm25_scores.get(chunk_id, 0.0),
        })

    fused.sort(
        key=lambda item: (
            -item["rrf_score"],
            str(item["chunk"]["chunk_id"]),
        )
    )

    return fused


def rerank_with_cross_encoder(
    query: str,
    candidates: list[dict],
    model: CrossEncoder,
) -> list[dict]:
    """Rerank a shortlist using query-chunk relevance scores."""
    shortlist = candidates[:RERANK_CANDIDATE_POOL]

    if not shortlist:
        return candidates

    pairs = [
        (query, searchable_text(item["chunk"]))
        for item in shortlist
    ]

    scores = model.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked = []

    for item, score in zip(shortlist, scores):
        updated = dict(item)
        updated["cross_encoder_score"] = float(score)
        reranked.append(updated)

    reranked.sort(
        key=lambda item: (
            -item["cross_encoder_score"],
            str(item["chunk"]["chunk_id"]),
        )
    )

    # Preserve candidates beyond the reranking shortlist,
    # in their original RRF order.
    return reranked + candidates[len(shortlist):]


def retrieve(
    query: str,
    *,
    rerank: bool = True,
) -> list[dict]:
    """
    Retrieve and rank financial evidence.

    Pipeline:
    BM25 + vector search -> RRF -> cross-encoder.

    Returns ranked result dictionaries containing each original
    chunk and the retrieval scores. The full candidate ranking is
    returned; callers decide how many chunks to use.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    chunks = load_chunks()

    if not VECTOR_DB.exists():
        raise FileNotFoundError(
            f"Vector store not found: {VECTOR_DB}"
        )

    chunk_map = {
        str(chunk["chunk_id"]): chunk
        for chunk in chunks
    }

    client = chromadb.PersistentClient(path=str(VECTOR_DB))
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    indexed_ids = set(
        collection.get(include=["metadatas"])["ids"]
    )

    if set(chunk_map) != indexed_ids:
        raise ValueError(
            "Chunk files and vector index do not match. "
            "Check whether the source chunks changed."
        )

    # 1. BM25
    corpus = [
        tokenize(
            "Metric: " + str(chunk.get("title") or "") + "\n"
            + searchable_text(chunk)
        )
        for chunk in chunks
    ]

    query_tokens = tokenize(query)

    if not query_tokens:
        raise ValueError("Query contains no searchable terms.")

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)

    bm25_order = sorted(
        range(len(scores)),
        key=lambda index: (
            -float(scores[index]),
            str(chunks[index]["chunk_id"]),
        ),
    )

    bm25_ranks = {
        str(chunks[index]["chunk_id"]): rank
        for rank, index in enumerate(bm25_order, start=1)
    }

    bm25_scores = {
        str(chunks[index]["chunk_id"]): float(scores[index])
        for index in range(len(chunks))
    }

    # 2. Semantic vector search using the existing index
    embedding_model = SentenceTransformer(MODEL_NAME)

    query_vector = embedding_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    vector_results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(CANDIDATE_POOL, len(chunks)),
        include=["distances"],
    )

    vector_ids = vector_results["ids"][0]

    vector_ranks = {
        chunk_id: rank
        for rank, chunk_id in enumerate(vector_ids, start=1)
    }

    # 3. RRF
    fused = reciprocal_rank_fusion(
        bm25_ranks=bm25_ranks,
        vector_ranks=vector_ranks,
        chunk_map=chunk_map,
        bm25_scores=bm25_scores,
    )

    # 4. Cross-encoder reranking
    if rerank:
        reranker = CrossEncoder(RERANK_MODEL_NAME)
        fused = rerank_with_cross_encoder(
            query=query,
            candidates=fused,
            model=reranker,
        )

    return fused
