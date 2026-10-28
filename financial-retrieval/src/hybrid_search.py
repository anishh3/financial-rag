
from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from retrieval import (
    CANDIDATE_POOL,
    RERANK_CANDIDATE_POOL,
    retrieve,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QuarterLens hybrid financial retrieval"
    )
    parser.add_argument("query", help="Financial question to search")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Inspect the RRF ranking without cross-encoder reranking",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    try:
        results = retrieve(
            args.query,
            rerank=not args.no_rerank,
        )

        print(f"\nQuery: {args.query}")
        print(f"BM25 / vector candidate pool: {CANDIDATE_POOL}")
        print(
            "Cross-encoder shortlist: "
            f"{0 if args.no_rerank else RERANK_CANDIDATE_POOL}"
        )
        print(f"Ranked candidates: {len(results)}")
        print(f"Displaying: {min(args.top_k, len(results))}")

        for rank, item in enumerate(results[:args.top_k], start=1):
            chunk = item["chunk"]

            print("\n" + "=" * 70)
            print(f"Rank: {rank}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Document: {chunk.get('source_document')}")
            print(f"Page: {chunk.get('source_page')}")
            print(f"Title: {chunk.get('title')}")
            print(f"RRF score: {item['rrf_score']:.6f}")
            print(f"BM25 rank: {item['bm25_rank'] or 'Outside BM25 pool'}")
            print(
                f"Vector rank: "
                f"{item['vector_rank'] or 'Outside vector pool'}"
            )

            if "cross_encoder_score" in item:
                print(
                    "Cross-encoder score: "
                    f"{item['cross_encoder_score']:.4f}"
                )

            print(f"\nEvidence:\n{chunk['content']}")

    except Exception as exc:
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
