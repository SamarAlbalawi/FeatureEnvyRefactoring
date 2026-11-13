"""Hybrid retrieval using Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


RRF_K = 60


def _fuse(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for rank, hit in enumerate(first, start=1):
        document_id = hit["DocumentId"]
        by_id.setdefault(document_id, dict(hit))
        scores[document_id] = scores.get(document_id, 0.0) + 1 / (RRF_K + rank)
    for rank, hit in enumerate(second, start=1):
        document_id = hit["DocumentId"]
        by_id.setdefault(document_id, dict(hit))
        scores[document_id] = scores.get(document_id, 0.0) + 1 / (RRF_K + rank)

    fused = []
    for document_id, hit in by_id.items():
        hit["RRFScore"] = scores[document_id]
        hit["DocumentId"] = document_id
        fused.append(hit)
    fused.sort(key=lambda hit: (-hit["RRFScore"], hit["DocumentId"]))
    for rank, hit in enumerate(fused[:top_k], start=1):
        hit["Rank"] = rank
    return fused[:top_k]


def retrieve(
    *,
    bm25: Callable[[], dict[str, Any]],
    dense: Callable[[], dict[str, Any]],
    output_path: str | Path,
    target_document_id: str,
    top_k: int = 10,
    strategy: str = "hybrid-rrf-v1",
) -> dict[str, Any]:
    """Fuse two retrieval result sets while preserving target exclusion."""
    if top_k != 10:
        raise ValueError("Hybrid Retrieval requires exactly top_k=10.")
    bm25_result = bm25()
    dense_result = dense()
    bm25_hits = bm25_result.get("Results", [])
    dense_hits = dense_result.get("Results", [])
    results = _fuse(bm25_hits, dense_hits, top_k=top_k)
    if any(hit["DocumentId"] == target_document_id for hit in results):
        raise ValueError("Hybrid retrieval included the target document.")
    result = {
        "Strategy": strategy,
        "Fusion": {"Method": "RRF", "K": RRF_K},
        "QueryTargetDocumentId": target_document_id,
        "TopK": top_k,
        "BM25ResultPath": bm25_result.get("OutputPath"),
        "DenseResultPath": dense_result.get("OutputPath"),
        "Results": results,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-results", type=Path, required=True)
    parser.add_argument("--dense-results", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bm25 = json.loads(args.bm25_results.read_text(encoding="utf-8"))
    dense = json.loads(args.dense_results.read_text(encoding="utf-8"))
    result = retrieve(
        bm25=lambda: bm25,
        dense=lambda: dense,
        output_path=args.output,
        target_document_id=args.target_id,
    )
    print(f"HYBRID RESULTS: {len(result['Results'])}")
    print("TARGET EXCLUDED: YES")


if __name__ == "__main__":
    main()
