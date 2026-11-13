"""Dependency-aware Dense Retrieval using Roslyn semantic evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dense_retriever import (
    DEFAULT_BATCH_SIZE,
    _load_corpus,
    _load_or_encode,
)
from dependency_aware_bm25_retriever import (
    _dependency_entries,
    _dependency_paths,
    _envied_types,
    _load_json,
    _load_jsonl,
    _related,
)


def retrieve(
    *,
    corpus_path: str | Path,
    target_document_id: str,
    model_path: str | Path,
    cache_path: str | Path,
    dependency_analysis_path: str | Path,
    semantic_sidecar_path: str | Path,
    output_path: str | Path,
    top_k: int = 10,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cuda",
) -> dict[str, Any]:
    if top_k != 10:
        raise ValueError("Dependency-aware Dense Retrieval requires exactly top_k=10.")

    corpus = Path(corpus_path)
    documents = _load_corpus(corpus)
    targets = [
        document for document in documents
        if document["DocumentId"] == target_document_id
    ]
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one target document, found {len(targets)}.")
    target = targets[0]

    embeddings, cache_metadata, cache_status = _load_or_encode(
        documents=documents,
        corpus_path=corpus,
        model_path=Path(model_path),
        cache_path=Path(cache_path),
        batch_size=batch_size,
        device=device,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
        raise ValueError("Embedding cache does not match the corpus document count.")
    target_index = next(
        index for index, document in enumerate(documents)
        if document["DocumentId"] == target_document_id
    )
    target_embedding = embeddings[target_index]
    scores = embeddings @ target_embedding

    analysis = _load_json(Path(dependency_analysis_path))
    sidecar_documents = _load_jsonl(Path(semantic_sidecar_path))
    sidecar_by_id = {document["DocumentId"]: document for document in sidecar_documents}
    if set(document["DocumentId"] for document in documents) != set(sidecar_by_id):
        raise ValueError("Semantic sidecar IDs do not exactly match the corpus IDs.")

    entries = _dependency_entries(analysis)
    envied_types = _envied_types(analysis)
    dependency_paths = _dependency_paths(entries)
    target_method_symbols = {
        str(entry["Symbol"])
        for entry in analysis.get("Callees", [])
        if isinstance(entry, dict) and entry.get("Symbol")
    }
    target_member_symbols = {
        str(entry["Symbol"])
        for collection in ("Fields", "Properties")
        for entry in analysis.get(collection, [])
        if isinstance(entry, dict)
        and entry.get("Scope") == "FOREIGN"
        and entry.get("Symbol")
    }

    ranked: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        if document["DocumentId"] == target_document_id:
            continue
        category, reasons, evidence = _related(
            document,
            target=target,
            entries=entries,
            envied_types=envied_types,
            dependency_paths=dependency_paths,
            target_method_symbols=target_method_symbols,
            target_member_symbols=target_member_symbols,
            sidecar_by_id=sidecar_by_id,
        )
        ranked.append(
            {
                "Category": category,
                "Reasons": reasons,
                "DependencyEvidence": evidence,
                "Score": float(scores[index]),
                "DocumentId": document["DocumentId"],
                "Repository": document["Repository"],
                "File": document["File"],
                "ContainingType": document["ContainingType"],
                "MethodName": document["MethodName"],
                "Signature": document["Signature"],
                "StartLine": document["StartLine"],
                "EndLine": document["EndLine"],
            }
        )

    ranked.sort(key=lambda hit: (hit["Category"], -hit["Score"], hit["DocumentId"]))
    results = ranked[:top_k]
    for rank, result in enumerate(results, start=1):
        result["Rank"] = rank

    result = {
        "Strategy": "dependency-aware-dense-v1",
        "CorpusPath": str(corpus.resolve()),
        "DependencyAnalysisPath": str(Path(dependency_analysis_path).resolve()),
        "SemanticSidecarPath": str(Path(semantic_sidecar_path).resolve()),
        "CorpusDocumentCount": len(documents),
        "QueryTargetDocumentId": target_document_id,
        "QueryTargetSignature": target["Signature"],
        "ModelIdentifier": cache_metadata["ModelIdentifier"],
        "ModelFileFingerprint": cache_metadata["ModelFileFingerprint"],
        "InputFormat": cache_metadata["InputFormat"],
        "EmbeddingSettings": cache_metadata["EmbeddingSettings"],
        "CachePath": str(Path(cache_path).resolve()),
        "CacheStatus": cache_status,
        "TopK": top_k,
        "PriorityCategories": {
            "0": "direct semantic callee",
            "1": "exact shared foreign field/property symbol",
            "2": "method in Roslyn envied type",
            "3": "host or supporting dependency method",
            "4": "Dense cosine fallback",
        },
        "UnresolvedDependencies": analysis.get("UnresolvedDependencies", []),
        "Results": results,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--semantic-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = retrieve(
        corpus_path=args.corpus,
        target_document_id=args.target_id,
        model_path=args.model,
        cache_path=args.cache,
        dependency_analysis_path=args.dependencies,
        semantic_sidecar_path=args.semantic_sidecar,
        output_path=args.output,
        device=args.device,
    )
    print(f"CORPUS DOCUMENTS READ: {result['CorpusDocumentCount']}")
    print(f"CACHE: {result['CacheStatus']}")
    print(f"DEPENDENCY-AWARE DENSE RESULTS: {len(result['Results'])}")
    for hit in result["Results"]:
        print(
            f"{hit['Rank']}. {hit['ContainingType']}.{hit['MethodName']} "
            f"category={hit['Category']} score={hit['Score']:.15f}"
        )


if __name__ == "__main__":
    main()
