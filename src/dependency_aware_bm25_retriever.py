"""Dependency-aware BM25 retrieval using Roslyn semantic evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from bm25_retriever import _build_query, _load_corpus, _search


_DEPENDENCY_COLLECTIONS = (
    "Callees",
    "Fields",
    "Properties",
    "ReferencedTypes",
    "Interfaces",
    "LocalDependencies",
    "ForeignDependencies",
)


def _normalise_path(value: str) -> str:
    return value.replace("\\", "/").lower()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Unable to read JSONL: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        documents.append(value)
    return documents


def _dependency_entries(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for collection in _DEPENDENCY_COLLECTIONS:
        values = analysis.get(collection, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, dict):
                continue
            key = (
                str(value.get("Relationship", "")),
                str(value.get("Symbol", "")),
                str(value.get("DeclaringType", "")),
            )
            if key not in seen:
                seen.add(key)
                entries.append(value)
    return entries


def _symbol_method(symbol: str) -> str:
    return symbol.split("(", 1)[0].rsplit(".", 1)[-1]


def _dependency_paths(entries: Iterable[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for entry in entries:
        location = entry.get("Location")
        if isinstance(location, dict) and isinstance(location.get("File"), str):
            paths.add(_normalise_path(location["File"]))
    return paths


def _envied_types(analysis: dict[str, Any]) -> set[str]:
    types: set[str] = set()
    candidates = analysis.get("LikelyEnviedTypeCandidates", [])
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and isinstance(candidate.get("DeclaringType"), str):
                types.add(candidate["DeclaringType"])
    return types


def _related(
    document: dict[str, Any],
    *,
    target: dict[str, Any],
    entries: list[dict[str, Any]],
    envied_types: set[str],
    dependency_paths: set[str],
    target_method_symbols: set[str],
    target_member_symbols: set[str],
    sidecar_by_id: dict[str, dict[str, Any]],
) -> tuple[int, list[str], dict[str, list[dict[str, str]]]]:
    reasons: list[str] = []
    evidence: dict[str, list[dict[str, str]]] = {
        "DirectMethodRelationships": [],
        "VerifiedMemberRelationships": [],
        "ExactSharedMemberRelationships": [],
        "EnviedTypeAssociations": [],
    }
    if document["Signature"] in target_method_symbols:
        reasons.append("direct semantic callee")
        evidence["DirectMethodRelationships"].append(
            {
                "relationship": "CALL",
                "symbol": document["Signature"],
                "declaring_type": document["ContainingType"],
            }
        )

    sidecar = sidecar_by_id.get(document["DocumentId"], {})
    candidate_member_symbols = {
        str(reference.get("Symbol"))
        for collection in ("FieldAccesses", "PropertyAccesses")
        for reference in sidecar.get(collection, [])
        if isinstance(reference, dict) and reference.get("Symbol")
    }
    shared_symbols = sorted(candidate_member_symbols & target_member_symbols)
    for symbol in shared_symbols:
        evidence["ExactSharedMemberRelationships"].append(
            {
                "relationship": (
                    "FIELD_ACCESS"
                    if any(
                        reference.get("Symbol") == symbol
                        for reference in sidecar.get("FieldAccesses", [])
                        if isinstance(reference, dict)
                    )
                    else "PROPERTY_ACCESS"
                ),
                "symbol": symbol,
                "declaring_type": symbol.rsplit(".", 1)[0],
                "verification": "Roslyn sidecar symbol intersection",
            }
        )
    if shared_symbols:
        reasons.append("exact shared foreign member symbol")

    document_path = _normalise_path(document["File"])
    if any(path.endswith(document_path) for path in dependency_paths):
        reasons.append("method in supporting dependency source file")
    if document["ContainingType"] in envied_types:
        reasons.append("method in Roslyn envied type")
        evidence["EnviedTypeAssociations"].append(
            {
                "association": "method containing type is a Roslyn envied type",
                "declaring_type": document["ContainingType"],
            }
        )
    if document["ContainingType"] == target["ContainingType"]:
        reasons.append("method in target host type")
    if document["Signature"] in target_method_symbols:
        category = 0
    elif shared_symbols:
        category = 1
    elif "method in Roslyn envied type" in reasons:
        category = 2
    elif (
        "method in target host type" in reasons
        or "method in supporting dependency source file" in reasons
    ):
        category = 3
    else:
        category = 4
        reasons.append("lexical BM25 only")
    return category, sorted(set(reasons)), evidence


def retrieve(
    *,
    corpus_path: str | Path,
    target_document_id: str,
    dependency_analysis_path: str | Path,
    semantic_sidecar_path: str | Path,
    output_path: str | Path,
    top_k: int = 10,
) -> dict[str, Any]:
    corpus = Path(corpus_path)
    analysis = _load_json(Path(dependency_analysis_path))
    sidecar_documents = _load_jsonl(Path(semantic_sidecar_path))
    sidecar_by_id = {document["DocumentId"]: document for document in sidecar_documents}
    documents = _load_corpus(corpus)
    targets = [doc for doc in documents if doc["DocumentId"] == target_document_id]
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one target document, found {len(targets)}.")
    target = targets[0]
    baseline_hits = _search(
        documents, _build_query(target), target_document_id, len(documents)
    )
    scores = {hit["DocumentId"]: hit["Score"] for hit in baseline_hits}
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
    if set(document["DocumentId"] for document in documents) != set(sidecar_by_id):
        raise ValueError("Semantic sidecar IDs do not exactly match the corpus IDs.")
    ranked: list[dict[str, Any]] = []
    for document in documents:
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
                "Score": scores[document["DocumentId"]],
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
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "Strategy": "dependency-aware-bm25-v3",
        "CorpusPath": str(corpus.resolve()),
        "DependencyAnalysisPath": str(Path(dependency_analysis_path).resolve()),
        "SemanticSidecarPath": str(Path(semantic_sidecar_path).resolve()),
        "CorpusDocumentCount": len(documents),
        "QueryTargetDocumentId": target_document_id,
        "QueryTargetSignature": target["Signature"],
        "TopK": top_k,
        "BM25Parameters": {"K1": 1.5, "B": 0.75},
        "PriorityCategories": {
            "0": "direct semantic callee",
            "1": "exact shared foreign field/property symbol",
            "2": "method in Roslyn envied type",
            "3": "host or supporting dependency method",
            "4": "lexical BM25 only",
        },
        "UnresolvedDependencies": analysis.get("UnresolvedDependencies", []),
        "Results": results,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--semantic-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = retrieve(
        corpus_path=args.corpus,
        target_document_id=args.target_id,
        dependency_analysis_path=args.dependencies,
        semantic_sidecar_path=args.semantic_sidecar,
        output_path=args.output,
    )
    print(f"CORPUS DOCUMENTS READ: {result['CorpusDocumentCount']}")
    print(f"DEPENDENCY-AWARE RESULTS: {len(result['Results'])}")
    for hit in result["Results"]:
        print(
            f"{hit['Rank']}. {hit['ContainingType']}.{hit['MethodName']} "
            f"category={hit['Category']} score={hit['Score']:.15f}"
        )


if __name__ == "__main__":
    main()
