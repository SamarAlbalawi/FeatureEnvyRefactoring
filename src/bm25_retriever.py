"""Baseline lexical BM25 retrieval over a JSONL method corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


K1 = 1.5
B = 0.75
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
CAMEL_CASE_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text):
        for part in CAMEL_CASE_BOUNDARY.split(match.group(0)):
            if part.strip():
                tokens.append(part.lower())
    return tokens


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on corpus line {line_number}.") from error
            if not isinstance(document, dict):
                raise ValueError(f"Corpus line {line_number} is not an object.")
            documents.append(document)
    return documents


def _build_query(target: dict[str, Any]) -> str:
    return "\n".join(
        [
            target["Namespace"],
            target["ContainingType"],
            target["Signature"],
            target["SourceCode"],
        ]
    )


def _search(
    documents: list[dict[str, Any]],
    query: str,
    excluded_document_id: str,
    top_k: int,
) -> list[dict[str, Any]]:
    indexed: list[tuple[dict[str, Any], int, Counter[str]]] = []
    document_frequency: Counter[str] = Counter()
    for document in documents:
        terms = _tokenize(document["RetrievalText"])
        term_frequency = Counter(terms)
        indexed.append((document, len(terms), term_frequency))
        document_frequency.update(term_frequency.keys())

    average_length = (
        sum(length for _, length, _ in indexed) / len(indexed) if indexed else 0
    )
    query_terms = list(dict.fromkeys(_tokenize(query)))
    scored: list[dict[str, Any]] = []
    for document, length, term_frequency in indexed:
        if document["DocumentId"] == excluded_document_id:
            continue
        score = 0.0
        for term in query_terms:
            frequency = term_frequency.get(term)
            document_count = document_frequency.get(term)
            if not frequency or not document_count:
                continue
            inverse_document_frequency = math.log(
                1.0 + (len(indexed) - document_count + 0.5)
                / (document_count + 0.5)
            )
            denominator = frequency + K1 * (
                1.0 - B + B * length / average_length
            )
            score += inverse_document_frequency * (
                frequency * (K1 + 1.0) / denominator
            )
        scored.append(
            {
                "Rank": 0,
                "Score": score,
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

    scored.sort(key=lambda hit: (-hit["Score"], hit["DocumentId"]))
    for rank, hit in enumerate(scored[:top_k], start=1):
        hit["Rank"] = rank
    return scored[:top_k]


def retrieve(
    *,
    corpus_path: str | Path,
    target_document_id: str,
    output_path: str | Path,
    top_k: int = 10,
) -> dict[str, Any]:
    corpus = Path(corpus_path)
    output = Path(output_path)
    documents = _load_corpus(corpus)
    targets = [document for document in documents if document["DocumentId"] == target_document_id]
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one target document, found {len(targets)}.")
    target = targets[0]
    query = _build_query(target)
    results = _search(documents, query, target_document_id, top_k)
    result = {
        "CorpusPath": str(corpus.resolve()),
        "CorpusDocumentCount": len(documents),
        "QueryTargetDocumentId": target["DocumentId"],
        "QueryTargetSignature": target["Signature"],
        "QueryText": query,
        "K1": K1,
        "B": B,
        "TopK": top_k,
        "Results": results,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _identities(results: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(hit["ContainingType"], hit["MethodName"]) for hit in results]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = retrieve(
        corpus_path=args.corpus,
        target_document_id=args.target_id,
        output_path=args.output,
    )
    second_path = args.output.with_name(args.output.stem + "-second.json")
    second = retrieve(
        corpus_path=args.corpus,
        target_document_id=args.target_id,
        output_path=second_path,
    )
    deterministic = (
        _identities(result["Results"]) == _identities(second["Results"])
        and [hit["Score"] for hit in result["Results"]]
        == [hit["Score"] for hit in second["Results"]]
    )
    print(f"CORPUS DOCUMENTS READ: {result['CorpusDocumentCount']}")
    print("TARGET FOUND: YES")
    print(f"TARGET EXCLUDED: {'YES' if all(hit['DocumentId'] != args.target_id for hit in result['Results']) else 'NO'}")
    print(f"CANDIDATE DOCUMENTS: {result['CorpusDocumentCount'] - 1}")
    print(f"DETERMINISTIC SECOND RUN: {'PASS' if deterministic else 'FAIL'}")
    for hit in result["Results"]:
        print(
            f"{hit['Rank']}. {hit['ContainingType']}.{hit['MethodName']} "
            f"score={hit['Score']:.15f}"
        )


if __name__ == "__main__":
    main()
