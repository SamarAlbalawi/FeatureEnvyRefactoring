"""Deterministic local-model dense retrieval over a JSONL method corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


INPUT_FORMAT = (
    "Document RetrievalText exactly as stored in the corpus; target uses the "
    "same RetrievalText field."
)
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = "cuda"


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
            if not isinstance(document.get("DocumentId"), str):
                raise ValueError(f"Corpus line {line_number} has no DocumentId.")
            if not isinstance(document.get("RetrievalText"), str):
                raise ValueError(f"Corpus line {line_number} has no RetrievalText.")
            documents.append(document)
    if not documents:
        raise ValueError("The corpus is empty.")
    return documents


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corpus_hash(path: Path) -> str:
    return _sha256_file(path)


def _model_fingerprint(model_path: Path) -> dict[str, Any]:
    files = []
    for file_path in sorted(path for path in model_path.rglob("*") if path.is_file()):
        files.append(
            {
                "Path": str(file_path.relative_to(model_path)),
                "Size": file_path.stat().st_size,
                "SHA256": _sha256_file(file_path),
            }
        )
    if not files:
        raise ValueError(f"No model files found under {model_path}.")
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"SHA256": digest, "Files": files}


def _cache_metadata(
    *,
    corpus_path: Path,
    model_path: Path,
    model_fingerprint: dict[str, Any],
    corpus_count: int,
    embedding_dimension: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    return {
        "SchemaVersion": "dense-embeddings-v1",
        "CorpusPath": str(corpus_path.resolve()),
        "CorpusSHA256": _corpus_hash(corpus_path),
        "CorpusDocumentCount": corpus_count,
        "ModelIdentifier": str(model_path.resolve()),
        "ModelFileFingerprint": model_fingerprint,
        "InputFormat": INPUT_FORMAT,
        "EmbeddingSettings": {
            "NormalizeEmbeddings": True,
            "BatchSize": batch_size,
            "Device": device,
            "EmbeddingDimension": embedding_dimension,
        },
    }


def _load_or_encode(
    *,
    documents: list[dict[str, Any]],
    corpus_path: Path,
    model_path: Path,
    cache_path: Path,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any], str]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local embedding model directory is missing: {model_path}")
    model_fingerprint = _model_fingerprint(model_path)
    corpus_hash = _corpus_hash(corpus_path)
    cache_metadata_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and cache_metadata_path.is_file():
        metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("CorpusSHA256") == corpus_hash
            and metadata.get("CorpusDocumentCount") == len(documents)
            and metadata.get("ModelFileFingerprint") == model_fingerprint
            and metadata.get("InputFormat") == INPUT_FORMAT
            and metadata.get("EmbeddingSettings", {}).get("BatchSize") == batch_size
            and metadata.get("EmbeddingSettings", {}).get("Device") == device
        ):
            embeddings = np.load(cache_path, allow_pickle=False)
            if embeddings.shape[0] == len(documents):
                return embeddings, metadata, "HIT"

    from sentence_transformers import SentenceTransformer

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model = SentenceTransformer(
        str(model_path),
        device=device,
        local_files_only=True,
    )
    embeddings = model.encode(
        [document["RetrievalText"] for document in documents],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)
    metadata = _cache_metadata(
        corpus_path=corpus_path,
        model_path=model_path,
        model_fingerprint=model_fingerprint,
        corpus_count=len(documents),
        embedding_dimension=int(embeddings.shape[1]),
        batch_size=batch_size,
        device=device,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    cache_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return embeddings, metadata, "MISS"


def retrieve(
    *,
    corpus_path: str | Path,
    target_document_id: str,
    model_path: str | Path,
    cache_path: str | Path,
    output_path: str | Path,
    top_k: int = 10,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
) -> dict[str, Any]:
    if top_k != 10:
        raise ValueError("Dense Retrieval requires exactly top_k=10.")
    corpus = Path(corpus_path)
    model_directory = Path(model_path)
    documents = _load_corpus(corpus)
    target_matches = [
        document for document in documents
        if document["DocumentId"] == target_document_id
    ]
    if len(target_matches) != 1:
        raise ValueError(f"Expected exactly one target document, found {len(target_matches)}.")
    target = target_matches[0]
    embeddings, cache_metadata, cache_status = _load_or_encode(
        documents=documents,
        corpus_path=corpus,
        model_path=model_directory,
        cache_path=Path(cache_path),
        batch_size=batch_size,
        device=device,
    )
    target_embedding = embeddings[documents.index(target)]
    scores = embeddings @ target_embedding
    scored: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        if document["DocumentId"] == target_document_id:
            continue
        scored.append(
            {
                "Rank": 0,
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
    scored.sort(key=lambda hit: (-hit["Score"], hit["DocumentId"]))
    results = scored[:top_k]
    for rank, hit in enumerate(results, start=1):
        hit["Rank"] = rank
    result = {
        "Retriever": "Dense",
        "CorpusPath": str(corpus.resolve()),
        "CorpusDocumentCount": len(documents),
        "QueryTargetDocumentId": target_document_id,
        "QueryTargetSignature": target["Signature"],
        "ModelIdentifier": cache_metadata["ModelIdentifier"],
        "ModelFileFingerprint": cache_metadata["ModelFileFingerprint"],
        "InputFormat": INPUT_FORMAT,
        "EmbeddingSettings": cache_metadata["EmbeddingSettings"],
        "CachePath": str(Path(cache_path).resolve()),
        "CacheStatus": cache_status,
        "TopK": top_k,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()
    result = retrieve(
        corpus_path=args.corpus,
        target_document_id=args.target_id,
        model_path=args.model,
        cache_path=args.cache,
        output_path=args.output,
        device=args.device,
    )
    print(f"CORPUS DOCUMENTS READ: {result['CorpusDocumentCount']}")
    print(f"DENSE CACHE: {result['CacheStatus']}")
    print("TARGET EXCLUDED: YES")
    for hit in result["Results"]:
        print(f"{hit['Rank']}. {hit['ContainingType']}.{hit['MethodName']} score={hit['Score']:.15f}")


if __name__ == "__main__":
    main()
