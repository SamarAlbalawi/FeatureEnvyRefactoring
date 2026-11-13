"""Build and validate a Roslyn semantic dependency sidecar for the method corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


class SidecarError(RuntimeError):
    """Raised when sidecar construction or validation fails."""


def _stable_id(document: dict[str, Any]) -> str:
    canonical = "\n".join(
        [
            document["Repository"],
            document["Commit"],
            document["File"].replace("\\", "/"),
            document["ContainingType"],
            document["Signature"],
        ]
    )
    return "cs-method-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise SidecarError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise SidecarError(f"Expected an object at {path}:{line_number}")
            documents.append(value)
    return documents


def build_sidecar(
    *,
    analyzer: Path,
    project: Path,
    repository_root: Path,
    repository: str,
    commit: str,
    msbuild: Path,
    corpus: Path,
    output: Path,
) -> dict[str, Any]:
    for path, label in (
        (analyzer, "Roslyn analyzer"),
        (project, "C# project"),
        (repository_root, "repository root"),
        (msbuild, "MSBuild SDK"),
        (corpus, "method corpus"),
    ):
        if not path.exists():
            raise SidecarError(f"{label} does not exist: {path}")

    if msbuild.parent.name.lower() == "sdk":
        dotnet_root = msbuild.parent.parent
    elif msbuild.parent.parent.name.lower() == "sdk":
        dotnet_root = msbuild.parent.parent.parent
    else:
        dotnet_root = msbuild
    executable = (
        [str(dotnet_root / "dotnet.exe"), str(analyzer)]
        if analyzer.suffix.lower() == ".dll"
        else [str(analyzer)]
    )
    command = executable + [
        "--mode",
        "dependencies",
        "--project",
        str(project),
        "--repository-root",
        str(repository_root),
        "--repository",
        repository,
        "--commit",
        commit,
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment["ROSLYN_MSBUILD_PATH"] = str(msbuild)
    environment["DOTNET_ROOT"] = str(dotnet_root)
    environment["PATH"] = str(dotnet_root) + os.pathsep + environment.get("PATH", "")
    completed = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SidecarError(
            f"Roslyn sidecar build failed with exit code {completed.returncode}.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    corpus_documents = _load_jsonl(corpus)
    sidecar_documents = _load_jsonl(output)
    corpus_by_location: dict[tuple[str, str, int], dict[str, Any]] = {}
    for document in corpus_documents:
        key = (
            document["File"],
            document["ContainingType"],
            document["StartLine"],
        )
        if key in corpus_by_location:
            raise SidecarError(f"Corpus has duplicate method locations: {key}")
        corpus_by_location[key] = document

    for document in sidecar_documents:
        key = (
            document["File"],
            document["ContainingType"],
            document["StartLine"],
        )
        corpus_document = corpus_by_location.get(key)
        if corpus_document is None:
            raise SidecarError(f"Sidecar method does not match corpus location: {key}")
        document["DocumentId"] = corpus_document["DocumentId"]
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for document in sidecar_documents:
            stream.write(json.dumps(document, separators=(",", ":")) + "\n")

    corpus_by_id = {document["DocumentId"]: document for document in corpus_documents}
    sidecar_by_id = {document["DocumentId"]: document for document in sidecar_documents}
    if len(corpus_by_id) != len(corpus_documents):
        raise SidecarError("Corpus contains duplicate document IDs.")
    if len(sidecar_by_id) != len(sidecar_documents):
        raise SidecarError("Sidecar contains duplicate document IDs.")
    if set(corpus_by_id) != set(sidecar_by_id):
        raise SidecarError("Sidecar IDs do not exactly match the corpus IDs.")
    if any(_stable_id(document) != document["DocumentId"] for document in corpus_documents):
        raise SidecarError("Existing corpus contains a non-deterministic document ID.")

    return {
        "CorpusDocuments": len(corpus_documents),
        "SidecarDocuments": len(sidecar_documents),
        "UniqueSidecarIds": len(sidecar_by_id),
        "IdsMatchCorpus": set(corpus_by_id) == set(sidecar_by_id),
        "SidecarPath": str(output.resolve()),
        "RoslynStdout": completed.stdout,
        "RoslynStderr": completed.stderr,
        "Documents": sidecar_documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--msbuild", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_sidecar(**vars(args))
    print(f"CORPUS DOCUMENTS: {result['CorpusDocuments']}")
    print(f"SIDECAR DOCUMENTS: {result['SidecarDocuments']}")
    print(f"IDS MATCH CORPUS: {'YES' if result['IdsMatchCorpus'] else 'NO'}")
    print(f"SIDECAR: {result['SidecarPath']}")


if __name__ == "__main__":
    main()
