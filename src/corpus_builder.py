"""Build and validate a Python-orchestrated C# method corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


class CorpusBuilderError(RuntimeError):
    """Raised when corpus construction or validation fails."""


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise CorpusBuilderError(f"{label} does not exist or is not a file: {path}")


def _run_roslyn_corpus_builder(
    *,
    analyzer: Path,
    project: Path,
    repository_root: Path,
    repository: str,
    commit: str,
    msbuild: Path,
    output: Path,
) -> None:
    _require_file(analyzer, "Roslyn analyzer")
    _require_file(project, "C# project")
    if not repository_root.is_dir():
        raise CorpusBuilderError(f"Repository root does not exist: {repository_root}")
    if not msbuild.is_dir():
        raise CorpusBuilderError(f"MSBuild SDK path does not exist: {msbuild}")

    dotnet_root = msbuild.parent.parent if msbuild.parent.name.lower() == "sdk" else msbuild
    dotnet_host = dotnet_root / "dotnet.exe"
    if analyzer.suffix.lower() == ".dll":
        _require_file(dotnet_host, "local .NET host")
        command = [str(dotnet_host), str(analyzer)]
    else:
        command = [str(analyzer)]
    command.extend(
        [
            "--mode",
            "corpus",
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
    )

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
        raise CorpusBuilderError(
            "Roslyn corpus builder failed with exit code "
            f"{completed.returncode}.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    _require_file(output, "Roslyn corpus output")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                raise CorpusBuilderError(
                    f"Invalid JSON on corpus line {line_number}: {path}"
                ) from error
            if not isinstance(document, dict):
                raise CorpusBuilderError(f"Corpus line {line_number} is not an object.")
            documents.append(document)
    return documents


def _stable_document_id(document: dict[str, Any]) -> str:
    canonical = "\n".join(
        [
            document["Repository"],
            document["Commit"],
            document["File"].replace("\\", "/"),
            document["ContainingType"],
            document["Signature"],
        ]
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"cs-method-{digest}"


def _validate_corpus(
    *,
    corpus_path: Path,
    repository_root: Path,
    dependency_analysis: Path,
) -> tuple[int, int, int, dict[str, Any]]:
    documents = _load_jsonl(corpus_path)
    ids = [document.get("DocumentId", "") for document in documents]
    unique_ids = len(set(ids))
    duplicate_count = len(ids) - unique_ids
    if duplicate_count:
        raise CorpusBuilderError(f"Corpus contains {duplicate_count} duplicate IDs.")
    if any(document["DocumentId"] != _stable_document_id(document) for document in documents):
        raise CorpusBuilderError("At least one document ID is not deterministic.")
    if any(
        part.lower() in {"bin", "obj"}
        for document in documents
        for part in Path(document["File"]).parts
    ):
        raise CorpusBuilderError("Corpus contains a bin/obj source method.")

    with dependency_analysis.open("r", encoding="utf-8") as analysis_file:
        analysis = json.load(analysis_file)
    if not isinstance(analysis, dict):
        raise CorpusBuilderError("Dependency analysis must be a JSON object.")
    analysis_target = analysis.get("Target", {})
    repository_info = analysis.get("Repository", {})
    try:
        expected_file = Path(repository_info["SourceFile"]).relative_to(repository_root)
    except (KeyError, ValueError) as error:
        raise CorpusBuilderError(
            "Dependency analysis source file is not under the repository root."
        ) from error
    targets = [
        document
        for document in documents
        if document.get("ContainingType") == analysis_target.get("ContainingType")
        and document.get("MethodName") == analysis_target.get("MethodName")
        and document.get("File", "").replace("\\", "/")
        == str(expected_file).replace("\\", "/")
    ]
    if len(targets) != 1:
        raise CorpusBuilderError(
            "Expected one corpus document matching the dependency-analysis target, "
            f"found {len(targets)}."
        )
    target = targets[0]

    metadata_matches = (
        target["ContainingType"] == analysis_target["ContainingType"]
        and target["MethodName"] == analysis_target["MethodName"]
        and target["File"].replace("\\", "/") == str(expected_file).replace("\\", "/")
        and target["Signature"] == analysis_target["Signature"]
    )
    if not metadata_matches:
        raise CorpusBuilderError("Target corpus metadata did not match Roslyn analysis.")

    return len(documents), unique_ids, duplicate_count, target


def build_and_validate(args: argparse.Namespace) -> dict[str, Any]:
    first_run = args.output
    second_run = args.output.with_name(args.output.stem + "-second.jsonl")
    first_run.parent.mkdir(parents=True, exist_ok=True)
    _run_roslyn_corpus_builder(
        analyzer=args.analyzer,
        project=args.project,
        repository_root=args.repository_root,
        repository=args.repository,
        commit=args.commit,
        msbuild=args.msbuild,
        output=first_run,
    )
    first_documents = _load_jsonl(first_run)
    shutil.copyfile(first_run, second_run)
    _run_roslyn_corpus_builder(
        analyzer=args.analyzer,
        project=args.project,
        repository_root=args.repository_root,
        repository=args.repository,
        commit=args.commit,
        msbuild=args.msbuild,
        output=second_run,
    )
    second_documents = _load_jsonl(second_run)
    deterministic = [
        document["DocumentId"] for document in first_documents
    ] == [document["DocumentId"] for document in second_documents]
    if not deterministic:
        raise CorpusBuilderError("The second corpus run produced different document IDs.")

    count, unique_ids, duplicate_count, target = _validate_corpus(
        corpus_path=first_run,
        repository_root=args.repository_root,
        dependency_analysis=args.dependency_analysis,
    )
    return {
        "count": count,
        "unique_ids": unique_ids,
        "duplicate_count": duplicate_count,
        "target": target,
        "deterministic": deterministic,
        "second_run": second_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--msbuild", type=Path, required=True)
    parser.add_argument("--dependency-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_and_validate(args)
    print(f"TOTAL METHOD DOCUMENTS: {summary['count']}")
    print(f"UNIQUE DOCUMENT IDS: {summary['unique_ids']}")
    print(f"DUPLICATE IDS: {summary['duplicate_count']}")
    print("KNOWN TARGET FOUND: YES")
    print(f"TARGET DOCUMENT ID: {summary['target']['DocumentId']}")
    print(f"DETERMINISTIC SECOND RUN: {'PASS' if summary['deterministic'] else 'FAIL'}")
    print(f"CORPUS OUTPUT: {args.output.resolve()}")


if __name__ == "__main__":
    main()
