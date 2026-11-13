"""Build repository-wide corpus and dependency indexes from verified local sources."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from analysis_project import ensure_source_project


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _projects_for_cases(root: Path, cases: list[dict[str, Any]]) -> list[Path]:
    projects = [
        path
        for path in root.rglob("*.csproj")
        if "bin" not in path.parts and "obj" not in path.parts
    ]
    selected: set[Path] = set()
    for case in cases:
        source = root.joinpath(*Path(case["File"]).parts)
        matches = [project for project in projects if source.is_relative_to(project.parent)]
        if matches:
            selected.add(max(matches, key=lambda project: len(project.parts)))
    return sorted(selected, key=lambda path: str(path).lower())


def _run_analyzer(
    *,
    analyzer: Path,
    project: Path,
    root: Path,
    repository: str,
    commit: str,
    msbuild: Path,
    mode: str,
    output: Path,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
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
        mode,
        "--project",
        str(project),
        "--repository-root",
        str(root),
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
    log = {
        "Command": command,
        "ExitCode": completed.returncode,
        "Stdout": completed.stdout,
        "Stderr": completed.stderr,
    }
    (output.with_suffix(output.suffix + ".log.json")).write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{mode} analyzer failed for {project}: {completed.stderr}")
    return log


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path}")
            documents.append(value)
    return documents


def _write_jsonl(path: Path, documents: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(document, separators=(",", ":")) + "\n" for document in documents),
        encoding="utf-8",
    )


def _deduplicate_documents(
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_id: dict[str, dict[str, Any]] = {}
    conflicts: list[str] = []
    for document in documents:
        document_id = document["DocumentId"]
        previous = by_id.get(document_id)
        if previous is None:
            by_id[document_id] = document
        elif {
            key: value for key, value in previous.items() if key != "Project"
        } != {
            key: value for key, value in document.items() if key != "Project"
        }:
            conflicts.append(document_id)
    return list(by_id.values()), conflicts


def prepare(
    *,
    dataset: Path,
    roots: dict[str, Path],
    analyzer: Path,
    msbuild: Path,
    output: Path,
    repository_filter: set[str] | None = None,
) -> dict[str, Any]:
    from dataset_manifest import build_manifest

    manifest = build_manifest(dataset)
    output.mkdir(parents=True, exist_ok=True)
    by_repository: dict[str, list[dict[str, Any]]] = {}
    for case in manifest["Cases"]:
        by_repository.setdefault(case["Repository"], []).append(case)

    repositories: list[dict[str, Any]] = []
    for revision in manifest["RepositoryRevisions"]:
        repository = revision["Repository"]
        if repository_filter is not None and repository not in repository_filter:
            continue
        root = roots.get(repository)
        record: dict[str, Any] = {
            "Repository": repository,
            "Commit": revision["Commit"],
            "CaseCount": len(by_repository[repository]),
        }
        if root is None or not root.is_dir():
            record.update({"Status": "BLOCKED", "Category": "REPOSITORY_UNAVAILABLE"})
            repositories.append(record)
            continue
        projects = _projects_for_cases(root, by_repository[repository])
        if repository == "files-community/Files" or not projects:
            projects = [
                ensure_source_project(
                    repository=repository,
                    root=root,
                    output=output / "analysis-projects",
                )
            ]
        record["Projects"] = [str(project) for project in projects]

        repository_output = output / repository.replace("/", "__")
        corpus_parts: list[dict[str, Any]] = []
        dependency_parts: list[dict[str, Any]] = []
        project_records: list[dict[str, Any]] = []
        for index, project in enumerate(projects, start=1):
            project_output = repository_output / f"project-{index:03d}"
            corpus_part = project_output / "corpus.jsonl"
            dependency_part = project_output / "dependencies.jsonl"
            corpus_log = _run_analyzer(
                analyzer=analyzer,
                project=project,
                root=root,
                repository=repository,
                commit=revision["Commit"],
                msbuild=msbuild,
                mode="corpus",
                output=corpus_part,
            )
            dependency_log = _run_analyzer(
                analyzer=analyzer,
                project=project,
                root=root,
                repository=repository,
                commit=revision["Commit"],
                msbuild=msbuild,
                mode="dependencies",
                output=dependency_part,
            )
            corpus_part_documents = _read_jsonl(corpus_part)
            dependency_part_documents = _read_jsonl(dependency_part)
            project_records.append(
                {
                    "Project": str(project),
                    "CorpusDocuments": len(corpus_part_documents),
                    "DependencyDocuments": len(dependency_part_documents),
                    "CorpusExitCode": corpus_log["ExitCode"],
                    "DependencyExitCode": dependency_log["ExitCode"],
                }
            )
            corpus_parts.extend(corpus_part_documents)
            dependency_parts.extend(dependency_part_documents)

        corpus_parts, corpus_conflicts = _deduplicate_documents(corpus_parts)
        dependency_parts, dependency_conflicts = _deduplicate_documents(dependency_parts)
        corpus_ids = {document["DocumentId"] for document in corpus_parts}
        dependency_ids = {document["DocumentId"] for document in dependency_parts}
        if corpus_conflicts:
            record.update(
                {
                    "Status": "BLOCKED",
                    "Category": "CONFLICTING_CORPUS_DOCUMENT_ID",
                    "ConflictingDocumentIds": corpus_conflicts,
                }
            )
        elif dependency_conflicts:
            record.update(
                {
                    "Status": "BLOCKED",
                    "Category": "CONFLICTING_DEPENDENCY_DOCUMENT_ID",
                    "ConflictingDocumentIds": dependency_conflicts,
                }
            )
        elif corpus_ids != dependency_ids:
            record.update({"Status": "BLOCKED", "Category": "CORPUS_DEPENDENCY_ID_MISMATCH"})
        else:
            corpus_path = repository_output / "repository-corpus.jsonl"
            dependency_path = repository_output / "repository-dependencies.jsonl"
            _write_jsonl(corpus_path, sorted(corpus_parts, key=lambda d: d["DocumentId"]))
            _write_jsonl(dependency_path, sorted(dependency_parts, key=lambda d: d["DocumentId"]))
            record.update(
                {
                    "Status": "READY",
                    "CorpusPath": str(corpus_path.resolve()),
                    "DependencyPath": str(dependency_path.resolve()),
                    "CorpusDocuments": len(corpus_parts),
                    "DependencyDocuments": len(dependency_parts),
                }
            )
        record["ProjectRecords"] = project_records
        repositories.append(record)

    result = {
        "SchemaVersion": "repository-index-preparation-v1",
        "Repositories": repositories,
        "ReadyRepositories": sum(item["Status"] == "READY" for item in repositories),
        "BlockedRepositories": sum(item["Status"] == "BLOCKED" for item in repositories),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        repository, path = value.split("=", 1)
        result[repository] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repository-root", action="append", default=[])
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--msbuild", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", action="append", default=[])
    args = parser.parse_args()
    result = prepare(
        dataset=args.dataset,
        roots=_mapping(args.repository_root),
        analyzer=args.analyzer,
        msbuild=args.msbuild,
        output=args.output,
        repository_filter=set(args.repository) if args.repository else None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
