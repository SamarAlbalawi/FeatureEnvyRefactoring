"""Run resumable, local-only dataset analysis and retrieval experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import traceback
from typing import Any

from analyze_dataset_case import (
    _find_local_source,
    _parse_target_identifier,
)
from analysis_project import ensure_source_project
from dataset_manifest import build_manifest
from feature_envy_metrics import evaluate_file
from roslyn_bridge import invoke_roslyn
from repository_preparer import materialize_archive, write_preparation_record


@dataclass(frozen=True)
class RepositoryConfig:
    repository: str
    root: Path | None
    revision_metadata: Path | None


def _parse_mapping(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use repository=path form: {value}")
        repository, path = value.split("=", 1)
        if not repository or not path:
            raise ValueError(f"{label} contains an empty repository or path: {value}")
        result[repository] = Path(path)
    return result


def _resolve_index_inputs(
    values: list[str],
) -> tuple[dict[str, Path], dict[str, Path]]:
    corpora: dict[str, Path] = {}
    sidecars: dict[str, Path] = {}
    for value in values:
        if "=" in value:
            repository, summary_path = value.split("=", 1)
            summaries = [_load_json(Path(summary_path))]
        else:
            repository = None
            summary_path = value
            summaries = [
                _load_json(path)
                for path in sorted(
                    Path(summary_path).glob("repository-index*/summary.json")
                )
            ]
            if not summaries:
                summaries = [_load_json(Path(summary_path))]
        records = [
            item
            for summary in summaries
            for item in summary.get("Repositories", [])
        ]
        if repository is None:
            for item in records:
                if item.get("Status") != "READY":
                    continue
                repository = item["Repository"]
                corpora[repository] = Path(item["CorpusPath"])
                sidecars[repository] = Path(
                    item.get("SemanticSidecarPath", item["DependencyPath"])
                )
            continue
        repositories = summaries[0].get("Repositories", [])
        record = next(
            item for item in repositories if item.get("Repository") == repository
        )
        if record.get("Status") != "READY":
            raise ValueError(f"Repository index is not ready: {repository}")
        corpora[repository] = Path(record["CorpusPath"])
        sidecars[repository] = Path(
            record.get("SemanticSidecarPath", record["DependencyPath"])
        )
    return corpora, sidecars


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _verify_revision(
    repository: str,
    commit: str,
    root: Path | None,
    metadata: Path | None,
) -> dict[str, Any]:
    if root is None or not root.is_dir():
        return {
            "status": "BLOCKED",
            "category": "REPOSITORY_UNAVAILABLE",
            "message": str(root) if root is not None else "No repository root was supplied.",
        }
    if metadata is None or not metadata.is_file():
        return {
            "status": "BLOCKED",
            "category": "REVISION_UNVERIFIED",
            "message": "No commit metadata or Git checkout was available.",
        }
    try:
        value = _load_json(metadata)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "BLOCKED",
            "category": "REVISION_METADATA_INVALID",
            "message": str(error),
        }
    verified_sha = value.get("sha")
    if verified_sha is None and isinstance(value.get("Records"), list):
        for item in value["Records"]:
            if item.get("Repository") == repository:
                verified_sha = item.get("VerifiedCommit")
                break
    if verified_sha != commit:
        return {
            "status": "BLOCKED",
            "category": "REVISION_MISMATCH",
            "message": f"Metadata sha {verified_sha} does not match {commit}.",
        }
    return {
        "status": "READY",
        "category": "REVISION_VERIFIED",
        "message": f"{repository}@{commit}",
        "metadata": str(metadata.resolve()),
    }


def _find_project(root: Path, source: Path, repository: str, project_output: Path) -> Path:
    candidates = [
        path
        for path in root.rglob("*.csproj")
        if "bin" not in path.parts and "obj" not in path.parts
    ]
    containing = [
        path for path in candidates
        if source.is_relative_to(path.parent)
    ]
    if containing and repository != "files-community/Files":
        return min(containing, key=lambda path: len(path.parts))
    if candidates and repository != "files-community/Files":
        return min(candidates, key=lambda path: len(path.parts))
    return ensure_source_project(
        repository=repository,
        root=root,
        output=project_output,
    )


def _find_target_id(corpus: Path, case: dict[str, Any]) -> str:
    containing_type, method_name = _parse_target_identifier(case["CaseId"])
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        document = json.loads(line)
        if (
            document.get("ContainingType") == containing_type
            and document.get("MethodName") == method_name
            and document.get("File", "").replace("\\", "/")
            == case["File"].replace("\\", "/")
        ):
            return document["DocumentId"]
    raise ValueError("Target method is absent from the repository corpus.")


def _run_retrieval(
    *,
    case: dict[str, Any],
    analysis_path: Path,
    root: Path,
    corpus: Path,
    model: Path | None,
    semantic_sidecar: Path | None,
    output: Path,
    device: str,
) -> dict[str, Any]:
    if (
        model is None
        or semantic_sidecar is None
        or not model.is_dir()
        or not semantic_sidecar.is_file()
        or not corpus.is_file()
    ):
        return {
            "status": "BLOCKED",
            "category": "RETRIEVAL_INPUT_MISSING",
            "message": (
                "Dense model or semantic sidecar is unavailable: "
                f"model={model}, corpus={corpus}, sidecar={semantic_sidecar}"
            ),
        }
    from bm25_retriever import retrieve as bm25
    from dense_retriever import retrieve as dense
    from dependency_aware_bm25_retriever import retrieve as dependency_bm25
    from dependency_aware_dense_retriever import retrieve as dependency_dense
    from hybrid_retriever import retrieve as hybrid

    target_id = _find_target_id(corpus, case)
    output.mkdir(parents=True, exist_ok=True)
    bm25_result = bm25(
        corpus_path=corpus,
        target_document_id=target_id,
        output_path=output / "bm25.json",
    )
    dense_result = dense(
        corpus_path=corpus,
        target_document_id=target_id,
        model_path=model,
        cache_path=output / "dense-embeddings.npy",
        output_path=output / "dense.json",
        device=device,
    )
    dependency_bm25_result = dependency_bm25(
        corpus_path=corpus,
        target_document_id=target_id,
        dependency_analysis_path=analysis_path,
        semantic_sidecar_path=semantic_sidecar,
        output_path=output / "dependency-aware-bm25.json",
    )
    dependency_dense_result = dependency_dense(
        corpus_path=corpus,
        target_document_id=target_id,
        model_path=model,
        cache_path=output / "dense-embeddings.npy",
        dependency_analysis_path=analysis_path,
        semantic_sidecar_path=semantic_sidecar,
        output_path=output / "dependency-aware-dense.json",
        device=device,
    )
    baseline_hybrid = hybrid(
        bm25=lambda: bm25_result,
        dense=lambda: dense_result,
        output_path=output / "hybrid.json",
        target_document_id=target_id,
    )
    dependency_hybrid = hybrid(
        bm25=lambda: dependency_bm25_result,
        dense=lambda: dependency_dense_result,
        output_path=output / "dependency-aware-hybrid.json",
        target_document_id=target_id,
        strategy="dependency-aware-hybrid-rrf-v1",
    )
    return {
        "status": "PASS",
        "target_document_id": target_id,
        "strategies": {
            "BM25": len(bm25_result["Results"]),
            "Dense": len(dense_result["Results"]),
            "Hybrid": len(baseline_hybrid["Results"]),
            "DependencyAwareBM25": len(dependency_bm25_result["Results"]),
            "DependencyAwareDense": len(dependency_dense_result["Results"]),
            "DependencyAwareHybrid": len(dependency_hybrid["Results"]),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_manifest(args.dataset)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    checkpoint = _load_json(checkpoint_path) if checkpoint_path.is_file() else {
        "SchemaVersion": "offline-orchestration-v1",
        "DatasetPath": str(Path(args.dataset).resolve()),
        "Cases": {},
    }
    roots = _parse_mapping(args.repository_root, "--repository-root")
    archives = _parse_mapping(args.repository_archive, "--repository-archive")
    preparation_records: dict[str, Any] = {}
    acquisition_root = output / "acquired-repositories"
    for repository, archive in archives.items():
        materialized_root = acquisition_root / re.sub(r"[^A-Za-z0-9_.-]", "_", repository)
        preparation_records[repository] = materialize_archive(archive, materialized_root)
        if preparation_records[repository]["status"] == "READY":
            roots.setdefault(repository, materialized_root)
    write_preparation_record(output / "repository-preparation.json", preparation_records)
    metadata = _parse_mapping(args.revision_metadata, "--revision-metadata")
    corpus_by_repo = _parse_mapping(args.corpus, "--corpus")
    sidecar_by_repo = _parse_mapping(args.semantic_sidecar, "--semantic-sidecar")
    indexed_corpora, indexed_sidecars = _resolve_index_inputs(
        args.repository_index + args.repository_index_root
    )
    corpus_by_repo.update(indexed_corpora)
    sidecar_by_repo.update(indexed_sidecars)
    case_records = manifest["Cases"]
    for case in case_records:
        case_id = case["CaseId"]
        previous = checkpoint["Cases"].get(case_id)
        if previous and previous.get("status") == "PASS":
            continue
        repo = case["Repository"]
        case_dir = output / "cases" / re.sub(r"[^A-Za-z0-9_.-]", "_", case_id[:100])
        record: dict[str, Any] = {
            "CaseId": case_id,
            "Repository": repo,
            "Commit": case["Commit"],
            "FeatureEnvyLabel": case["FeatureEnvyLabel"],
        }
        root = roots.get(repo)
        revision = _verify_revision(repo, case["Commit"], root, metadata.get(repo))
        record["revision"] = revision
        if revision["status"] != "READY":
            record.update({"status": "BLOCKED", "failure": revision})
            checkpoint["Cases"][case_id] = record
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
            continue
        try:
            source = _find_local_source(root, case["File"])
            project = _find_project(
                root,
                source,
                repo,
                output / "analysis-projects",
            )
            analysis_path = case_dir / "analysis.json"
            previous_analysis = (
                previous.get("analysis", {}).get("path") if previous else None
            )
            if (
                previous
                and previous.get("analysis", {}).get("status") == "PASS"
                and previous_analysis
                and Path(previous_analysis).is_file()
            ):
                analysis_path = Path(previous_analysis)
                analysis = _load_json(analysis_path)
            else:
                analysis = invoke_roslyn(
                    analyzer_path=args.analyzer,
                    project_path=project,
                    source_file_path=source,
                    msbuild_path=args.msbuild,
                    method_name=_parse_target_identifier(case_id)[1],
                    containing_type=_parse_target_identifier(case_id)[0],
                    signature=case_id,
                    source_line=case.get("StartLine"),
                    repository=repo,
                    commit=case["Commit"],
                    dataset_annotation=str(case["Annotation"]),
                    output_path=analysis_path,
                )
            metrics = evaluate_file(analysis_path)
            record["analysis"] = {"path": str(analysis_path.resolve()), "status": "PASS"}
            record["metrics"] = metrics
            if case["FeatureEnvyLabel"]:
                corpus = corpus_by_repo.get(repo)
                sidecar = sidecar_by_repo.get(repo)
                if corpus is None or sidecar is None:
                    record["retrieval"] = {
                        "status": "BLOCKED",
                        "category": "RETRIEVAL_INPUT_MISSING",
                        "message": "Repository corpus or semantic sidecar is unavailable.",
                    }
                else:
                    record["retrieval"] = _run_retrieval(
                        case=case,
                        analysis_path=analysis_path,
                        root=root,
                        corpus=corpus,
                        model=args.model,
                        semantic_sidecar=sidecar,
                        output=case_dir / "retrieval",
                        device=args.device,
                    )
                if record["retrieval"]["status"] != "PASS":
                    record.update(
                        {
                            "status": "BLOCKED",
                            "failure": record["retrieval"],
                        }
                    )
                else:
                    record["status"] = "PASS"
            else:
                record["retrieval"] = {"status": "NOT_APPLICABLE"}
                record["status"] = "PASS"
            record["detection_status"] = "PASS"
        except Exception as error:
            record.update(
                {
                    "status": "BLOCKED",
                    "failure": {
                        "category": "ANALYSIS_OR_RETRIEVAL_ERROR",
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                }
            )
        checkpoint["Cases"][case_id] = record
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    counts = {
        "total": len(case_records),
        "pass": sum(value.get("status") == "PASS" for value in checkpoint["Cases"].values()),
        "blocked": sum(value.get("status") == "BLOCKED" for value in checkpoint["Cases"].values()),
    }
    summary = {
        "SchemaVersion": "offline-orchestration-summary-v1",
        "Counts": counts,
        "BlockedByCategory": {},
        "Checkpoint": str(checkpoint_path.resolve()),
    }
    for value in checkpoint["Cases"].values():
        if value.get("status") == "BLOCKED":
            category = value.get("failure", {}).get("category", "UNKNOWN")
            summary["BlockedByCategory"][category] = summary["BlockedByCategory"].get(category, 0) + 1
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--msbuild", type=Path, required=True)
    parser.add_argument("--repository-root", action="append", default=[])
    parser.add_argument("--repository-archive", action="append", default=[])
    parser.add_argument("--revision-metadata", action="append", default=[])
    parser.add_argument("--corpus", action="append", default=[])
    parser.add_argument("--semantic-sidecar", action="append", default=[])
    parser.add_argument("--repository-index", action="append", default=[])
    parser.add_argument("--repository-index-root", action="append", default=[])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cpu")
    summary = run(parser.parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
