"""Prepare gated DeepSeek jobs from completed offline retrieval evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from dataset_manifest import build_manifest
from prompt_builder import build_refactoring_prompt, prompt_sha256


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


_SIGNATURE_TYPE_ALIASES = (
    ("System.Collections.Generic.Dictionary<", "Dictionary<"),
    ("System.Collections.Generic.IList<", "IList<"),
    ("System.Collections.Generic.IReadOnlyList<", "IReadOnlyList<"),
    ("System.Globalization.CultureInfo", "CultureInfo"),
    ("System.Reflection.ParameterInfo", "ParameterInfo"),
    ("System.Threading.CancellationToken", "CancellationToken"),
    ("System.DateTime", "DateTime"),
    ("System.Guid", "Guid"),
    ("System.Type", "Type"),
    ("System.Boolean", "bool"),
    ("System.Int32", "int"),
    ("System.Object", "object"),
    ("System.String", "string"),
)


def _canonical_signature(signature: str) -> str:
    canonical = signature
    for qualified, alias in _SIGNATURE_TYPE_ALIASES:
        canonical = re.sub(
            rf"(?<![\w.]){re.escape(qualified)}",
            alias,
            canonical,
        )
    return re.sub(r"\s+", " ", canonical).strip()


def _same_signature(left: str, right: str) -> bool:
    return _canonical_signature(left) == _canonical_signature(right)


def _find_case_directory(root: Path, case_id: str) -> Path:
    directory_name = re.sub(r"[^A-Za-z0-9_.-]", "_", case_id[:100])
    direct = root / "cases" / directory_name
    if (direct / "analysis.json").is_file():
        return direct
    for analysis_path in root.glob("cases/*/analysis.json"):
        analysis = _load_json(analysis_path)
        if _same_signature(
            str(analysis.get("Target", {}).get("Signature", "")),
            case_id,
        ):
            return analysis_path.parent
    raise FileNotFoundError(f"No completed analysis found for case: {case_id}")


def _load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            document = json.loads(line)
            documents[document["DocumentId"]] = document
    return documents


def _dependency_context(analysis: dict[str, Any]) -> dict[str, Any]:
    collections = (
        "Callees",
        "Fields",
        "Properties",
        "ReferencedTypes",
        "Interfaces",
        "LikelyEnviedTypeCandidates",
        "UnresolvedDependencies",
    )
    repository = analysis.get("Repository", {})
    target = analysis.get("Target", {})
    context: dict[str, Any] = {}
    if isinstance(repository, dict):
        context["Repository"] = {
            key: repository[key]
            for key in ("RepositoryIdentifier", "Commit")
            if key in repository
        }
    if isinstance(target, dict):
        context["Target"] = {
            key: target[key]
            for key in (
                "ContainingType",
                "MethodName",
                "Signature",
                "DatasetAnnotation",
            )
            if key in target
        }
    for collection in collections:
        values = analysis.get(collection)
        if isinstance(values, list):
            context[collection] = [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"Location", "ProjectPath", "SourceFile"}
                }
                for item in values
                if isinstance(item, dict)
            ]
    return context


def _target_document(
    corpus: dict[str, dict[str, Any]],
    case: dict[str, Any],
) -> dict[str, Any]:
    source_path = case["File"].replace("\\", "/").casefold()
    matches = [
        document
        for document in corpus.values()
        if _same_signature(
            str(document.get("Signature", "")),
            case["CaseId"],
        )
        and str(document.get("File", "")).replace("\\", "/").casefold()
        == source_path
        and document.get("Repository") == case["Repository"]
        and document.get("Commit") == case["Commit"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one pinned-corpus target for {case['CaseId']}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _retrieval_identity_error(
    result: dict[str, Any],
    case: dict[str, Any],
    target_document: dict[str, Any],
) -> str | None:
    target_id = target_document.get("DocumentId")
    if (
        not _same_signature(
            str(target_document.get("Signature", "")),
            case["CaseId"],
        )
        or target_document.get("Repository") != case["Repository"]
        or target_document.get("Commit") != case["Commit"]
        or not isinstance(target_id, str)
        or not target_id
    ):
        return "Corpus target does not match the exact case, repository, and commit."
    result_signature = result.get("QueryTargetSignature")
    if result_signature is not None and not _same_signature(
        str(result_signature),
        case["CaseId"],
    ):
        return "Retrieval result target signature does not match the exact CaseId."
    if result.get("QueryTargetDocumentId") != target_id:
        return "Retrieval result target document ID does not match the pinned corpus target."
    if (
        result.get("Repository", case["Repository"]) != case["Repository"]
        or result.get("Commit", case["Commit"]) != case["Commit"]
    ):
        return "Retrieval result repository or commit does not match the dataset case."
    return None


def _write_prepared_prompt(path: Path, prompt: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != prompt:
            raise FileExistsError(f"Existing prepared prompt differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(prompt, encoding="utf-8")
    temporary.replace(path)


def prepare_paired(
    *,
    dataset: Path,
    orchestration_roots: list[Path],
    repository_roots: dict[str, Path],
    corpus_by_repository: dict[str, Path],
    output: Path,
    retrieval_methods: tuple[str, ...] = ("bm25", "dense", "hybrid"),
    run_missing_retrieval: bool = False,
    model_path: Path | None = None,
    device: str = "cpu",
    dense_cache_by_repository: dict[str, Path] | None = None,
    case_ids: set[str] | None = None,
    retrieval_input_blockers: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Prepare baseline/augmented jobs with identical retrieved documents per pair."""
    manifest_path = output / "jobs.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite paired job directory: {output}")
    if not retrieval_methods or len(set(retrieval_methods)) != len(retrieval_methods):
        raise ValueError("Retrieval methods must be a non-empty unique sequence.")
    retrieval_files = {
        "bm25": "bm25.json",
        "dense": "dense.json",
        "hybrid": "hybrid.json",
    }
    unsupported = set(retrieval_methods) - retrieval_files.keys()
    if unsupported:
        raise ValueError(f"Unsupported retrieval methods: {sorted(unsupported)}")

    manifest = build_manifest(dataset)
    positive_cases = [
        case for case in manifest["Cases"] if case["FeatureEnvyLabel"]
    ]
    if case_ids is not None:
        known_case_ids = {case["CaseId"] for case in positive_cases}
        unknown_case_ids = case_ids - known_case_ids
        if unknown_case_ids:
            raise ValueError(
                f"Requested case IDs are not positive dataset cases: "
                f"{sorted(unknown_case_ids)}"
            )
    output.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []

    def add_blocked_jobs(case: dict[str, Any], reason: str, methods: tuple[str, ...]) -> None:
        blocked.append({"CaseId": case["CaseId"], "Reason": reason})
        for method in methods:
            pair_id = hashlib.sha256(
                f"{case['CaseId']}\0{method}".encode("utf-8")
            ).hexdigest()[:16]
            for condition in ("baseline", "dependency_augmented"):
                blocked_jobs.append(
                    {
                        "JobId": f"{method}/{condition}/{pair_id}",
                        "PairId": f"{method}/{pair_id}",
                        "CaseId": case["CaseId"],
                        "Repository": case["Repository"],
                        "Commit": case["Commit"],
                        "SourcePath": case["File"],
                        "StartLine": case["StartLine"],
                        "EndLine": case["EndLine"],
                        "Annotation": case["Annotation"],
                        "RetrievalMethod": method,
                        "Condition": condition,
                        "Mode": "BLOCKED",
                        "Blocker": reason,
                    }
                )

    blocked_jobs: list[dict[str, Any]] = []
    for case in positive_cases:
        if case_ids is not None and case["CaseId"] not in case_ids:
            continue
        case_directory = None
        for root in orchestration_roots:
            try:
                candidate = _find_case_directory(root, case["CaseId"])
            except FileNotFoundError:
                continue
            analysis_path = candidate / "analysis.json"
            analysis = _load_json(analysis_path)
            repository = analysis.get("Repository", {})
            target = analysis.get("Target", {})
            if (
                repository.get("RepositoryIdentifier") != case["Repository"]
                or repository.get("Commit") != case["Commit"]
                or not _same_signature(
                    str(target.get("Signature", "")),
                    case["CaseId"],
                )
            ):
                continue
            case_directory = candidate
            break
        if case_directory is None:
            add_blocked_jobs(
                case,
                "No matching pinned-revision analysis evidence.",
                retrieval_methods,
            )
            continue

        analysis = _load_json(case_directory / "analysis.json")
        corpus_path = corpus_by_repository.get(case["Repository"])
        repository_root = repository_roots.get(case["Repository"])
        if corpus_path is None or not corpus_path.is_file():
            add_blocked_jobs(case, "Repository corpus unavailable.", retrieval_methods)
            continue
        if repository_root is None:
            add_blocked_jobs(
                case,
                "Pinned repository source root unavailable.",
                retrieval_methods,
            )
            continue

        corpus = _load_corpus(corpus_path)
        try:
            target_document = _target_document(corpus, case)
            target_id = target_document["DocumentId"]
            source_file = repository_root.joinpath(*Path(case["File"]).parts)
            source_lines = source_file.read_text(encoding="utf-8").splitlines(keepends=True)
            target_source = "".join(
                source_lines[case["StartLine"] - 1 : case["EndLine"]]
            )
        except (OSError, ValueError, KeyError) as error:
            add_blocked_jobs(case, str(error), retrieval_methods)
            continue

        dependency_context = _dependency_context(analysis)
        dependency_evidence_fields = (
            "Callees",
            "Fields",
            "Properties",
            "ReferencedTypes",
            "Interfaces",
            "LikelyEnviedTypeCandidates",
            "UnresolvedDependencies",
        )
        if not any(dependency_context.get(field) for field in dependency_evidence_fields):
            add_blocked_jobs(
                case,
                "Matching Roslyn analysis contains no dependency evidence.",
                retrieval_methods,
            )
            continue
        target_hash = hashlib.sha256(case["CaseId"].encode("utf-8")).hexdigest()[:16]
        generated_retrieval_directory = (
            output
            / "retrieval-generated"
            / re.sub(r"[^A-Za-z0-9_.-]", "_", case["Repository"])
            / target_hash
        )
        result_paths = {
            method: case_directory / "retrieval" / filename
            for method, filename in retrieval_files.items()
        }

        def result_path_for(method: str) -> Path:
            path = result_paths[method]
            if path.is_file():
                return path
            generated = generated_retrieval_directory / retrieval_files[method]
            if generated.is_file():
                return generated
            if not run_missing_retrieval:
                return path
            generated_retrieval_directory.mkdir(parents=True, exist_ok=True)
            if method == "bm25":
                from bm25_retriever import retrieve as retrieve_bm25

                retrieve_bm25(
                    corpus_path=corpus_path,
                    target_document_id=target_id,
                    output_path=generated,
                )
            elif method == "dense":
                if model_path is None:
                    raise ValueError("A local model path is required for missing Dense results.")
                from dense_retriever import retrieve as retrieve_dense

                retrieve_dense(
                    corpus_path=corpus_path,
                    target_document_id=target_id,
                    model_path=model_path,
                    cache_path=(
                        (dense_cache_by_repository or {}).get(case["Repository"])
                        or (
                            output
                            / "retrieval-cache"
                            / re.sub(r"[^A-Za-z0-9_.-]", "_", case["Repository"])
                            / "dense-embeddings.npy"
                        )
                    ),
                    output_path=generated,
                    device=device,
                )
            elif method == "hybrid":
                from hybrid_retriever import retrieve as retrieve_hybrid

                bm25_path = result_path_for("bm25")
                dense_path = result_path_for("dense")
                if not bm25_path.is_file() or not dense_path.is_file():
                    raise FileNotFoundError(
                        "Hybrid retrieval requires saved Standard BM25 and Dense results."
                    )
                bm25_result = _load_json(bm25_path)
                dense_result = _load_json(dense_path)
                retrieve_hybrid(
                    bm25=lambda: bm25_result,
                    dense=lambda: dense_result,
                    output_path=generated,
                    target_document_id=target_id,
                )
            return generated

        for method in retrieval_methods:
            try:
                result_path = result_path_for(method)
            except (OSError, RuntimeError, ValueError) as error:
                add_blocked_jobs(case, str(error), (method,))
                continue
            if not result_path.is_file():
                blocker_reason = (
                    retrieval_input_blockers or {}
                ).get(case["CaseId"], {}).get(method)
                add_blocked_jobs(
                    case,
                    blocker_reason
                    or f"Missing {method} retrieval result: {result_path}",
                    (method,),
                )
                continue
            result = _load_json(result_path)
            hits = result.get("Results")
            if (
                (identity_error := _retrieval_identity_error(
                    result,
                    case,
                    target_document,
                )) is not None
                or not isinstance(hits, list)
                or len(hits) != 10
            ):
                add_blocked_jobs(
                    case,
                    (
                        f"{method} result identity mismatch: {identity_error}"
                        if identity_error
                        else f"{method} result violates the top-10 contract."
                    ),
                    (method,),
                )
                continue
            if method == "hybrid" and (
                result.get("Fusion", {}).get("Method") != "RRF"
                or result.get("Fusion", {}).get("K") != 60
            ):
                add_blocked_jobs(
                    case,
                    "Hybrid result does not declare RRF with k=60.",
                    (method,),
                )
                continue
            hit_ids = [hit.get("DocumentId") for hit in hits if isinstance(hit, dict)]
            if (
                len(hit_ids) != 10
                or len(set(hit_ids)) != 10
                or target_id in hit_ids
                or any(document_id not in corpus for document_id in hit_ids)
                or any(
                    corpus[document_id].get("Repository") != case["Repository"]
                    or corpus[document_id].get("Commit") != case["Commit"]
                    for document_id in hit_ids
                    if document_id in corpus
                )
            ):
                add_blocked_jobs(
                    case,
                    (
                        f"{method} result has invalid, duplicate, target, or "
                        "cross-revision hits."
                    ),
                    (method,),
                )
                continue
            retrieved = [
                {
                    **corpus[hit["DocumentId"]],
                    "Rank": hit["Rank"],
                    "Score": hit.get("RRFScore", hit.get("Score", 0.0)),
                }
                for hit in hits
            ]
            pair_id = hashlib.sha256(
                f"{case['CaseId']}\0{method}".encode("utf-8")
            ).hexdigest()[:16]
            for condition in ("baseline", "dependency_augmented"):
                prompt = build_refactoring_prompt(
                    target={
                        "Repository": case["Repository"],
                        "Commit": case["Commit"],
                        "MethodIdentifier": case["CaseId"],
                    },
                    target_source=target_source,
                    target_location=(
                        f"{case['File']}#L{case['StartLine']}-L{case['EndLine']}"
                    ),
                    host_class_info=(
                        f"Containing type: {target_document['ContainingType']}"
                    ),
                    retrieved_documents=retrieved,
                    dependency_context=(
                        dependency_context
                        if condition == "dependency_augmented"
                        else None
                    ),
                )
                job_id = f"{method}/{condition}/{pair_id}"
                job_directory = output / method / condition / pair_id
                prompt_path = job_directory / "prompt.txt"
                _write_prepared_prompt(prompt_path, prompt)
                jobs.append(
                    {
                        "JobId": job_id,
                        "PairId": f"{method}/{pair_id}",
                        "CaseId": case["CaseId"],
                        "Repository": case["Repository"],
                        "Commit": case["Commit"],
                        "SourcePath": case["File"],
                        "StartLine": case["StartLine"],
                        "EndLine": case["EndLine"],
                        "Annotation": case["Annotation"],
                        "RetrievalMethod": method,
                        "Condition": condition,
                        "TargetDocumentId": target_id,
                        "RetrievedDocumentIds": hit_ids,
                        "RetrievalResultPath": str(result_path.resolve()),
                        "AnalysisPath": str((case_directory / "analysis.json").resolve()),
                        "BaselineAnalysisPath": str(
                            (case_directory / "analysis.json").resolve()
                        ),
                        "PromptPath": str(prompt_path.resolve()),
                        "Mode": "PREPARED_ONLY",
                        "LiveRequest": False,
                        "PromptSHA256": prompt_sha256(prompt),
                    }
                )

    manifest_output = {
        "SchemaVersion": "paired-deepseek-job-manifest-v1",
        "DatasetPath": str(Path(dataset).resolve()),
        "Model": "deepseek-flash",
        "Temperature": 0.0,
        "MaxTokens": 4096,
        "Thinking": "disabled",
        "RetrievalMethods": list(retrieval_methods),
        "Conditions": ["baseline", "dependency_augmented"],
        "Jobs": jobs,
        "BlockedCases": blocked,
        "BlockedJobs": blocked_jobs,
        "EstimatedApiCalls": len(jobs),
        "LiveRequestsMade": 0,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest_output, indent=2),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest_output


def prepare(
    *,
    dataset: Path,
    orchestration_roots: list[Path],
    repository_roots: dict[str, Path],
    corpus_by_repository: dict[str, Path],
    output: Path,
    strategies: tuple[str, ...],
) -> dict[str, Any]:
    manifest = build_manifest(dataset)
    output.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []
    for case in manifest["Cases"]:
        if not case["FeatureEnvyLabel"]:
            continue
        case_directory = None
        for root in orchestration_roots:
            try:
                case_directory = _find_case_directory(root, case["CaseId"])
                break
            except FileNotFoundError:
                continue
        if case_directory is None:
            blocked.append({"CaseId": case["CaseId"], "Reason": "No completed analysis evidence."})
            continue
        retrieval_directory = case_directory / "retrieval"
        corpus_path = corpus_by_repository.get(case["Repository"])
        if corpus_path is None:
            blocked.append({"CaseId": case["CaseId"], "Reason": "Repository corpus unavailable."})
            continue
        corpus = _load_corpus(corpus_path)
        repository_root = repository_roots.get(case["Repository"])
        if repository_root is None:
            blocked.append({"CaseId": case["CaseId"], "Reason": "Repository root unavailable."})
            continue
        target_source_path = Path(case["File"])
        source_file = repository_root.joinpath(*target_source_path.parts)
        if not source_file.is_file():
            blocked.append(
                {"CaseId": case["CaseId"], "Reason": "Source workspace is not persisted in orchestration output."}
            )
            continue
        source_lines = source_file.read_text(encoding="utf-8").splitlines(keepends=True)
        target_source = "".join(source_lines[case["StartLine"] - 1 : case["EndLine"]])
        for strategy in strategies:
            result_path = retrieval_directory / f"{strategy}.json"
            if not result_path.is_file():
                blocked.append(
                    {"CaseId": case["CaseId"], "Reason": f"Missing retrieval result: {result_path.name}"}
                )
                continue
            result = _load_json(result_path)
            retrieved = [
                {
                    **corpus[hit["DocumentId"]],
                    "Rank": hit["Rank"],
                    "Score": hit.get("RRFScore", hit.get("Score", 0.0)),
                }
                for hit in result.get("Results", [])
                if hit["DocumentId"] in corpus
            ]
            prompt = build_refactoring_prompt(
                target={
                    "Repository": case["Repository"],
                    "Commit": case["Commit"],
                    "MethodIdentifier": case["CaseId"],
                },
                target_source=target_source,
                target_location=f"{case['File']}#L{case['StartLine']}-L{case['EndLine']}",
                host_class_info=f"Containing type: {case['CaseId'].rsplit('.', 1)[0]}",
                retrieved_documents=retrieved,
            )
            job_directory = output / strategy / str(len(jobs) + 1).zfill(3)
            job_directory.mkdir(parents=True, exist_ok=False)
            prompt_path = job_directory / "prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            jobs.append(
                {
                    "JobId": job_directory.relative_to(output).as_posix(),
                    "CaseId": case["CaseId"],
                    "Repository": case["Repository"],
                    "Commit": case["Commit"],
                    "Strategy": strategy,
                    "PromptPath": str(prompt_path.resolve()),
                    "Mode": "PREPARED_ONLY",
                    "LiveRequest": False,
                }
            )
    manifest_output = {
        "SchemaVersion": "deepseek-generation-job-manifest-v1",
        "Model": "deepseek-flash",
        "Temperature": 0.0,
        "MaxTokens": 4096,
        "Thinking": "disabled",
        "Jobs": jobs,
        "BlockedPositiveCases": blocked,
        "EstimatedApiCalls": len(jobs),
        "LiveRequestsMade": 0,
    }
    (output / "jobs.json").write_text(json.dumps(manifest_output, indent=2), encoding="utf-8")
    return manifest_output


def _mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        repository, path = value.split("=", 1)
        result[repository] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--orchestration-root", type=Path, action="append", required=True)
    parser.add_argument("--repository-root", action="append", default=[])
    parser.add_argument("--corpus", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Prepare BM25, Dense, and Hybrid baseline/augmentation prompt pairs.",
    )
    parser.add_argument(
        "--retrieval-methods",
        nargs="+",
        default=["bm25", "dense", "hybrid"],
    )
    parser.add_argument(
        "--run-missing-retrieval",
        action="store_true",
        help="Run only absent Standard BM25/Dense/Hybrid results from saved inputs.",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dense-cache", action="append", default=[])
    parser.add_argument("--case-id", action="append")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["hybrid", "dependency-aware-hybrid"],
    )
    args = parser.parse_args()
    if args.paired:
        result = prepare_paired(
            dataset=args.dataset,
            orchestration_roots=args.orchestration_root,
            repository_roots=_mapping(args.repository_root),
            corpus_by_repository=_mapping(args.corpus),
            output=args.output,
            retrieval_methods=tuple(args.retrieval_methods),
            run_missing_retrieval=args.run_missing_retrieval,
            model_path=args.model,
            device=args.device,
            dense_cache_by_repository=_mapping(args.dense_cache),
            case_ids=set(args.case_id) if args.case_id is not None else None,
        )
    else:
        result = prepare(
            dataset=args.dataset,
            orchestration_roots=args.orchestration_root,
            repository_roots=_mapping(args.repository_root),
            corpus_by_repository=_mapping(args.corpus),
            output=args.output,
            strategies=tuple(args.strategies),
        )
    print(f"PREPARED JOBS: {len(result['Jobs'])}")
    print(
        f"BLOCKED POSITIVE CASES: "
        f"{len(result.get('BlockedCases', result.get('BlockedPositiveCases', [])))}"
    )
    print(f"ESTIMATED API CALLS: {result['EstimatedApiCalls']}")
    print("LIVE REQUESTS MADE: 0")


if __name__ == "__main__":
    main()
