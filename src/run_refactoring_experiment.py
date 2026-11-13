"""Run resumable paired retrieval-context refactoring jobs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

from analyze_dataset_case import _parse_target_identifier
from deepseek_client import DeepSeekConfig
from deepseek_experiment import _extract_text, run_experiment
from feature_envy_metrics import evaluate_analysis
from patch_normalizer import apply_changed_files, parse_changed_files
from prepare_generation_jobs import prepare_paired
from prompt_builder import prompt_sha256
from roslyn_bridge import invoke_roslyn


_IGNORED_COPY_NAMES = {".vs", "bin", "obj"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(json.dumps(value, indent=2))
            temporary.flush()
            os.fsync(temporary.fileno())

        for attempt in range(5):
            try:
                os.replace(temporary_path, path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _sha256(path: Path) -> str:
    return prompt_sha256(path.read_bytes())


def _config_fingerprint(config: DeepSeekConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "endpoint": config.endpoint,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }


def _response_text(response_path: Path) -> str:
    response = _read_json(response_path)
    return _extract_text(response)


def _proposal_payload(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("DeepSeek content is not a JSON proposal.") from error
    if not isinstance(value, dict):
        raise ValueError("DeepSeek proposal must be a JSON object.")
    if value.get("status") not in {"READY", "INSUFFICIENT_CONTEXT"}:
        raise ValueError("DeepSeek proposal has an unsupported status.")
    return value


def _compatible_saved_output(
    prompt_hash: str,
    roots: list[Path],
    expected_config: dict[str, Any],
) -> Path | None:
    for root in roots:
        if not root.is_dir():
            continue
        for prompt_path in root.rglob("prompt.txt"):
            try:
                if _sha256(prompt_path) != prompt_hash:
                    continue
                directory = prompt_path.parent
                error_path = directory / "error-status.json"
                config_path = directory / "model-config.json"
                response_path = directory / "response.json"
                if not all(path.is_file() for path in (error_path, config_path, response_path)):
                    continue
                error_status = _read_json(error_path)
                saved_config = _read_json(config_path)
                if error_status.get("status") != "NONE":
                    continue
                if any(saved_config.get(key) != value for key, value in expected_config.items()):
                    continue
                response = _read_json(response_path)
                choices = response.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                first = choices[0]
                if not isinstance(first, dict) or first.get("finish_reason") not in (None, "stop"):
                    continue
                text = _extract_text(response)
                proposal = _proposal_payload(text)
                if proposal.get("status") == "READY":
                    parse_changed_files(proposal)
                return directory
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    return None


def _saved_run_for_prompt(
    prompt_hash: str,
    root: Path,
    expected_config: dict[str, Any],
) -> Path | None:
    if not root.is_dir():
        return None
    matches: list[Path] = []
    for response_path in root.glob("bm25-deepseek-*/response.json"):
        directory = response_path.parent
        prompt_path = directory / "prompt.txt"
        config_path = directory / "model-config.json"
        if not prompt_path.is_file() or not config_path.is_file():
            continue
        try:
            if _sha256(prompt_path) != prompt_hash:
                continue
            saved_config = _read_json(config_path)
            if any(saved_config.get(key) != value for key, value in expected_config.items()):
                continue
            _read_json(response_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        matches.append(directory)
    if len(matches) > 1:
        raise ValueError(
            f"Multiple saved DeepSeek responses match prompt hash {prompt_hash}."
        )
    return matches[0] if matches else None


def _load_revision_sha(metadata_path: Path, repository: str) -> str | None:
    metadata = _read_json(metadata_path)
    if isinstance(metadata.get("sha"), str):
        return metadata["sha"]
    records = metadata.get("Records")
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict) and record.get("Repository") == repository:
                value = record.get("VerifiedCommit")
                return value if isinstance(value, str) else None
    return None


def _safe_job_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _new_validation_attempt(output: Path, job_id: str) -> Path:
    attempts = output / "validation" / _safe_job_component(job_id)
    attempt_number = 1
    while True:
        candidate = attempts / f"attempt-{attempt_number:03d}"
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            attempt_number += 1


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _IGNORED_COPY_NAMES}


def _new_validation_workspace(source_root: Path, configured_root: str | None) -> Path:
    if configured_root:
        root = Path(configured_root).resolve()
    elif os.name == "nt":
        root = Path(os.environ.get("SystemDrive", "C:") + "\\") / "fe-work"
    else:
        root = Path(tempfile.gettempdir()) / "fe-work"
    root = root.resolve()
    if root.is_relative_to(source_root.resolve()):
        raise ValueError("Validation workspace root must not be inside the source root.")
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="v-", dir=root)) / "w"


def _workspace_path(workspace: Path, relative_path: str, label: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a safe workspace-relative path.")
    resolved = (workspace / relative).resolve()
    if not resolved.is_relative_to(workspace.resolve()):
        raise ValueError(f"{label} escapes the isolated workspace.")
    return resolved


def _verify_source_revision(
    source_root: Path,
    expected_commit: str,
    configured_commit: str | None,
) -> None:
    git_directory = source_root / ".git"
    if git_directory.exists():
        head = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if head.returncode != 0 or head.stdout.strip() != expected_commit:
            raise ValueError("Source checkout HEAD does not match the pinned commit.")
        status = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode != 0 or status.stdout.strip():
            raise ValueError("Source checkout is not clean; refusing to validate a mixed revision.")
    elif configured_commit != expected_commit:
        raise ValueError(
            "Non-Git source roots require an explicit matching source_commit."
        )


def _run_commands(
    *,
    commands: list[list[str]],
    workspace: Path,
    project: Path,
    test_project: Path | None,
    output_directory: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    if not commands:
        return {"status": "BLOCKED", "reason": "No commands configured."}
    results: list[dict[str, Any]] = []
    output_directory.mkdir(parents=True, exist_ok=True)
    for index, command in enumerate(commands, start=1):
        if not command or any(not isinstance(token, str) for token in command):
            return {"status": "BLOCKED", "reason": f"Invalid command at index {index}."}
        substitutions = {
            "{workspace}": str(workspace),
            "{project}": str(project),
            "{test_project}": str(test_project) if test_project else "",
        }
        expanded = []
        for token in command:
            for key, value in substitutions.items():
                token = token.replace(key, value)
            expanded.append(token)
        log_base = output_directory / f"command-{index:02d}"
        try:
            completed = subprocess.run(
                expanded,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
            (log_base.with_suffix(".stdout.log")).write_text(
                completed.stdout, encoding="utf-8"
            )
            (log_base.with_suffix(".stderr.log")).write_text(
                completed.stderr, encoding="utf-8"
            )
            results.append(
                {
                    "command": expanded,
                    "exit_code": completed.returncode,
                    "status": "PASS" if completed.returncode == 0 else "FAIL",
                    "stdout_log": str(log_base.with_suffix(".stdout.log")),
                    "stderr_log": str(log_base.with_suffix(".stderr.log")),
                }
            )
            if completed.returncode != 0:
                return {"status": "FAIL", "commands": results}
        except subprocess.TimeoutExpired as error:
            results.append(
                {
                    "command": expanded,
                    "status": "TIMEOUT",
                    "timeout_seconds": timeout_seconds,
                    "stdout": error.stdout.decode(errors="replace")
                    if isinstance(error.stdout, bytes)
                    else error.stdout,
                    "stderr": error.stderr.decode(errors="replace")
                    if isinstance(error.stderr, bytes)
                    else error.stderr,
                }
            )
            return {"status": "FAIL", "commands": results}
        except OSError as error:
            results.append(
                {"command": expanded, "status": "ERROR", "error": str(error)}
            )
            return {"status": "FAIL", "commands": results}
    return {"status": "PASS", "commands": results}


def _validate_job(
    *,
    job: dict[str, Any],
    proposal: dict[str, Any],
    config: dict[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    repository = job["Repository"]
    expected_commit = job["Commit"]
    revision_metadata = Path(config["revision_metadata"])
    if _load_revision_sha(revision_metadata, repository) != expected_commit:
        return {"status": "BLOCKED", "reason": "Pinned revision metadata mismatch."}

    source_root = Path(config["source_root"]).resolve()
    if not source_root.is_dir():
        return {"status": "BLOCKED", "reason": f"Source root is unavailable: {source_root}"}
    try:
        _verify_source_revision(
            source_root,
            expected_commit,
            config.get("source_commit"),
        )
    except (OSError, ValueError) as error:
        return {"status": "BLOCKED", "reason": str(error)}
    try:
        workspace = _new_validation_workspace(source_root, config.get("workspace_root"))
    except (OSError, ValueError) as error:
        return {
            "status": "BLOCKED",
            "reason": f"Could not create an isolated validation workspace: {error}",
        }
    try:
        shutil.copytree(source_root, workspace, ignore=_copy_ignore)
        changed_files = parse_changed_files(proposal)
        patch_results = apply_changed_files(workspace, changed_files)
    except (OSError, ValueError) as error:
        return {
            "status": "BLOCKED",
            "reason": f"Workspace preparation or patch application failed: {error}",
            "workspace": str(workspace),
        }
    try:
        project = _workspace_path(workspace, config["project"], "project")
        source_file = _workspace_path(workspace, job["SourcePath"], "target source")
        test_project = (
            _workspace_path(workspace, config["test_project"], "test project")
            if config.get("test_project")
            else None
        )
    except ValueError as error:
        return {"status": "BLOCKED", "reason": str(error)}
    if not project.is_file():
        return {"status": "BLOCKED", "reason": f"Project file is missing: {project}"}
    if not source_file.is_file():
        return {"status": "BLOCKED", "reason": f"Target source is missing: {source_file}"}

    restore = {"status": "NOT_RUN", "reason": "No restore commands configured."}
    if "restore_commands" in config:
        restore = _run_commands(
            commands=config["restore_commands"],
            workspace=workspace,
            project=project,
            test_project=test_project,
            output_directory=output_directory / "restore",
            timeout_seconds=int(config.get("timeout_seconds", 1800)),
        )
        if restore.get("status") != "PASS":
            return {
                "status": "BLOCKED",
                "reason": "Configured dependency restore did not pass.",
                "restore": restore,
                "patch_application": patch_results,
                "workspace": str(workspace),
            }

    analyzer_path = Path(config["analyzer"])
    msbuild_path = Path(config["msbuild"])
    containing_type, method_name = _parse_target_identifier(job["CaseId"])
    analysis_path = output_directory / "post-refactoring-analysis.json"
    try:
        analysis = invoke_roslyn(
            analyzer_path=analyzer_path,
            project_path=project,
            source_file_path=source_file,
            msbuild_path=msbuild_path,
            method_name=method_name,
            containing_type=containing_type,
            signature=job["CaseId"],
            source_line=job.get("StartLine"),
            repository=repository,
            commit=expected_commit,
            dataset_annotation=str(job.get("Annotation", "")),
            output_path=analysis_path,
        )
    except Exception as error:
        return {
            "status": "BLOCKED",
            "reason": f"Post-refactoring Roslyn analysis failed: {error}",
            "patch_application": patch_results,
            "workspace": str(workspace),
        }
    if (
        analysis.get("Repository", {}).get("Commit") != expected_commit
        or analysis.get("Target", {}).get("Signature") != job["CaseId"]
    ):
        return {
            "status": "BLOCKED",
            "reason": "Post-refactoring analyzer returned a different target or revision.",
            "patch_application": patch_results,
            "workspace": str(workspace),
        }
    metrics = evaluate_analysis(analysis)
    baseline_analysis = _read_json(Path(job["BaselineAnalysisPath"]))
    baseline_metrics = evaluate_analysis(baseline_analysis)
    smell_eliminated = (
        baseline_metrics.get("Status") == "DEFINITE"
        and baseline_metrics.get("FeatureEnvy") is True
        and metrics.get("Status") == "DEFINITE"
        and metrics.get("FeatureEnvy") is False
    )
    build = _run_commands(
        commands=config.get("build_commands", []),
        workspace=workspace,
        project=project,
        test_project=test_project,
        output_directory=output_directory / "build",
        timeout_seconds=int(config.get("timeout_seconds", 1800)),
    )
    tests = (
        _run_commands(
            commands=config.get("test_commands", []),
            workspace=workspace,
            project=project,
            test_project=test_project,
            output_directory=output_directory / "tests",
            timeout_seconds=int(config.get("timeout_seconds", 1800)),
        )
        if build.get("status") == "PASS"
        else {"status": "NOT_RUN", "reason": "Build did not pass."}
    )
    return {
        "status": "COMPLETE",
        "workspace": str(workspace),
        "patch_application": patch_results,
        "restore": restore,
        "baseline_metrics": baseline_metrics,
        "post_refactoring_metrics": metrics,
        "baseline_detected_feature_envy": (
            baseline_metrics.get("Status") == "DEFINITE"
            and baseline_metrics.get("FeatureEnvy") is True
        ),
        "post_refactoring_smell_absent": (
            metrics.get("Status") == "DEFINITE"
            and metrics.get("FeatureEnvy") is False
        ),
        "smell_eliminated": smell_eliminated,
        "build": build,
        "tests": tests,
        "analysis_path": str(analysis_path),
    }


def _paired_summary(
    jobs: list[dict[str, Any]],
    blocked_jobs: list[dict[str, Any]],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    conditions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for job in [*jobs, *blocked_jobs]:
        state = checkpoint.get("Jobs", {}).get(job["JobId"], {})
        key = job["CaseId"]
        conditions.setdefault(key, {}).setdefault(job["RetrievalMethod"], []).append(
            {
                "condition": job["Condition"],
                "generation_status": state.get(
                    "generation_status",
                    "BLOCKED" if job.get("Mode") == "BLOCKED" else "PENDING",
                ),
                "validation_status": state.get("validation", {}).get("status", "NOT_RUN"),
                "smell_eliminated": state.get("validation", {}).get("smell_eliminated"),
                "build_status": state.get("validation", {}).get("build", {}).get("status"),
                "test_status": state.get("validation", {}).get("tests", {}).get("status"),
            }
        )
    pairs = []
    for case_id, methods in conditions.items():
        for method, values in methods.items():
            by_condition = {value["condition"]: value for value in values}
            pairs.append(
                {
                    "CaseId": case_id,
                    "RetrievalMethod": method,
                    "Baseline": by_condition.get("baseline"),
                    "DependencyAugmented": by_condition.get("dependency_augmented"),
                }
            )
    aggregate: dict[str, dict[str, dict[str, int]]] = {}
    for method in ("bm25", "dense", "hybrid"):
        aggregate[method] = {}
        for condition in ("baseline", "dependency_augmented"):
            cells = [
                checkpoint.get("Jobs", {}).get(job["JobId"], {})
                for job in [*jobs, *blocked_jobs]
                if job["RetrievalMethod"] == method
                and job["Condition"] == condition
            ]
            validations = [cell.get("validation", {}) for cell in cells]
            aggregate[method][condition] = {
                "jobs": len(cells),
                "generation_complete": sum(
                    cell.get("generation_status") in {"COMPLETE", "REUSED"}
                    for cell in cells
                ),
                "smell_eliminated": sum(
                    validation.get("smell_eliminated") is True
                    for validation in validations
                ),
                "build_pass": sum(
                    validation.get("build", {}).get("status") == "PASS"
                    for validation in validations
                ),
                "tests_pass": sum(
                    validation.get("tests", {}).get("status") == "PASS"
                    for validation in validations
                ),
            }
    return {
        "SchemaVersion": "paired-refactoring-results-v1",
        "Pairs": pairs,
        "AggregateByMethodAndCondition": aggregate,
    }


def run(
    *,
    jobs_manifest: Path,
    output: Path,
    config: DeepSeekConfig,
    live: bool,
    reuse_roots: list[Path],
    validation_config_path: Path | None,
) -> dict[str, Any]:
    manifest = _read_json(jobs_manifest)
    jobs = manifest.get("Jobs")
    if not isinstance(jobs, list):
        raise ValueError("Job manifest has no Jobs list.")
    blocked_jobs = manifest.get("BlockedJobs", [])
    if not isinstance(blocked_jobs, list):
        raise ValueError("BlockedJobs must be a list when present.")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    checkpoint = (
        _read_json(checkpoint_path)
        if checkpoint_path.is_file()
        else {"SchemaVersion": "paired-experiment-checkpoint-v1", "Jobs": {}}
    )
    validation_configs = (
        _read_json(validation_config_path)
        if validation_config_path is not None
        else {}
    )
    expected_model = _config_fingerprint(config)
    live_requests = 0

    for job in blocked_jobs:
        job_id = job["JobId"]
        if job_id not in checkpoint["Jobs"]:
            checkpoint["Jobs"][job_id] = {
                "generation_status": "BLOCKED",
                "failure": job.get("Blocker", "Inputs for this condition are unavailable."),
            }
    _write_json(checkpoint_path, checkpoint)

    for job in jobs:
        job_id = job["JobId"]
        prompt_path = Path(job["PromptPath"])
        if not prompt_path.is_file():
            checkpoint["Jobs"][job_id] = {
                "generation_status": "BLOCKED",
                "failure": f"Prompt is missing: {prompt_path}",
            }
            _write_json(checkpoint_path, checkpoint)
            continue
        prompt_hash = _sha256(prompt_path)
        if job.get("PromptSHA256") != prompt_hash:
            checkpoint["Jobs"][job_id] = {
                "generation_status": "BLOCKED",
                "failure": "Prepared prompt hash does not match the job manifest.",
            }
            _write_json(checkpoint_path, checkpoint)
            continue
        previous = checkpoint["Jobs"].get(job_id, {})
        recovered_run: Path | None = None
        if previous.get("generation_status") in {"RUNNING", "RESPONSE_SAVED"}:
            previous_run = previous.get("run_directory")
            if isinstance(previous_run, str):
                candidate = Path(previous_run)
                if (candidate / "response.json").is_file():
                    recovered_run = candidate
            if recovered_run is None:
                recovered_run = _saved_run_for_prompt(
                    prompt_hash,
                    output / "deepseek-runs",
                    expected_model,
                )
            if recovered_run is None:
                continue
            checkpoint["Jobs"][job_id] = {
                "generation_status": "RESPONSE_SAVED",
                "prompt_sha256": prompt_hash,
                "model": expected_model,
                "run_directory": str(recovered_run),
            }
            _write_json(checkpoint_path, checkpoint)

        previous_generation_complete = (
            previous.get("generation_status") in {"COMPLETE", "REUSED"}
            and previous.get("prompt_sha256") == prompt_hash
        )
        if previous_generation_complete and previous.get("validation", {}).get(
            "status"
        ) in {"COMPLETE", "BLOCKED"}:
            continue
        previous_run = (
            Path(previous["run_directory"])
            if previous_generation_complete and previous.get("run_directory")
            else None
        )
        if recovered_run is not None:
            run_directory = recovered_run
            generation_status = "COMPLETE"
        elif previous_run and (previous_run / "response.json").is_file():
            run_directory = previous_run
            generation_status = previous["generation_status"]
        else:
            reused = _compatible_saved_output(prompt_hash, reuse_roots, expected_model)
            if reused is not None:
                run_directory = reused
                generation_status = "REUSED"
            elif not live:
                checkpoint["Jobs"][job_id] = {
                    "generation_status": "AWAITING_APPROVAL",
                    "prompt_sha256": prompt_hash,
                    "model": expected_model,
                }
                _write_json(checkpoint_path, checkpoint)
                continue
            else:
                checkpoint["Jobs"][job_id] = {
                    "generation_status": "RUNNING",
                    "prompt_sha256": prompt_hash,
                    "model": expected_model,
                }
                _write_json(checkpoint_path, checkpoint)

                def persist_saved_response(run_directory: Path) -> None:
                    checkpoint["Jobs"][job_id] = {
                        "generation_status": "RESPONSE_SAVED",
                        "prompt_sha256": prompt_hash,
                        "model": expected_model,
                        "run_directory": str(run_directory),
                    }
                    _write_json(checkpoint_path, checkpoint)

                run_directory = run_experiment(
                    prompt_path=prompt_path,
                    output_root=output / "deepseek-runs",
                    config=config,
                    live=True,
                    response_saved=persist_saved_response,
                )
                live_requests += 1
                generation_status = "COMPLETE"

        error_status = _read_json(run_directory / "error-status.json")
        response_path = run_directory / "response.json"
        if error_status.get("status") != "NONE" or not response_path.is_file():
            checkpoint["Jobs"][job_id] = {
                "generation_status": "FAILED",
                "prompt_sha256": prompt_hash,
                "run_directory": str(run_directory),
                "failure": error_status,
            }
            _write_json(checkpoint_path, checkpoint)
            continue
        try:
            text = _response_text(response_path)
            proposal = _proposal_payload(text)
            if proposal.get("status") == "READY":
                parse_changed_files(proposal)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            checkpoint["Jobs"][job_id] = {
                "generation_status": "INVALID_RESPONSE",
                "prompt_sha256": prompt_hash,
                "run_directory": str(run_directory),
                "failure": str(error),
            }
            _write_json(checkpoint_path, checkpoint)
            continue

        state: dict[str, Any] = {
            "generation_status": generation_status,
            "prompt_sha256": prompt_hash,
            "run_directory": str(run_directory),
            "proposal_status": proposal["status"],
        }
        if proposal["status"] == "READY" and validation_config_path is not None:
            repo_config = validation_configs.get(job["Repository"])
            if not isinstance(repo_config, dict):
                state["validation"] = {
                    "status": "BLOCKED",
                    "reason": "No repository validation configuration was supplied.",
                }
            else:
                validation_directory = (
                    _new_validation_attempt(output, job_id)
                )
                state["validation"] = _validate_job(
                    job=job,
                    proposal=proposal,
                    config=repo_config,
                    output_directory=validation_directory,
                )
        else:
            state["validation"] = {
                "status": "NOT_RUN",
                "reason": (
                    "Validation configuration was not supplied."
                    if proposal["status"] == "READY"
                    else "Model returned INSUFFICIENT_CONTEXT."
                ),
            }
        checkpoint["Jobs"][job_id] = state
        _write_json(checkpoint_path, checkpoint)

    summary = _paired_summary(jobs, blocked_jobs, checkpoint)
    _write_json(output / "paired-results.json", summary)
    return {
        "Checkpoint": str(checkpoint_path.resolve()),
        "PairedResults": str((output / "paired-results.json").resolve()),
        "JobCount": len(jobs) + len(blocked_jobs),
        "LiveRequestsMadeThisRun": live_requests,
        "GenerationCounts": dict(
            __import__("collections").Counter(
                state.get("generation_status", "PENDING")
                for state in checkpoint["Jobs"].values()
            )
        ),
    }


def _mapping(values: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected repository=path mapping, got: {value}")
        repository, path = value.split("=", 1)
        mapping[repository] = Path(path)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--orchestration-root", type=Path, action="append", default=[])
    parser.add_argument("--repository-root", action="append", default=[])
    parser.add_argument("--corpus", action="append", default=[])
    parser.add_argument("--run-missing-retrieval", action="store_true")
    parser.add_argument("--embedding-model", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--retrieval-methods",
        nargs="+",
        default=["bm25", "dense", "hybrid"],
    )
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--endpoint", default=DeepSeekConfig.endpoint)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    parser.add_argument("--validation-config", type=Path)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Permit live DeepSeek requests; requires separate explicit approval.",
    )
    args = parser.parse_args()
    if args.jobs is not None:
        jobs_path = args.jobs
    else:
        if (
            args.dataset is None
            or not args.orchestration_root
            or not args.repository_root
            or not args.corpus
        ):
            parser.error(
                "Provide --jobs, or provide --dataset, --orchestration-root, "
                "--repository-root, and --corpus to prepare paired jobs."
            )
        prepared_directory = args.output / "paired-jobs"
        jobs_path = prepared_directory / "jobs.json"
        if not jobs_path.is_file():
            prepare_paired(
                dataset=args.dataset,
                orchestration_roots=args.orchestration_root,
                repository_roots=_mapping(args.repository_root),
                corpus_by_repository=_mapping(args.corpus),
                output=prepared_directory,
                retrieval_methods=tuple(args.retrieval_methods),
                run_missing_retrieval=args.run_missing_retrieval,
                model_path=args.embedding_model,
                device=args.device,
            )
    config = DeepSeekConfig(
        model=args.model,
        endpoint=args.endpoint,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    result = run(
        jobs_manifest=jobs_path,
        output=args.output / "execution" if args.jobs is None else args.output,
        config=config,
        live=args.live,
        reuse_roots=args.reuse_root,
        validation_config_path=args.validation_config,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
