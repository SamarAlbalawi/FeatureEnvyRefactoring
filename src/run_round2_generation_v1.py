"""Prepare and run the independent, at-most-once second generation round."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib import error, request

import msvcrt

from patch_normalizer import parse_changed_files
from prompt_builder import prompt_sha256
from run_refactoring_experiment import _proposal_payload


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "artifacts" / "paired-generation-prep-v6" / "jobs.json"
ROUND_ONE = ROOT / "artifacts" / "paired-generation-live-v6"
OUTPUT = ROOT / "artifacts" / "paired-generation-live-round2-v1"
SETTINGS = {
    "model": "deepseek-flash",
    "endpoint": "https://api.deepseek.com/chat/completions",
    "temperature": 0.0,
    "max_tokens": 4096,
    "timeout_seconds": 60.0,
    "thinking": "disabled",
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_json_new(path: Path, value: object) -> None:
    write_new(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def inspect_inputs() -> dict:
    manifest = read_json(MANIFEST)
    if any(manifest.get(key) != value for key, value in {
        "Model": SETTINGS["model"], "Temperature": SETTINGS["temperature"],
        "MaxTokens": SETTINGS["max_tokens"], "Thinking": SETTINGS["thinking"],
        "RetrievalMethods": ["bm25", "dense", "hybrid"],
        "Conditions": ["baseline", "dependency_augmented"],
    }.items()):
        raise ValueError("Manifest generation settings differ from the first round.")
    jobs = manifest["Jobs"]
    blocked = manifest["BlockedJobs"]
    if len(jobs) != 210 or len(blocked) != 6:
        raise ValueError("Expected 210 prepared jobs and six blocked jobs.")
    ids = [job["JobId"] for job in jobs]
    if len(set(ids)) != 210 or set(ids) & {job["JobId"] for job in blocked}:
        raise ValueError("Duplicate or overlapping job IDs.")
    checkpoint = read_json(ROUND_ONE / "checkpoint.json")["Jobs"]
    pairs: dict[str, list[dict]] = defaultdict(list)
    frozen_jobs = []
    for job in jobs:
        jid = job["JobId"]
        state = checkpoint[jid]
        if job.get("Mode") != "PREPARED_ONLY":
            raise ValueError(f"Job is not prepared: {jid}")
        source_prompt = Path(job["PromptPath"])
        raw = source_prompt.read_bytes()
        logical_hash = prompt_sha256(raw)
        if logical_hash != job["PromptSHA256"] or logical_hash != state.get("prompt_sha256"):
            raise ValueError(f"Manifest or first-round checkpoint prompt mismatch: {jid}")
        old_run = Path(state["run_directory"])
        if source_prompt.read_text(encoding="utf-8") != (old_run / "prompt.txt").read_text(encoding="utf-8"):
            raise ValueError(f"First-round request prompt differs: {jid}")
        config = read_json(old_run / "model-config.json")
        for field in ("model", "endpoint", "temperature", "max_tokens", "timeout_seconds"):
            if config.get(field) != SETTINGS[field]:
                raise ValueError(f"First-round {field} differs: {jid}")
        if config.get("mode") != "LIVE" or config.get("live_request") is not True:
            raise ValueError(f"First-round live-request settings unavailable: {jid}")
        retrieval_path = Path(job["RetrievalResultPath"])
        retrieval = read_json(retrieval_path)
        ranked = [item["DocumentId"] for item in retrieval["Results"][:10]]
        if len(ranked) != 10 or ranked != job["RetrievedDocumentIds"]:
            raise ValueError(f"Top-10 retrieval evidence differs: {jid}")
        prompt = source_prompt.read_text(encoding="utf-8")
        context = prompt.split("## 4. Retrieved repository context\n\n", 1)
        if len(context) != 2:
            raise ValueError(f"Retrieved-context prompt section missing: {jid}")
        extracted_ids = [
            line.removeprefix("Document ID: ")
            for line in context[1].splitlines()
            if line.startswith("Document ID: ")
        ]
        if extracted_ids != ranked:
            raise ValueError(f"Prompt's ranked document IDs differ: {jid}")
        pairs[job["PairId"]].append(job)
        frozen_jobs.append({
            "job_id": jid,
            "pair_id": job["PairId"],
            "prompt_path": str(source_prompt),
            "prompt_sha256": logical_hash,
            "prompt_file_sha256": hashlib.sha256(raw).hexdigest(),
            "retrieval_path": str(retrieval_path),
            "retrieval_sha256": sha256(retrieval_path),
            "analysis_path": job["AnalysisPath"],
            "analysis_sha256": sha256(Path(job["AnalysisPath"])),
            "first_round_status": state.get("generation_status"),
        })
    if len(pairs) != 105:
        raise ValueError("Expected 105 matched pairs.")
    for pair_id, members in pairs.items():
        if len(members) != 2 or {x["Condition"] for x in members} != {"baseline", "dependency_augmented"}:
            raise ValueError(f"Invalid condition pairing: {pair_id}")
        for key in ("CaseId", "Repository", "Commit", "SourcePath", "TargetDocumentId", "RetrievedDocumentIds", "RetrievalMethod"):
            if members[0][key] != members[1][key]:
                raise ValueError(f"Pair mismatch for {key}: {pair_id}")
        by_condition = {member["Condition"]: member for member in members}
        baseline = Path(by_condition["baseline"]["PromptPath"]).read_text(encoding="utf-8")
        augmented = Path(by_condition["dependency_augmented"]["PromptPath"]).read_text(encoding="utf-8")
        baseline_context = baseline.split("## 4. Retrieved repository context\n\n", 1)[1].split("\n\n## 5. Required output format", 1)[0]
        augmented_context = augmented.split("## 4. Retrieved repository context\n\n", 1)[1].split("\n\n## 5. Roslyn-derived dependency information", 1)[0]
        if baseline_context != augmented_context:
            raise ValueError(f"Paired retrieved-context text differs: {pair_id}")
    if Counter(job["RetrievalMethod"] for job in jobs) != {"bm25": 70, "dense": 70, "hybrid": 70}:
        raise ValueError("Retrieval method counts are unbalanced.")
    return {
        "schema_version": "paired-generation-round2-preflight-v1",
        "round_id": 2,
        "manifest_path": str(MANIFEST),
        "manifest_sha256": sha256(MANIFEST),
        "round_one_checkpoint_sha256": sha256(ROUND_ONE / "checkpoint.json"),
        "runner_sha256": sha256(Path(__file__)),
        "request_settings": SETTINGS,
        "request_order": "Original manifest Jobs order; blocked jobs excluded.",
        "job_count": len(jobs),
        "pair_count": len(pairs),
        "jobs": frozen_jobs,
    }


def job_directory(index: int, jid: str) -> Path:
    return OUTPUT / "jobs" / f"{index:03d}-{hashlib.sha256(jid.encode()).hexdigest()[:16]}"


def next_action(directory_exists: bool) -> str:
    # Any existing reservation is skipped, including interrupted requests.
    return "SKIP_RESERVED" if directory_exists else "RESERVE_ONCE"


def prepare() -> None:
    if OUTPUT.exists() and not (OUTPUT / "preflight.json").is_file():
        raise FileExistsError(f"Refusing to use an unrelated output directory: {OUTPUT}")
    frozen = inspect_inputs()
    if next_action(False) != "RESERVE_ONCE" or next_action(True) != "SKIP_RESERVED":
        raise RuntimeError("Resume reservation invariant failed.")
    if not OUTPUT.exists():
        OUTPUT.mkdir(parents=True)
    write_json_new(OUTPUT / "preflight-final.json", frozen)
    print(json.dumps({"jobs": frozen["job_count"], "pairs": frozen["pair_count"],
                      "settings": SETTINGS, "round_id": 2, "api_calls": 0}, indent=2))


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def classify(body: bytes) -> dict:
    try:
        value = json.loads(body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Response is not a JSON object.")
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("Response has no choices.")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Response has no textual message content.")
        result = {"finish_reason": choice.get("finish_reason"), "usage": value.get("usage")}
        try:
            proposal = _proposal_payload(message["content"])
            if proposal["status"] == "READY":
                parse_changed_files(proposal)
        except (ValueError, KeyError) as exc:
            result.update(status="INVALID_RESPONSE", error=str(exc))
        else:
            result.update(status="COMPLETE", proposal_status=proposal["status"])
        return result
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        return {"status": "INVALID_RESPONSE", "error": str(exc), "finish_reason": None, "usage": None}


def live() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
    frozen = read_json(OUTPUT / "preflight-final.json")
    if inspect_inputs() != frozen:
        raise RuntimeError("Input evidence changed since preparation; no requests sent.")
    lock_path = OUTPUT / "runner.lock"
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        if lock.read(1) == b"":
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another round-two runner holds the lock.") from exc
        try:
            execute(frozen)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def execute(frozen: dict) -> None:
    jobs = read_json(MANIFEST)["Jobs"]
    (OUTPUT / "jobs").mkdir(exist_ok=True)
    opener = request.build_opener(NoRedirect)
    for index, (job, evidence) in enumerate(zip(jobs, frozen["jobs"]), 1):
        directory = job_directory(index, job["JobId"])
        if next_action(directory.exists()) == "SKIP_RESERVED":
            print(f"SKIP {job['JobId']} (already reserved; never automatically resent)", flush=True)
            continue
        if (sha256(Path(evidence["prompt_path"])) != evidence["prompt_file_sha256"]
                or sha256(Path(evidence["retrieval_path"])) != evidence["retrieval_sha256"]
                or sha256(Path(evidence["analysis_path"])) != evidence["analysis_sha256"]):
            raise RuntimeError(f"Input evidence changed before request: {job['JobId']}")
        # mkdir is the persistent at-most-once reservation. An interruption
        # after it leaves an indeterminate attempt, which is never resent.
        directory.mkdir()
        prompt = Path(job["PromptPath"]).read_text(encoding="utf-8")
        write_new(directory / "prompt.txt", prompt.encode("utf-8"))
        write_json_new(directory / "attempt.json", {
            "round_id": 2, "job_id": job["JobId"], "pair_id": job["PairId"],
            "prompt_sha256": evidence["prompt_sha256"], "settings": SETTINGS,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "response_provenance": "new round-two request; no reuse",
        })
        payload = json.dumps({
            "model": SETTINGS["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": SETTINGS["temperature"],
            "max_tokens": SETTINGS["max_tokens"],
            "thinking": {"type": SETTINGS["thinking"]},
        }).encode("utf-8")
        req = request.Request(
            SETTINGS["endpoint"], data=payload,
            headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"],
                     "Content-Type": "application/json"}, method="POST",
        )
        try:
            with opener.open(req, timeout=SETTINGS["timeout_seconds"]) as response:
                body = response.read()
                http_status = response.status
        except error.HTTPError as exc:
            body = exc.read()
            write_new(directory / "response.raw", body)
            write_json_new(directory / "result.json", {
                "round_id": 2, "job_id": job["JobId"], "pair_id": job["PairId"],
                "status": "FAILED", "http_status": exc.code,
                "error": f"DeepSeek HTTP error {exc.code}.",
                "finish_reason": None, "usage": None,
                "response_sha256": hashlib.sha256(body).hexdigest(),
                "response_path": str(directory / "response.raw"),
            })
            print(f"FAILED {job['JobId']} (HTTP {exc.code})", flush=True)
            continue
        except (error.URLError, TimeoutError) as exc:
            message = "DeepSeek request timed out." if isinstance(exc, TimeoutError) else "DeepSeek request could not be sent."
            write_json_new(directory / "result.json", {
                "round_id": 2, "job_id": job["JobId"], "pair_id": job["PairId"],
                "status": "FAILED", "error": message, "finish_reason": None, "usage": None,
                "response_provenance": "no response received",
            })
            print(f"FAILED {job['JobId']} ({message})", flush=True)
            continue
        write_new(directory / "response.raw", body)
        result = classify(body)
        result.update({
            "round_id": 2, "job_id": job["JobId"], "pair_id": job["PairId"],
            "http_status": http_status, "response_path": str(directory / "response.raw"),
            "response_sha256": hashlib.sha256(body).hexdigest(),
            "response_provenance": "new round-two request; exact response bytes",
        })
        write_json_new(directory / "result.json", result)
        print(f"{result['status']} {job['JobId']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        live()


if __name__ == "__main__":
    main()
