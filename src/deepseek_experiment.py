"""Prepare or execute one guarded BM25-to-DeepSeek experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from deepseek_client import DeepSeekClient, DeepSeekConfig, DeepSeekError


def _new_run_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = root / f"bm25-deepseek-{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"bm25-deepseek-{stamp}-{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _write_json(path: Path, value: Any) -> None:
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
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Response contains no choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Response contains no textual message content.")
    return message["content"]


def run_experiment(
    *,
    prompt_path: Path,
    output_root: Path,
    config: DeepSeekConfig,
    live: bool,
    response_saved: Callable[[Path], None] | None = None,
) -> Path:
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8")
    run_directory = _new_run_directory(output_root)
    model_config = {
        "mode": "LIVE" if live else "MOCK",
        "model": config.model,
        "endpoint": config.endpoint,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout_seconds": config.timeout_seconds,
        "live_request": live,
    }
    _write_json(run_directory / "model-config.json", model_config)
    (run_directory / "prompt.txt").write_text(prompt, encoding="utf-8")

    error_status: dict[str, Any] = {"status": "NONE", "message": None}
    response: dict[str, Any]
    if not live:
        response = {
            "mode": "MOCK",
            "model": config.model,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "MOCK RESPONSE: live DeepSeek was not called.",
                    }
                }
            ],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }
    else:
        try:
            response = DeepSeekClient(config).complete(prompt, allow_live_request=True)
        except DeepSeekError as error:
            error_status = {"status": "ERROR", "message": str(error)}
            response = {}

    _write_json(run_directory / "response.json", response)
    _write_json(run_directory / "error-status.json", error_status)
    if response_saved is not None:
        response_saved(run_directory)
    extracted_text = ""
    if response:
        try:
            extracted_text = _extract_text(response)
        except ValueError as error:
            error_status = {"status": "ERROR", "message": str(error)}
    (run_directory / "extracted-text.txt").write_text(extracted_text, encoding="utf-8")
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    _write_json(
        run_directory / "token-usage.json",
        {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    )
    return run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("artifacts/offline-bm25-smoke/prompt.txt"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/deepseek-experiments"),
    )
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--endpoint", default=DeepSeekConfig.endpoint)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Explicitly permit one live API request.",
    )
    args = parser.parse_args()
    config = DeepSeekConfig(
        model=args.model,
        endpoint=args.endpoint,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    run_directory = run_experiment(
        prompt_path=args.prompt,
        output_root=args.output_root,
        config=config,
        live=args.live,
    )
    print(f"MODE: {'LIVE' if args.live else 'MOCK'}")
    print(f"MODEL: {config.model}")
    print(f"OUTPUT: {run_directory.resolve()}")
    print("LIVE REQUEST: NOT MADE" if not args.live else "LIVE REQUEST: ATTEMPTED")


if __name__ == "__main__":
    main()
