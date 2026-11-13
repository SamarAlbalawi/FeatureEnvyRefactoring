"""Invoke the existing Roslyn analyzer and return its JSON result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any


class RoslynBridgeError(RuntimeError):
    """Raised when the Roslyn analyzer cannot be invoked or parsed."""


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RoslynBridgeError(f"{label} does not exist or is not a file: {path}")


def _dotnet_root(msbuild: Path) -> Path:
    """Resolve either an SDK root, versioned SDK, or dotnet root input."""
    if msbuild.name.lower() == "sdk":
        return msbuild.parent
    if msbuild.parent.name.lower() == "sdk":
        return msbuild.parent.parent
    return msbuild


def invoke_roslyn(
    *,
    analyzer_path: str | Path,
    project_path: str | Path,
    source_file_path: str | Path,
    msbuild_path: str | Path,
    method_name: str,
    containing_type: str | None = None,
    signature: str | None = None,
    source_line: int | None = None,
    repository: str = "unknown",
    commit: str = "unknown",
    dataset_annotation: str | None = None,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run Roslyn and parse the dependency-analysis JSON output."""
    analyzer = Path(analyzer_path)
    project = Path(project_path)
    source_file = Path(source_file_path)
    msbuild = Path(msbuild_path)
    output = Path(output_path)

    _require_file(analyzer, "Roslyn analyzer")
    _require_file(project, "C# project")
    _require_file(source_file, "C# source file")
    if not msbuild.is_dir():
        raise RoslynBridgeError(
            f"ROSLYN_MSBUILD_PATH does not exist or is not a directory: {msbuild}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    dotnet_root = _dotnet_root(msbuild)
    dotnet_host = dotnet_root / "dotnet.exe"
    if analyzer.suffix.lower() == ".dll":
        _require_file(dotnet_host, "local .NET host")
        command = [str(dotnet_host), str(analyzer)]
    else:
        command = [str(analyzer)]
    command.extend([
        "--project",
        str(project),
        "--source-file",
        str(source_file),
        "--method",
        method_name,
        "--repository",
        repository,
        "--commit",
        commit,
        "--output",
        str(output),
    ])
    if containing_type:
        command.extend(["--containing-type", containing_type])
    if signature:
        command.extend(["--signature", signature])
    if source_line is not None:
        command.extend(["--source-line", str(source_line)])
    if dataset_annotation:
        command.extend(["--dataset-annotation", dataset_annotation])

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
        raise RoslynBridgeError(
            "Roslyn analyzer failed with exit code "
            f"{completed.returncode}.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    if not output.is_file():
        raise RoslynBridgeError(
            f"Roslyn completed successfully but output JSON is missing: {output}"
        )

    try:
        with output.open("r", encoding="utf-8") as result_file:
            return json.load(result_file)
    except json.JSONDecodeError as error:
        raise RoslynBridgeError(
            f"Roslyn output is not valid JSON: {output}"
        ) from error
    except OSError as error:
        raise RoslynBridgeError(f"Could not read Roslyn output: {output}") from error


def _verification_summary(args: argparse.Namespace) -> None:
    result = invoke_roslyn(
        analyzer_path=args.analyzer,
        project_path=args.project,
        source_file_path=args.source_file,
        msbuild_path=args.msbuild,
        method_name=args.method,
        containing_type=args.containing_type,
        signature=args.signature,
        repository=args.repository,
        commit=args.commit,
        output_path=args.output,
    )
    target = result.get("Target", {})
    repository = result.get("Repository", {})
    candidates = result.get("LikelyEnviedTypeCandidates", [])
    strongest = candidates[0] if candidates else {}
    print(f"REPOSITORY: {repository.get('RepositoryIdentifier', '')}")
    print(f"COMMIT: {repository.get('Commit', '')}")
    print(f"TARGET: {target.get('ContainingType', '')}.{target.get('MethodName', '')}")
    print(f"FOREIGN DEPENDENCIES: {len(result.get('ForeignDependencies', []))}")
    print(
        "STRONGEST ENVIED TYPE: "
        f"{strongest.get('DeclaringType', '')}"
    )
    print(f"ENVIED TYPE SCORE: {strongest.get('Score', '')}")
    print(f"UNRESOLVED DEPENDENCIES: {len(result.get('UnresolvedDependencies', []))}")
    print(f"OUTPUT: {Path(args.output).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--msbuild", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--containing-type")
    parser.add_argument("--signature")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    _verification_summary(parser.parse_args())


if __name__ == "__main__":
    main()
