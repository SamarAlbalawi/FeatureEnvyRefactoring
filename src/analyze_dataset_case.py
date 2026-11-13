"""Analyze one dataset case with the existing Roslyn dependency analyzer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from dataset_loader import load_dataset, positive_cases
from roslyn_bridge import invoke_roslyn


TARGET_METHOD_IDENTIFIER = (
    "Newtonsoft.Json.Serialization.DefaultContractResolver."
    "CreatePropertyFromConstructorParameter("
)
GITHUB_TREE_PATTERN = re.compile(
    r"^/(?P<owner>[^/]+)/(?P<repository>[^/]+)/tree/"
    r"(?P<commit>[^/]+)/(?P<path>.+)$"
)


@dataclass(frozen=True)
class DatasetCase:
    method_identifier: str
    repository: str
    commit: str
    source_path: str
    start_line: int | None
    end_line: int | None
    severity: int
    project_url: str


def _parse_github_url(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise ValueError(f"Expected a GitHub HTTPS URL: {url}")
    match = GITHUB_TREE_PATTERN.match(parsed.path)
    if match is None:
        raise ValueError(f"Could not parse GitHub repository URL: {url}")
    repository = f"{match.group('owner')}/{match.group('repository')}"
    return repository, match.group("commit"), unquote(match.group("path"))


def _parse_source_location(url: str) -> tuple[str, int | None, int | None]:
    repository, commit, path = _parse_github_url(url)
    fragment = urlparse(url).fragment
    lines = re.fullmatch(r"L(?P<start>\d+)(?:-L(?P<end>\d+))?", fragment)
    start_line = int(lines.group("start")) if lines else None
    end_line = int(lines.group("end")) if lines and lines.group("end") else None
    return f"{repository}|{commit}|{path}", start_line, end_line


def _parse_target_identifier(identifier: str) -> tuple[str, str]:
    method_start = identifier.rfind(".", 0, identifier.index("("))
    if method_start < 0:
        raise ValueError(f"Could not parse method identifier: {identifier}")
    containing_type = identifier[:method_start]
    method_name = identifier[method_start + 1 : identifier.index("(")]
    if "<" in method_name:
        method_name = method_name.split("<", 1)[0]
    return containing_type, method_name


def select_dataset_case(
    dataset_path: str | Path,
    case_id: str | None = None,
    *,
    positive_only: bool = True,
) -> DatasetCase:
    records = load_dataset(dataset_path)
    if positive_only:
        records = positive_cases(records)
    if case_id is None:
        matches = records[
            records["Code Snippet ID"].astype(str).str.startswith(TARGET_METHOD_IDENTIFIER)
        ]
    else:
        matches = records[records["Code Snippet ID"].astype(str) == case_id]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one dataset case, "
            f"found {len(matches)}."
        )

    record = matches.iloc[0]
    source_location, start_line, end_line = _parse_source_location(str(record["Link"]))
    repository, commit, source_path = source_location.split("|", maxsplit=2)
    return DatasetCase(
        method_identifier=str(record["Code Snippet ID"]),
        repository=repository,
        commit=commit,
        source_path=source_path,
        start_line=start_line,
        end_line=end_line,
        severity=int(record["Final annotation"]),
        project_url=str(record["Project Link"]),
    )


def _find_local_source(repository_root: Path, source_path: str) -> Path:
    source_file = repository_root / Path(*source_path.split("/"))
    if not source_file.is_file():
        raise FileNotFoundError(f"Dataset source file is not present locally: {source_file}")
    return source_file


def _run(args: argparse.Namespace) -> None:
    case = select_dataset_case(args.dataset, args.case_id)
    containing_type, method_name = _parse_target_identifier(case.method_identifier)
    source_file = _find_local_source(args.repository_root, case.source_path)
    result = invoke_roslyn(
        analyzer_path=args.analyzer,
        project_path=args.project,
        source_file_path=source_file,
        msbuild_path=args.msbuild,
        method_name=method_name,
        containing_type=containing_type,
        repository=case.repository,
        commit=case.commit,
        dataset_annotation=str(case.severity),
        output_path=args.output,
    )

    candidates = result.get("LikelyEnviedTypeCandidates", [])
    strongest = candidates[0] if candidates else {}
    foreign_count = len(result.get("ForeignDependencies", []))
    unresolved_count = len(result.get("UnresolvedDependencies", []))
    print(f"METHOD: {case.method_identifier}")
    print(f"SEVERITY: {case.severity}")
    print(f"REPOSITORY: {case.repository}")
    print(f"COMMIT: {case.commit}")
    print(f"SOURCE: {case.source_path}#L{case.start_line}-L{case.end_line}")
    print(f"FOREIGN DEPENDENCIES: {foreign_count}")
    print(f"STRONGEST ENVIED TYPE: {strongest.get('DeclaringType', '')}")
    print(f"ENVIED TYPE SCORE: {strongest.get('Score', '')}")
    print(f"UNRESOLVED DEPENDENCIES: {unresolved_count}")
    print(f"OUTPUT: {Path(args.output).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--msbuild", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
