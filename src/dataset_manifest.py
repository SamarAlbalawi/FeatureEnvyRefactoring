"""Create a reproducible manifest for every labeled dataset case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlparse

from dataset_loader import load_dataset


GITHUB_TREE_PATTERN = re.compile(
    r"^/(?P<owner>[^/]+)/(?P<repository>[^/]+)/tree/"
    r"(?P<commit>[^/]+)/(?P<path>.+)$"
)


def parse_source_url(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise ValueError(f"Expected a GitHub HTTPS URL: {url}")
    match = GITHUB_TREE_PATTERN.match(parsed.path)
    if match is None:
        raise ValueError(f"Could not parse GitHub source URL: {url}")
    fragment = parsed.fragment
    lines = re.fullmatch(r"L(?P<start>\d+)(?:-L(?P<end>\d+))?", fragment)
    return {
        "Repository": f"{match.group('owner')}/{match.group('repository')}",
        "Commit": match.group("commit"),
        "File": unquote(match.group("path")),
        "StartLine": int(lines.group("start")) if lines else None,
        "EndLine": int(lines.group("end")) if lines and lines.group("end") else None,
    }


def build_manifest(dataset_path: str | Path) -> dict[str, Any]:
    records = load_dataset(dataset_path)
    cases: list[dict[str, Any]] = []
    for _, record in records.iterrows():
        source = parse_source_url(str(record["Link"]))
        cases.append(
            {
                "CaseId": str(record["Code Snippet ID"]),
                "FeatureEnvyLabel": int(record["Final annotation"]) > 0,
                "Annotation": int(record["Final annotation"]),
                "CodeSmellField": str(record["Code Smell"]),
                "ProjectUrl": str(record["Project Link"]),
                **source,
            }
        )
    repositories = sorted(
        {
            (case["Repository"], case["Commit"])
            for case in cases
        }
    )
    return {
        "SchemaVersion": "dataset-manifest-v1",
        "DatasetPath": str(Path(dataset_path).resolve()),
        "CaseCount": len(cases),
        "PositiveFeatureEnvyCases": sum(case["FeatureEnvyLabel"] for case in cases),
        "NegativeFeatureEnvyCases": sum(not case["FeatureEnvyLabel"] for case in cases),
        "RepositoryRevisionCount": len(repositories),
        "RepositoryRevisions": [
            {"Repository": repository, "Commit": commit}
            for repository, commit in repositories
        ],
        "Cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"CASES: {manifest['CaseCount']}")
    print(f"POSITIVE FEATURE ENVY CASES: {manifest['PositiveFeatureEnvyCases']}")
    print(f"REPOSITORY REVISIONS: {manifest['RepositoryRevisionCount']}")
    print(f"OUTPUT: {args.output.resolve()}")


if __name__ == "__main__":
    main()
