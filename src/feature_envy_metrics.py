"""Literature-based ATFD/LAA/FDP evaluation from Roslyn analysis JSON.

Definitions used here:

* ATFD (Access To Foreign Data) is the number of distinct resolved
  field/property symbols whose Roslyn ``Scope`` is ``FOREIGN``.
* LAA (Locality of Attribute Accesses) is distinct local resolved
  field/property symbols divided by all distinct resolved data symbols.
* FDP (Foreign Data Providers) is the number of distinct declaring types of
  foreign field/property accesses.

Only field and property accesses are data accesses. Method calls, type
references, and callees are never counted. Roslyn records each source
occurrence for evidence, but repeated accesses to one symbol count once for
ATFD and LAA. A property access uses the resolved property symbol; its getter
is accessor semantics of that property, not a separate data attribute or
arbitrary method call.

Scope is taken from Roslyn's semantic classification. Inherited members are
not reclassified from that evidence: if Roslyn marks one foreign, it is
foreign; if inheritance or a member scope is unresolved, the result is
``UNCERTAIN`` instead of guessing. Unresolved references are reported and
prevent a definitive classification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ATFD_THRESHOLD = 5
LAA_THRESHOLD = 1 / 3
FDP_THRESHOLD = 5
DATA_COLLECTIONS = ("Fields", "Properties")


def _references(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for collection in DATA_COLLECTIONS:
        values = analysis.get(collection, [])
        if not isinstance(values, list):
            continue
        references.extend(value for value in values if isinstance(value, dict))
    return references


def evaluate_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one Roslyn analysis result without inventing unresolved values."""
    references = _references(analysis)
    unresolved = [
        value for value in analysis.get("UnresolvedDependencies", [])
        if isinstance(value, dict)
    ]
    invalid_scope = [
        reference for reference in references
        if reference.get("Scope") not in {"LOCAL", "FOREIGN"}
    ]
    invalid_identity = [
        reference for reference in references
        if not isinstance(reference.get("Symbol"), str)
        or not reference["Symbol"].strip()
    ]
    local = [reference for reference in references if reference.get("Scope") == "LOCAL"]
    foreign = [reference for reference in references if reference.get("Scope") == "FOREIGN"]
    invalid_foreign_type = [
        reference for reference in foreign
        if not isinstance(reference.get("DeclaringType"), str)
        or not reference["DeclaringType"].strip()
    ]
    local_symbols = {
        reference["Symbol"] for reference in local
        if isinstance(reference.get("Symbol"), str) and reference["Symbol"].strip()
    }
    foreign_symbols = {
        reference["Symbol"] for reference in foreign
        if isinstance(reference.get("Symbol"), str) and reference["Symbol"].strip()
    }
    foreign_types = sorted(
        {
            str(reference["DeclaringType"])
            for reference in foreign
            if isinstance(reference.get("DeclaringType"), str)
        }
    )
    uncertainty_reasons: list[str] = []
    if unresolved:
        uncertainty_reasons.append("Roslyn reported unresolved dependencies.")
    if invalid_scope:
        uncertainty_reasons.append(
            "One or more field/property accesses lack LOCAL or FOREIGN scope."
        )
    if invalid_identity:
        uncertainty_reasons.append(
            "One or more field/property accesses lack a resolved semantic symbol identity."
        )
    if invalid_foreign_type:
        uncertainty_reasons.append(
            "One or more foreign data accesses lack a resolved declaring type."
        )
    resolved_count = len(local_symbols) + len(foreign_symbols)
    laa = (len(local_symbols) / resolved_count) if resolved_count else None
    metrics = {
        "ATFD": len(foreign_symbols),
        "LAA": laa,
        "FDP": len(foreign_types),
    }
    classification = None
    if not uncertainty_reasons:
        classification = (
            metrics["ATFD"] > ATFD_THRESHOLD
            and metrics["LAA"] < LAA_THRESHOLD
            and metrics["FDP"] <= FDP_THRESHOLD
        )
    target = analysis.get("Target", {})
    return {
        "SchemaVersion": "feature-envy-metrics-v2",
        "Target": target,
        "Status": "UNCERTAIN" if uncertainty_reasons else "DEFINITE",
        "FeatureEnvy": classification,
        "Thresholds": {
            "ATFD": {"operator": ">", "value": ATFD_THRESHOLD},
            "LAA": {"operator": "<", "value": LAA_THRESHOLD},
            "FDP": {"operator": "<=", "value": FDP_THRESHOLD},
            "Combination": "AND",
        },
        "Metrics": metrics,
        "Evidence": {
            "LocalDataAccessOccurrences": local,
            "ForeignDataAccessOccurrences": foreign,
            "LocalDataAccessAttributes": sorted(local_symbols),
            "ForeignDataAccessAttributes": sorted(foreign_symbols),
            "ForeignDeclaringTypes": foreign_types,
            "ResolvedDataAccessCount": resolved_count,
            "MethodCallsExcluded": len(
                [value for value in analysis.get("Callees", []) if isinstance(value, dict)]
            ),
            "UnresolvedReferences": unresolved,
            "UncertaintyReasons": uncertainty_reasons,
            "RepeatedAccessPolicy": (
                "Occurrences are preserved as evidence; each resolved semantic "
                "field/property symbol counts once for ATFD/LAA."
            ),
            "InheritedMemberPolicy": (
                "Use Roslyn Scope; do not reclassify unresolved inheritance."
            ),
        },
    }


def evaluate_file(path: str | Path) -> dict[str, Any]:
    analysis_path = Path(path)
    try:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid Roslyn analysis JSON: {analysis_path}") from error
    if not isinstance(analysis, dict):
        raise ValueError(f"Expected a JSON object: {analysis_path}")
    result = evaluate_analysis(analysis)
    result["AnalysisPath"] = str(analysis_path.resolve())
    return result


def evaluate_files(paths: Iterable[str | Path]) -> dict[str, Any]:
    results = [evaluate_file(path) for path in paths]
    return {
        "SchemaVersion": "feature-envy-metrics-report-v2",
        "Definitions": {
            "ATFD": "Distinct resolved foreign field/property semantic symbols.",
            "LAA": "Distinct local data symbols divided by all distinct resolved data symbols.",
            "FDP": "Distinct declaring types of resolved foreign field/property accesses.",
            "DataAccesses": "Fields and properties only; method calls and type references excluded.",
        },
        "Results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_files(args.analysis)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for result in report["Results"]:
        print(
            f"{result['Target'].get('MethodName', '')}: "
            f"status={result['Status']} metrics={result['Metrics']} "
            f"feature_envy={result['FeatureEnvy']}"
        )
    print(f"OUTPUT: {output.resolve()}")


if __name__ == "__main__":
    main()
