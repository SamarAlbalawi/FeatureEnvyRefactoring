"""Evaluate Feature Envy predictions against the labeled dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset_loader import load_dataset


def _load_predictions(path: Path) -> dict[str, bool]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid prediction JSON: {path}") from error
    records = value.get("Predictions") if isinstance(value, dict) else value
    if not isinstance(records, list):
        raise ValueError("Predictions must be a JSON list or {\"Predictions\": [...]} object.")
    predictions: dict[str, bool] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Prediction {index} must be an object.")
        case_id = record.get("CaseId")
        prediction = record.get("FeatureEnvy")
        if not isinstance(case_id, str) or not isinstance(prediction, bool):
            raise ValueError(f"Prediction {index} requires CaseId and boolean FeatureEnvy.")
        if case_id in predictions:
            raise ValueError(f"Duplicate prediction for case: {case_id}")
        predictions[case_id] = prediction
    return predictions


def evaluate(dataset_path: str | Path, predictions_path: str | Path) -> dict[str, Any]:
    records = load_dataset(dataset_path)
    predictions = _load_predictions(Path(predictions_path))
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    missing: list[str] = []
    for _, record in records.iterrows():
        case_id = str(record["Code Snippet ID"])
        if case_id not in predictions:
            missing.append(case_id)
            continue
        actual = int(record["Final annotation"]) > 0
        predicted = predictions[case_id]
        if actual and predicted:
            counts["TP"] += 1
        elif not actual and not predicted:
            counts["TN"] += 1
        elif not actual and predicted:
            counts["FP"] += 1
        else:
            counts["FN"] += 1

    evaluated = sum(counts.values())
    precision = (
        counts["TP"] / (counts["TP"] + counts["FP"])
        if counts["TP"] + counts["FP"]
        else None
    )
    recall = (
        counts["TP"] / (counts["TP"] + counts["FN"])
        if counts["TP"] + counts["FN"]
        else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "SchemaVersion": "feature-envy-detection-evaluation-v1",
        "DatasetCases": len(records),
        "PredictionsProvided": len(predictions),
        "EvaluatedCases": evaluated,
        "MissingPredictionCount": len(missing),
        "MissingCaseIds": missing,
        "ConfusionMatrix": counts,
        "Metrics": {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": (counts["TP"] + counts["TN"]) / evaluated if evaluated else None,
        },
        "Complete": not missing and len(predictions) == len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"EVALUATED CASES: {result['EvaluatedCases']}")
    print(f"MISSING PREDICTIONS: {result['MissingPredictionCount']}")
    print(f"COMPLETE: {'YES' if result['Complete'] else 'NO'}")
    print(f"OUTPUT: {args.output.resolve()}")


if __name__ == "__main__":
    main()
