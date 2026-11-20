"""Evaluate per-image detection counts from YOLO-format label files."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def count_labels(directory: Path) -> dict[str, int]:
    if not directory.is_dir():
        raise SystemExit(f"not a directory: {directory}")
    counts = {}
    for path in sorted(directory.glob("*.txt")):
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts[path.stem] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-labels", type=Path, required=True)
    parser.add_argument("--prediction-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, predicted = count_labels(args.reference_labels), count_labels(args.prediction_labels)
    missing = sorted(set(reference) - set(predicted))
    if missing:
        raise SystemExit(f"predictions missing for {len(missing)} reference images; missing labels are not zero counts")
    rows = []
    for image_id in sorted(reference):
        observed, estimate = reference[image_id], predicted[image_id]
        rows.append((image_id, observed, estimate, estimate - observed, abs(estimate - observed)))
    if not rows:
        raise SystemExit("no paired label files")
    errors = [row[3] for row in rows]
    refs = [row[1] for row in rows]
    mae = sum(abs(x) for x in errors) / len(rows)
    rmse = math.sqrt(sum(x * x for x in errors) / len(rows))
    mean_ref = sum(refs) / len(refs)
    sst = sum((x - mean_ref) ** 2 for x in refs)
    metrics = {
        "n_images": len(rows), "mae": mae, "rmse": rmse,
        "rrmse_percent": None if mean_ref == 0 else 100 * rmse / mean_ref,
        "r2": None if sst == 0 else 1 - sum(x * x for x in errors) / sst,
        "mean_signed_error": sum(errors) / len(errors),
        "aggregate_bias_percent": None if sum(refs) == 0 else 100 * sum(errors) / sum(refs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "reference_count", "predicted_count", "signed_error", "absolute_error"])
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
