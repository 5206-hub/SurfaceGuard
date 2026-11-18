"""Select inference settings on validation only, then score the full test split."""
import json
from pathlib import Path

import numpy as np

from surfaceguard.dataset import batch_iter, split_records
from surfaceguard.metrics import PixelCounts
from surfaceguard.network import load_checkpoint, predict
from surfaceguard.vision import classical, combine


def collect_predictions(model, root, keys):
    records = []
    for x, y in batch_iter(root, keys, 1):
        gray = (x[0, 0].numpy() * 255).astype(np.uint8)
        probabilities = predict(model, gray)
        classical_score, _ = classical(gray)
        records.append((probabilities, classical_score, y[0].numpy()))
    return records


def score_predictions(records, threshold, weight):
    binary = PixelCounts()
    by_label = {f"label_{index}": PixelCounts() for index in (1, 2, 3)}
    for probabilities, classical_score, truth in records:
        predicted = combine(probabilities, classical_score, weight=weight, threshold=threshold)
        binary.update(predicted > 0, truth > 0)
        for index in (1, 2, 3):
            by_label[f"label_{index}"].update(predicted == index, truth == index)
    return {
        "binary": binary.results(),
        "per_label": {label: counts.results() for label, counts in by_label.items()},
    }


def score_classical(root, keys):
    binary = PixelCounts()
    for x, y in batch_iter(root, keys, 1):
        gray = (x[0, 0].numpy() * 255).astype(np.uint8)
        _, prediction = classical(gray)
        binary.update(prediction > 0, y[0].numpy() > 0)
    return binary.results()


def main():
    root = Path("data")
    checkpoint_path = Path("models/best.pt")
    split = split_records(root)
    if len(split["test"]) != 840:
        raise ValueError(f"Expected the preserved 840-image NEU-SEG test split, found {len(split['test'])}")

    model, checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["dataset_digest"] != split["dataset_digest"]:
        raise ValueError("Checkpoint was trained on another image list; refusing evaluation")

    validation_records = collect_predictions(model, root, split["val"])
    print(f"Cached model/classical predictions for {len(validation_records)} validation images.", flush=True)
    thresholds = [round(step / 100, 2) for step in range(10, 96, 5)]
    threshold_sweep = []
    for index, threshold in enumerate(thresholds, start=1):
        scores = score_predictions(validation_records, threshold, weight=1.0)["binary"]
        threshold_sweep.append({"threshold": threshold, "scores": scores})
        print(f"CNN threshold sweep {index}/{len(thresholds)}", flush=True)
    cnn_choice = max(threshold_sweep, key=lambda row: (row["scores"]["dice"], row["threshold"]))
    selected_threshold = cnn_choice["threshold"]

    weights = [round(step / 100, 2) for step in range(0, 101, 5)]
    weight_sweep = []
    for index, weight in enumerate(weights, start=1):
        scores = score_predictions(validation_records, selected_threshold, weight)["binary"]
        weight_sweep.append({"weight": weight, "scores": scores})
        print(f"Hybrid weight sweep {index}/{len(weights)}", flush=True)
    hybrid_choice = max(weight_sweep, key=lambda row: (row["scores"]["dice"], -row["weight"]))
    selected_weight = hybrid_choice["weight"]

    selection = {
        "selected_on": "validation only",
        "selection_metric": "pooled binary Dice",
        "validation_images": len(split["val"]),
        "checkpoint_dataset_digest": split["dataset_digest"],
        "checkpoint_epoch": "unknown; epoch is not stored in checkpoint",
        "classical_threshold": 0.65,
        "cnn_foreground_threshold": selected_threshold,
        "hybrid_foreground_threshold": selected_threshold,
        "hybrid_cnn_weight": selected_weight,
        "hybrid_classical_weight": round(1.0 - selected_weight, 2),
        "threshold_tie_break": "higher threshold",
        "hybrid_weight_tie_break": "lower CNN weight",
        "cnn_validation_scores": score_predictions(validation_records, selected_threshold, 1.0),
        "hybrid_validation_scores": score_predictions(validation_records, selected_threshold, selected_weight),
        "cnn_threshold_sweep": threshold_sweep,
        "hybrid_weight_sweep": weight_sweep,
    }
    selection_path = Path("models/selection.json")
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print("Validation-only settings saved before test evaluation:")
    print(json.dumps({key: selection[key] for key in (
        "cnn_foreground_threshold", "hybrid_foreground_threshold", "hybrid_cnn_weight",
        "hybrid_classical_weight", "cnn_validation_scores", "hybrid_validation_scores",
    )}, indent=2))

    previous = {}
    results_path = Path("results.json")
    if results_path.is_file():
        old_report = json.loads(results_path.read_text(encoding="utf-8"))
        if old_report.get("images") == 150:
            previous = {
                "images": 150,
                "already_inspected": True,
                "note": "This subset was inspected in the earlier project run; it was not used to select the settings above.",
                "metrics": {
                    key: old_report[key]
                    for key in ("classical_binary", "cnn_binary", "hybrid_binary")
                    if key in old_report
                },
            }

    # Selection has been finalized and recorded before the test split is read.
    test_binary = {name: PixelCounts() for name in ("classical", "cnn", "hybrid")}
    test_labels = {
        name: {f"label_{index}": PixelCounts() for index in (1, 2, 3)}
        for name in ("cnn", "hybrid")
    }
    for image_index, (x, y) in enumerate(batch_iter(root, split["test"], 1), start=1):
        gray = (x[0, 0].numpy() * 255).astype(np.uint8)
        truth = y[0].numpy()
        classical_score, classical_mask = classical(gray)
        probabilities = predict(model, gray)
        cnn_prediction = combine(
            probabilities, classical_score, weight=1.0, threshold=selected_threshold
        )
        hybrid_prediction = combine(
            probabilities, classical_score, weight=selected_weight, threshold=selected_threshold
        )
        test_binary["classical"].update(classical_mask > 0, truth > 0)
        test_binary["cnn"].update(cnn_prediction > 0, truth > 0)
        test_binary["hybrid"].update(hybrid_prediction > 0, truth > 0)
        for index in (1, 2, 3):
            test_labels["cnn"][f"label_{index}"].update(cnn_prediction == index, truth == index)
            test_labels["hybrid"][f"label_{index}"].update(hybrid_prediction == index, truth == index)
        if image_index % 100 == 0:
            print(f"Test evaluation {image_index}/{len(split['test'])}", flush=True)

    report = {
        "split": "official test",
        "images": len(split["test"]),
        "evaluation_scope": "All 840 official test images. Settings were selected on validation before test evaluation.",
        "previous_test_subset": previous,
        "selected_settings": {
            key: selection[key] for key in (
                "cnn_foreground_threshold", "hybrid_foreground_threshold", "hybrid_cnn_weight",
                "hybrid_classical_weight", "classical_threshold",
            )
        },
        "classical_binary": test_binary["classical"].results(),
        "cnn_binary": test_binary["cnn"].results(),
        "hybrid_binary": test_binary["hybrid"].results(),
        "classical_per_label": "Not applicable: classical output is unclassified binary foreground.",
        "cnn_per_label": {label: counts.results() for label, counts in test_labels["cnn"].items()},
        "hybrid_per_label": {label: counts.results() for label, counts in test_labels["hybrid"].items()},
    }
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Full held-out test report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
