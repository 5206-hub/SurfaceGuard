"""Select new-checkpoint settings on validation, then compare on all test images."""
import json
from pathlib import Path

import numpy as np

from surfaceguard.dataset import batch_iter, split_records
from surfaceguard.metrics import PixelCounts
from surfaceguard.network import load_checkpoint, predict
from surfaceguard.vision import classical, combine


def cache_predictions(model, root, keys):
    cached = []
    for x, y in batch_iter(root, keys, 1):
        gray = (x[0, 0].numpy() * 255).astype(np.uint8)
        probabilities = predict(model, gray)
        classical_score, _ = classical(gray)
        cached.append((probabilities, classical_score, y[0].numpy()))
    return cached


def class_scores(records, threshold, weight):
    counts = {f"label_{index}": PixelCounts() for index in (1, 2, 3)}
    for probabilities, classical_score, truth in records:
        prediction = combine(probabilities, classical_score, weight=weight, threshold=threshold)
        for index in (1, 2, 3):
            counts[f"label_{index}"].update(prediction == index, truth == index)
    per_label = {label: metric.results() for label, metric in counts.items()}
    mean_dice = float(np.mean([per_label[f"label_{index}"]["dice"] or 0.0 for index in (1, 2, 3)]))
    return {"mean_label_dice": round(mean_dice, 4), "per_label": per_label}


def main():
    root = Path("data")
    split = split_records(root)
    if len(split["test"]) != 840:
        raise ValueError(f"Expected all 840 official test images, found {len(split['test'])}")

    baseline_path = Path("models/best.pt")
    new_path = Path("models/best_perclass.pt")
    baseline, baseline_meta = load_checkpoint(baseline_path)
    new_model, new_meta = load_checkpoint(new_path)
    for name, metadata in (("baseline", baseline_meta), ("new", new_meta)):
        if metadata["dataset_digest"] != split["dataset_digest"]:
            raise ValueError(f"{name} checkpoint does not match this dataset split")

    baseline_selection = json.loads(Path("models/selection.json").read_text(encoding="utf-8"))
    if baseline_selection.get("checkpoint_dataset_digest") != split["dataset_digest"]:
        raise ValueError("Existing baseline settings do not match this dataset split")
    baseline_settings = {
        "cnn_foreground_threshold": float(baseline_selection["cnn_foreground_threshold"]),
        "hybrid_foreground_threshold": float(baseline_selection["hybrid_foreground_threshold"]),
        "hybrid_cnn_weight": float(baseline_selection["hybrid_cnn_weight"]),
        "hybrid_classical_weight": float(baseline_selection["hybrid_classical_weight"]),
        "selected_on": baseline_selection["selected_on"],
    }

    validation = cache_predictions(new_model, root, split["val"])
    print(f"Cached new-checkpoint predictions for {len(validation)} validation images.", flush=True)
    thresholds = [round(step / 100, 2) for step in range(10, 96, 5)]
    threshold_sweep = []
    for threshold in thresholds:
        threshold_sweep.append({"threshold": threshold, "scores": class_scores(validation, threshold, 1.0)})
    best_threshold = max(threshold_sweep, key=lambda row: (row["scores"]["mean_label_dice"], row["threshold"]))
    selected_threshold = best_threshold["threshold"]

    weights = [round(step / 100, 2) for step in range(0, 101, 5)]
    weight_sweep = []
    for weight in weights:
        weight_sweep.append({"weight": weight, "scores": class_scores(validation, selected_threshold, weight)})
    best_weight = max(weight_sweep, key=lambda row: (row["scores"]["mean_label_dice"], -row["weight"]))
    selected_weight = best_weight["weight"]

    new_settings = {
        "selected_on": "validation only",
        "selection_metric": "mean pooled Dice for Labels 1, 2, and 3; background excluded",
        "validation_images": len(split["val"]),
        "checkpoint_epoch": new_meta.get("epoch", "unknown"),
        "checkpoint_dataset_digest": split["dataset_digest"],
        "classical_threshold": 0.65,
        "cnn_foreground_threshold": selected_threshold,
        "hybrid_foreground_threshold": selected_threshold,
        "hybrid_cnn_weight": selected_weight,
        "hybrid_classical_weight": round(1.0 - selected_weight, 2),
        "cnn_validation_scores": class_scores(validation, selected_threshold, 1.0),
        "hybrid_validation_scores": class_scores(validation, selected_threshold, selected_weight),
        "cnn_threshold_sweep": threshold_sweep,
        "hybrid_weight_sweep": weight_sweep,
    }
    selection_path = Path("models/best_perclass_selection.json")
    selection_path.write_text(json.dumps(new_settings, indent=2), encoding="utf-8")
    print("New-model validation settings recorded before test evaluation:")
    print(json.dumps({key: new_settings[key] for key in (
        "cnn_foreground_threshold", "hybrid_cnn_weight", "hybrid_classical_weight",
        "cnn_validation_scores", "hybrid_validation_scores",
    )}, indent=2))

    prior_report = json.loads(Path("results.json").read_text(encoding="utf-8"))
    prior_subset = prior_report.get("previous_test_subset", {})
    prior_was_inspected = bool(prior_subset.get("already_inspected"))

    # No test image is loaded until both models and new validation settings are fixed.
    classical_counts = PixelCounts()
    baseline_metrics = {method: PixelCounts() for method in ("cnn", "hybrid")}
    new_metrics = {method: PixelCounts() for method in ("cnn", "hybrid")}
    per_label = {
        model_name: {method: {f"label_{index}": PixelCounts() for index in (1, 2, 3)} for method in ("cnn", "hybrid")}
        for model_name in ("baseline", "new")
    }
    for image_index, (x, y) in enumerate(batch_iter(root, split["test"], 1), start=1):
        gray = (x[0, 0].numpy() * 255).astype(np.uint8)
        truth = y[0].numpy()
        classical_score, classical_mask = classical(gray)
        classical_counts.update(classical_mask > 0, truth > 0)
        for model_name, model, settings, target_metrics in (
            ("baseline", baseline, baseline_settings, baseline_metrics),
            ("new", new_model, new_settings, new_metrics),
        ):
            probabilities = predict(model, gray)
            for method, weight, threshold in (
                ("cnn", 1.0, settings["cnn_foreground_threshold"]),
                ("hybrid", settings["hybrid_cnn_weight"], settings["hybrid_foreground_threshold"]),
            ):
                prediction = combine(probabilities, classical_score, weight=weight, threshold=threshold)
                target_metrics[method].update(prediction > 0, truth > 0)
                for index in (1, 2, 3):
                    per_label[model_name][method][f"label_{index}"].update(prediction == index, truth == index)
        if image_index % 100 == 0:
            print(f"Baseline/new test comparison {image_index}/{len(split['test'])}", flush=True)

    def summarize(metrics):
        return {name: counts.results() for name, counts in metrics.items()}

    report = {
        "split": "official test",
        "images": len(split["test"]),
        "evaluation_scope": "All 840 official test images; settings were fixed using validation data only.",
        "test_set_history": {
            "previous_150_image_subset_already_inspected": prior_was_inspected,
            "note": "The first 150 test images were inspected during earlier development; all 840 results here include those images.",
        },
        "settings": {"saved_baseline": baseline_settings, "new_model": new_settings},
        "classical_binary": classical_counts.results(),
        "classical_per_label": "Not applicable: classical output is unclassified binary foreground.",
        "saved_baseline": {
            "checkpoint": str(baseline_path),
            "epoch": baseline_meta.get("epoch", "unknown"),
            "cnn_binary": baseline_metrics["cnn"].results(),
            "hybrid_binary": baseline_metrics["hybrid"].results(),
            "cnn_per_label": summarize(per_label["baseline"]["cnn"]),
            "hybrid_per_label": summarize(per_label["baseline"]["hybrid"]),
        },
        "new_model": {
            "checkpoint": str(new_path),
            "epoch": new_meta.get("epoch", "unknown"),
            "cnn_binary": new_metrics["cnn"].results(),
            "hybrid_binary": new_metrics["hybrid"].results(),
            "cnn_per_label": summarize(per_label["new"]["cnn"]),
            "hybrid_per_label": summarize(per_label["new"]["hybrid"]),
        },
    }
    out = Path("comparison_perclass_results.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Full test comparison saved:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
