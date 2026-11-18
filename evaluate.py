"""Evaluate three methods on held-out labelled steel defect images."""
import argparse
import json
from pathlib import Path

import numpy as np

from surfaceguard.dataset import batch_iter, split_records
from surfaceguard.metrics import PixelCounts
from surfaceguard.vision import classical, combine


def selection_for_checkpoint(checkpoint_path: str, selection_override: str = "") -> Path:
    """Resolve the validation-selected settings belonging to a checkpoint.

    The two published checkpoints use legacy filenames, while additional
    checkpoints conventionally use ``<checkpoint_stem>_selection.json``.
    Callers evaluating any other checkpoint must supply ``--selection`` if
    that companion file does not exist.
    """
    if selection_override:
        path = Path(selection_override)
    else:
        checkpoint = Path(checkpoint_path).resolve()
        project_models = Path("models").resolve()
        published = {
            (project_models / "best.pt").resolve(): project_models / "selection.json",
            (project_models / "best_perclass.pt").resolve(): project_models / "best_perclass_selection.json",
        }
        path = published.get(checkpoint, checkpoint.with_name(f"{checkpoint.stem}_selection.json"))
    if not path.is_file():
        raise FileNotFoundError(
            f"No selection configuration found for {checkpoint_path}. "
            "Pass --selection with the validation-selected configuration."
        )
    return path


def evaluate(args):
    root = Path(args.data)
    split = split_records(root)
    keys = split["test"][:args.limit] if args.limit else split["test"]
    output = {"split": "test", "images": len(keys), "classical_binary": PixelCounts()}
    model = None
    if args.checkpoint:
        from surfaceguard.network import load_checkpoint, predict
        model, checkpoint = load_checkpoint(args.checkpoint)
        if checkpoint["dataset_digest"] != split["dataset_digest"]:
            raise ValueError("Checkpoint was trained on another image list; test comparison would be invalid")
        selection_path = selection_for_checkpoint(args.checkpoint, args.selection)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("checkpoint_dataset_digest") != split["dataset_digest"]:
            raise ValueError("Selected settings were chosen for another dataset split")
        threshold = float(selection.get("cnn_foreground_threshold", checkpoint.get("threshold", 0.5)))
        fusion_weight = float(selection.get("hybrid_cnn_weight", checkpoint["fusion_weight"]))
        output.update({"cnn_binary": PixelCounts(), "hybrid_binary": PixelCounts(), "cnn_per_class": {f"label_{i}": PixelCounts() for i in (1, 2, 3)}, "hybrid_per_class": {f"label_{i}": PixelCounts() for i in (1, 2, 3)}})
        output["selected_settings"] = {
            "selection_file": str(selection_path),
            "cnn_foreground_threshold": threshold,
            "hybrid_foreground_threshold": threshold,
            "hybrid_cnn_weight": fusion_weight,
            "hybrid_classical_weight": round(1.0 - fusion_weight, 2),
        }
    for x, y in batch_iter(root, keys, 1):
        gray = (x[0, 0].numpy() * 255).astype(np.uint8)
        truth = y[0].numpy()
        classical_score, mask = classical(gray)
        output["classical_binary"].update(mask > 0, truth > 0)
        if model:
            probabilities = predict(model, gray)
            cnn = combine(probabilities, classical_score, weight=1.0, threshold=threshold)
            hybrid = combine(probabilities, classical_score, weight=fusion_weight, threshold=threshold)
            output["cnn_binary"].update(cnn > 0, truth > 0)
            output["hybrid_binary"].update(hybrid > 0, truth > 0)
            for category in (1, 2, 3):
                output["cnn_per_class"][f"label_{category}"].update(cnn == category, truth == category)
                output["hybrid_per_class"][f"label_{category}"].update(hybrid == category, truth == category)
    for key in ("classical_binary", "cnn_binary", "hybrid_binary"):
        if key in output:
            output[key] = output[key].results()
    for key in ("cnn_per_class", "hybrid_per_class"):
        if key in output:
            output[key] = {name: values.results() for name, values in output[key].items()}
    print(json.dumps(output, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--checkpoint", default="", help="Trained CNN checkpoint; omit for classical baseline")
    parser.add_argument("--selection", default="", help="Validation-selected settings for --checkpoint")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="results.json")
    evaluate(parser.parse_args())
