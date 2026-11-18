"""Phase 1 experiments: augmentation, cosine LR + early stopping, Label 1 class weight.

  python phase1_experiments.py train      # launch all runs in parallel (writes models/phase1/)
  python phase1_experiments.py select     # validation-only threshold/weight sweep per run + model choice
  python phase1_experiments.py test       # score the locked choice and the current model on the test split

Existing checkpoints in models/ are never written.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

OUT = Path("models/phase1")
COMMON = ["--epochs", "30", "--scheduler", "cosine", "--patience", "7", "--threads", "2"]
RUNS = {
    "noaug_w14": [],
    "aug_w14": ["--augment"],
    "aug_w18": ["--augment", "--l1-weight", "1.8"],
    "aug_w22": ["--augment", "--l1-weight", "2.2"],
    "aug_w25": ["--augment", "--l1-weight", "2.5"],
}
BASELINE = ("current_best_perclass", Path("models/best_perclass.pt"), Path("models/best_perclass_selection.json"))
THRESHOLDS = [round(step / 100, 2) for step in range(10, 96, 5)]
WEIGHTS = [round(step / 100, 2) for step in range(0, 101, 5)]


def train_all():
    OUT.mkdir(parents=True, exist_ok=True)
    procs = []
    for name, extra in RUNS.items():
        cmd = [sys.executable, "-u", "train.py", *COMMON, *extra,
               "--output", str(OUT / f"{name}.pt"), "--log", str(OUT / f"{name}.log"),
               "--history", str(OUT / f"{name}.history.json")]
        stdout = (OUT / f"{name}.stdout.txt").open("w", encoding="utf-8")
        procs.append((name, subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT)))
        print("started", name, " ".join(cmd), flush=True)
    for name, proc in procs:
        print(name, "exit code", proc.wait(), flush=True)


def cache(model, root, keys):
    from surfaceguard.dataset import batch_iter
    from surfaceguard.network import predict
    from surfaceguard.vision import classical
    records = []
    for x, y in batch_iter(root, keys, 1):
        gray = (x[0, 0].numpy() * 255).astype(np.uint8)
        records.append((predict(model, gray), classical(gray)[0], y[0].numpy()))
    return records


def score(records, threshold, weight):
    from surfaceguard.metrics import PixelCounts
    from surfaceguard.vision import combine
    binary = PixelCounts()
    labels = {f"label_{i}": PixelCounts() for i in (1, 2, 3)}
    for probs, cscore, truth in records:
        pred = combine(probs, cscore, weight=weight, threshold=threshold)
        binary.update(pred > 0, truth > 0)
        for i in (1, 2, 3):
            labels[f"label_{i}"].update(pred == i, truth == i)
    per_label = {k: v.results() for k, v in labels.items()}
    mean = float(np.mean([per_label[f"label_{i}"]["dice"] or 0.0 for i in (1, 2, 3)]))
    return {"mean_label_dice": round(mean, 4), "binary": binary.results(), "per_label": per_label}


def select_settings(records):
    """Same rule as compare_models.py: threshold first (CNN), then hybrid weight; validation only."""
    t_sweep = [{"threshold": t, "scores": score(records, t, 1.0)} for t in THRESHOLDS]
    t_best = max(t_sweep, key=lambda r: (r["scores"]["mean_label_dice"], r["threshold"]))["threshold"]
    w_sweep = [{"weight": w, "scores": score(records, t_best, w)} for w in WEIGHTS]
    w_best = max(w_sweep, key=lambda r: (r["scores"]["mean_label_dice"], -r["weight"]))["weight"]
    return t_best, w_best, t_sweep, w_sweep


def select():
    from surfaceguard.dataset import split_records
    from surfaceguard.network import load_checkpoint
    root = Path("data")
    split = split_records(root)
    summary = {}
    candidates = [(n, OUT / f"{n}.pt") for n in RUNS if (OUT / f"{n}.pt").is_file()]
    candidates.append(BASELINE[:2])
    for name, ckpt in candidates:
        model, meta = load_checkpoint(ckpt)
        if meta["dataset_digest"] != split["dataset_digest"]:
            raise ValueError(f"{name}: checkpoint trained on another image list")
        records = cache(model, root, split["val"])
        t, w, t_sweep, w_sweep = select_settings(records)
        cnn, hybrid = score(records, t, 1.0), score(records, t, w)
        selection = {
            "selected_on": "validation only",
            "selection_metric": "mean pooled Dice for Labels 1, 2, and 3; background excluded",
            "validation_images": len(split["val"]),
            "checkpoint": str(ckpt), "checkpoint_epoch": meta.get("epoch", "unknown"),
            "checkpoint_dataset_digest": split["dataset_digest"],
            "classical_threshold": 0.65,
            "cnn_foreground_threshold": t, "hybrid_foreground_threshold": t,
            "hybrid_cnn_weight": w, "hybrid_classical_weight": round(1.0 - w, 2),
            "cnn_validation_scores": cnn, "hybrid_validation_scores": hybrid,
            "cnn_threshold_sweep": t_sweep, "hybrid_weight_sweep": w_sweep,
        }
        if name != BASELINE[0]:
            (OUT / f"{name}_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        summary[name] = {k: selection[k] for k in ("checkpoint", "checkpoint_epoch", "cnn_foreground_threshold",
                                                   "hybrid_cnn_weight", "cnn_validation_scores",
                                                   "hybrid_validation_scores")}
        print(f"{name}: epoch={meta.get('epoch')} thr={t} w={w} val_cnn_mean={cnn['mean_label_dice']} "
              f"val_hybrid_mean={hybrid['mean_label_dice']} "
              f"L1={hybrid['per_label']['label_1']['dice']} L2={hybrid['per_label']['label_2']['dice']} "
              f"L3={hybrid['per_label']['label_3']['dice']}", flush=True)
    new_runs = [n for n in summary if n != BASELINE[0]]
    chosen = max(new_runs, key=lambda n: summary[n]["hybrid_validation_scores"]["mean_label_dice"])
    report = {
        "rule": "highest validation mean per-label hybrid Dice after validation-only settings selection",
        "chosen_run": chosen, "chosen_checkpoint": str(OUT / f"{chosen}.pt"),
        "validation_summary": summary,
        "test_split_used": False,
    }
    (OUT / "phase1_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("chosen on validation:", chosen)


def test():
    from surfaceguard.dataset import split_records
    from surfaceguard.network import load_checkpoint
    root = Path("data")
    split = split_records(root)
    report = json.loads((OUT / "phase1_validation_report.json").read_text(encoding="utf-8"))
    chosen = report["chosen_run"]
    entries = [
        (chosen, OUT / f"{chosen}.pt", OUT / f"{chosen}_selection.json"),
        BASELINE,
    ]
    results = {}
    for name, ckpt, sel_path in entries:
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        model, meta = load_checkpoint(ckpt)
        records = cache(model, root, split["test"])
        t, w = float(sel["cnn_foreground_threshold"]), float(sel["hybrid_cnn_weight"])
        results[name] = {
            "checkpoint": str(ckpt), "epoch": meta.get("epoch"), "settings_from": str(sel_path),
            "threshold": t, "hybrid_cnn_weight": w,
            "cnn": score(records, t, 1.0), "hybrid": score(records, t, w),
        }
        print(name, json.dumps({m: results[name][m]["mean_label_dice"] for m in ("cnn", "hybrid")}), flush=True)
    out = {
        "split": "official test", "images": len(split["test"]),
        "test_set_history": ("Not a fresh estimate: the first 150 test images were inspected in early development "
                             "and all 840 were scored for the earlier baseline comparison. The Phase 1 model and its "
                             "settings were fixed on validation before this run; only the single chosen run was scored."),
        "results": results,
    }
    (OUT / "phase1_test_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    {"train": train_all, "select": select, "test": test}[sys.argv[1]]()
