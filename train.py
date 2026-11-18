"""Train a compact four-class steel defect segmentation model."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from surfaceguard.augment import AugmentConfig, augment_pair
from surfaceguard.dataset import pairs, read_pair, split_records
from surfaceguard.metrics import PixelCounts
from surfaceguard.network import TinyUNet
from surfaceguard.vision import combine


def load_arrays(root, keys):
    mapping = {key: (img, mask) for key, img, mask in pairs(root)}
    loaded = [read_pair(*mapping[k]) for k in keys]
    return np.stack([x for x, _ in loaded]), np.stack([y for _, y in loaded])


def to_tensors(images, masks, device):
    x = torch.from_numpy(np.ascontiguousarray(images)).float().unsqueeze(1).div_(255).to(device)
    y = torch.from_numpy(np.ascontiguousarray(masks)).long().to(device)
    return x, y


@torch.inference_mode()
def validate(model, images, masks, batch_size, device):
    """Pooled per-label scores at threshold 0.5, CNN only (same rule as the original loop)."""
    model.eval()
    class_counts = {f"label_{index}": PixelCounts() for index in (1, 2, 3)}
    binary_counts = PixelCounts()
    for start in range(0, len(images), batch_size):
        x, _ = to_tensors(images[start:start + batch_size], masks[start:start + batch_size], device)
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
        for p, target in zip(probs, masks[start:start + batch_size]):
            # weight=1.0 ignores the classical score, so zeros are equivalent.
            prediction = combine(p, np.zeros(target.shape, np.float32), weight=1.0, threshold=0.5)
            binary_counts.update(prediction > 0, target > 0)
            for category in (1, 2, 3):
                class_counts[f"label_{category}"].update(prediction == category, target == category)
    per_label = {label: counts.results() for label, counts in class_counts.items()}
    return per_label, binary_counts.results()


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.data)
    split = split_records(root, args.seed)
    training = split["train"][:args.limit] if args.limit else split["train"]
    validation = split["val"][:max(15, min(150, args.limit // 5))] if args.limit else split["val"]
    if set(training) & set(split["test"]) or set(validation) & set(split["test"]):
        raise ValueError("Test keys found in training/validation selection")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads or min(4, torch.get_num_threads()))

    out = Path(args.output)
    log_path = Path(args.log)
    history_path = Path(args.history) if args.history else out.with_suffix(".history.json")
    for path in (out, log_path, history_path):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file {path}; choose a new path or pass --overwrite")
    for path in (out, log_path, history_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    train_x, train_y = load_arrays(root, training)
    val_x, val_y = load_arrays(root, validation)

    model = TinyUNet(width=args.width).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.min_lr)
                 if args.scheduler == "cosine" else None)
    class_weights = [args.bg_weight, args.l1_weight, args.l2_weight, args.l3_weight]
    weights = torch.tensor(class_weights, device=device)
    aug_cfg = AugmentConfig(rot90=args.rot90) if args.augment else None
    aug_rng = np.random.default_rng(args.seed)

    config = {
        "epochs": args.epochs, "batch_size": args.batch_size, "width": args.width, "lr": args.lr,
        "min_lr": args.min_lr, "scheduler": args.scheduler, "patience": args.patience,
        "class_weights": class_weights, "augmentation": aug_cfg.as_dict() if aug_cfg else None,
        "seed": args.seed, "training_images": len(training), "validation_images": len(validation),
    }
    log(f"device={device} train={len(training)} val={len(validation)} test(untouched)={len(split['test'])}")
    log(f"config={json.dumps(config)}")
    log("selection_metric=mean per-label Dice for labels 1,2,3; validation_threshold=0.5")

    history = {"config": config, "dataset_digest": split["dataset_digest"], "epochs": []}
    best, best_epoch, stale = -1.0, None, 0
    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        lr_now = optim.param_groups[0]["lr"]
        order = list(range(len(training)))
        random.Random(args.seed + epoch).shuffle(order)
        totals = {"loss": 0.0, "ce": 0.0, "dice": 0.0}
        seen = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            images, masks = train_x[idx], train_y[idx]
            if aug_cfg:
                pairs_aug = [augment_pair(i, m, aug_rng, aug_cfg) for i, m in zip(images, masks)]
                images = np.stack([i for i, _ in pairs_aug])
                masks = np.stack([m for _, m in pairs_aug])
            x, y = to_tensors(images, masks, device)
            logits = model(x)
            ce = F.cross_entropy(logits, y, weight=weights)
            probs = logits.softmax(dim=1)
            onehot = F.one_hot(y, 4).permute(0, 3, 1, 2)
            overlap = (probs[:, 1:] * onehot[:, 1:]).sum(dim=(0, 2, 3))
            dice = 1 - ((2 * overlap + 1) / (probs[:, 1:].sum(dim=(0, 2, 3)) + onehot[:, 1:].sum(dim=(0, 2, 3)) + 1)).mean()
            loss = ce + dice
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            n = len(idx)
            totals["loss"] += float(loss.item()) * n
            totals["ce"] += float(ce.item()) * n
            totals["dice"] += float(dice.item()) * n
            seen += n
        if scheduler:
            scheduler.step()

        per_label, binary = validate(model, val_x, val_y, args.batch_size, device)
        label_dice = {label: float(scores["dice"] or 0.0) for label, scores in per_label.items()}
        mean_dice = float(np.mean(list(label_dice.values())))
        elapsed = time.time() - started
        record = {
            "epoch": epoch + 1, "lr": lr_now, "seconds": round(elapsed, 1),
            "train_loss": round(totals["loss"] / seen, 5), "train_ce": round(totals["ce"] / seen, 5),
            "train_dice_loss": round(totals["dice"] / seen, 5),
            "val_mean_label_dice": round(mean_dice, 4), "val_binary": binary, "val_per_label": per_label,
        }
        history["epochs"].append(record)
        log(
            f"epoch={epoch + 1} completed loss={record['train_loss']:.4f} lr={lr_now:.6f} "
            f"val_label_1_dice={label_dice['label_1']:.4f} "
            f"val_label_2_dice={label_dice['label_2']:.4f} "
            f"val_label_3_dice={label_dice['label_3']:.4f} "
            f"val_mean_label_dice={mean_dice:.4f} val_binary_dice={binary['dice']:.4f} time={elapsed:.0f}s"
        )
        if mean_dice > best:
            best, best_epoch, stale = mean_dice, epoch + 1, 0
            torch.save({
                "weights": model.state_dict(), "width": args.width,
                "class_names": ["background", "label_1", "label_2", "label_3"],
                "fusion_weight": 1.0, "threshold": 0.5,
                "epoch": best_epoch, "epochs_requested": args.epochs,
                "best_checkpoint_metric": "mean_validation_label_dice_excluding_background",
                "validation_per_class_dice": label_dice,
                "validation_per_class_scores": per_label,
                "validation_mean_label_dice": mean_dice,
                "validation_binary_dice": binary["dice"],
                "seed": args.seed, "training_images": len(training), "validation_images": len(validation),
                "dataset_digest": split["dataset_digest"],
                "training_config": json.dumps(config),
            }, out)
            log(f"checkpoint_selected epoch={best_epoch} mean_validation_label_dice={best:.4f} path={out}")
        else:
            stale += 1
            log(f"checkpoint_kept epoch={best_epoch} best_mean_validation_label_dice={best:.4f} stale={stale}")
        history.update({"best_epoch": best_epoch, "best_val_mean_label_dice": round(best, 4)})
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if args.patience and stale >= args.patience:
            log(f"early_stop epoch={epoch + 1} no improvement for {stale} epochs")
            break
    history["stopped_epoch"] = len(history["epochs"])
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    log(f"training_complete epochs_run={len(history['epochs'])} best_epoch={best_epoch} best_mean_validation_label_dice={best:.4f} checkpoint={out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="models/best.pt")
    parser.add_argument("--log", default="models/train.log")
    parser.add_argument("--history", default="", help="Per-epoch JSON metrics (default: <output>.history.json)")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output/log/history files")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--patience", type=int, default=0, help="Early-stopping patience in epochs (0 = off)")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--rot90", action="store_true", help="Also apply random 90-degree turns when augmenting")
    parser.add_argument("--bg-weight", type=float, default=0.35)
    parser.add_argument("--l1-weight", type=float, default=1.4)
    parser.add_argument("--l2-weight", type=float, default=1.4)
    parser.add_argument("--l3-weight", type=float, default=1.4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Limit training and validation images for a quick run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.width <= 0 or (args.limit and args.limit < 20):
        parser.error("Epochs, batch size, and width must be positive; limit must be at least 20")
    train(args)
