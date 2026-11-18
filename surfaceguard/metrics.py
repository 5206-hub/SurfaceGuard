import numpy as np


def binary_metrics(prediction, truth):
    p, t = prediction > 0, truth > 0
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    return {
        "dice": (2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 1.0,
        "iou": (tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
        "precision": (tp / (tp + fp)) if tp + fp else (1.0 if fn == 0 else 0.0),
        "recall": (tp / (tp + fn)) if tp + fn else 1.0,
    }


def per_class(prediction, truth):
    return {f"label_{idx}": binary_metrics(prediction == idx, truth == idx) for idx in (1, 2, 3)}


def aggregate(rows):
    if not rows:
        raise ValueError("No evaluation images")
    return {key: round(float(np.mean([r[key] for r in rows])), 4) for key in rows[0]}


class PixelCounts:
    """Accumulate pixel confusion across images without rewarding absent classes."""

    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def update(self, prediction, truth):
        p, t = prediction.astype(bool), truth.astype(bool)
        self.tp += int((p & t).sum())
        self.fp += int((p & ~t).sum())
        self.fn += int((~p & t).sum())

    def results(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        def safe(value, denom):
            return round(value / denom, 4) if denom else None
        return {
            "dice": safe(2 * tp, 2 * tp + fp + fn),
            "iou": safe(tp, tp + fp + fn),
            "precision": safe(tp, tp + fp),
            "recall": safe(tp, tp + fn),
            "ground_truth_pixels": tp + fn,
            "predicted_pixels": tp + fp,
        }
