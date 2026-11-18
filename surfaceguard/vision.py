from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

NAMES = {0: "Background", 1: "Label 1", 2: "Label 2", 3: "Label 3"}
COLORS = {1: (255, 100, 75), 2: (66, 170, 255), 3: (255, 205, 83)}


def decode_image(blob: bytes) -> np.ndarray:
    with Image.open(BytesIO(blob)) as opened:
        img = opened.convert("RGB")
        if img.width * img.height > 15_000_000:
            raise ValueError("Image is too large (maximum 15 megapixels).")
        return np.asarray(img)


def prepare_gray(rgb: np.ndarray, size: tuple[int, int] = (200, 200)) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def classical(gray: np.ndarray, threshold: float = 0.65, kernel: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Returns a [0,1] heuristic likelihood and a cleaned binary mask."""
    if gray.ndim != 2:
        raise ValueError("Expected an 8-bit grayscale image")
    gray = gray.astype(np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)
    kernel = max(3, int(kernel) | 1)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    bright = cv2.morphologyEx(smoothed, cv2.MORPH_TOPHAT, element)
    dark = cv2.morphologyEx(smoothed, cv2.MORPH_BLACKHAT, element)
    response = np.maximum(bright, dark).astype(np.float32)
    scale = max(float(np.percentile(response, 99.5)), 12.0)
    score = np.clip(response / scale, 0, 1)
    binary = (score >= threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    binary = remove_small_regions(binary, 5)
    return score.astype(np.float32), binary.astype(np.uint8)


def remove_small_regions(mask: np.ndarray, minimum: int = 5) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    result = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum:
            result[labels == label] = 1
    return result


def combine(probabilities: np.ndarray, classical_score: np.ndarray, weight: float = 0.8, threshold: float = 0.5) -> np.ndarray:
    """Fuse foreground likelihood; model determines class. Background is label 0."""
    if probabilities.shape[0] != 4 or probabilities.shape[1:] != classical_score.shape:
        raise ValueError("Expected probabilities shaped [4,H,W] and score [H,W]")
    defect_probability = 1.0 - probabilities[0]
    fused_probability = weight * defect_probability + (1.0 - weight) * classical_score
    labels = probabilities[1:].argmax(axis=0).astype(np.uint8) + 1
    labels[fused_probability < threshold] = 0
    for category in (1, 2, 3):
        region = remove_small_regions((labels == category).astype(np.uint8), minimum=4)
        labels[(labels == category) & (region == 0)] = 0
    return labels


def overlay(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scaled = cv2.resize(rgb, (labels.shape[1], labels.shape[0]), interpolation=cv2.INTER_AREA)
    canvas = scaled.astype(np.float32)
    for label, color in COLORS.items():
        pixels = labels == label
        canvas[pixels] = 0.55 * canvas[pixels] + 0.45 * np.asarray(color)
    return canvas.clip(0, 255).astype(np.uint8)


def summary(labels: np.ndarray, mode: str) -> dict:
    total = labels.size
    regions = []
    for category in (1, 2, 3):
        binary = (labels == category).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        for n in range(1, count):
            x, y, w, h, pixels = [int(v) for v in stats[n]]
            regions.append({"type": NAMES[category] if mode != "classical" else "Unclassified", "x": x, "y": y, "width_px": w, "height_px": h, "area_px": pixels})
    pixel_count = int(np.count_nonzero(labels))
    return {
        "method": mode,
        "measurement": "fraction of inspected image pixels; not physical severity",
        "affected_area_percent": round(100.0 * pixel_count / total, 2),
        "affected_pixels": pixel_count,
        "inspected_pixels": int(total),
        "regions": regions,
    }
