"""Training-time augmentation for grayscale image / class-ID mask pairs.

Geometric transforms are applied identically to image and mask (mask uses
nearest-neighbour so class IDs are never blended). Photometric transforms
touch the image only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    # 90-degree turns change the rolling-direction statistics of strip images, so off by default.
    rot90: bool = False
    rotate_prob: float = 0.5
    max_angle: float = 15.0
    brightness: float = 0.10
    contrast: float = 0.15
    noise_prob: float = 0.3
    noise_std: float = 4.0

    def as_dict(self) -> dict:
        return asdict(self)


def geometric(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig):
    if image.shape != mask.shape or image.ndim != 2:
        raise ValueError("Expected 2-D image and mask with identical shapes")
    if rng.random() < cfg.hflip_prob:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < cfg.vflip_prob:
        image, mask = image[::-1, :], mask[::-1, :]
    if cfg.rot90:
        k = int(rng.integers(4))
        image, mask = np.rot90(image, k), np.rot90(mask, k)
    if cfg.max_angle > 0 and rng.random() < cfg.rotate_prob:
        angle = float(rng.uniform(-cfg.max_angle, cfg.max_angle))
        h, w = image.shape
        matrix = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1.0)
        image = cv2.warpAffine(np.ascontiguousarray(image), matrix, (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(np.ascontiguousarray(mask), matrix, (w, h),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def photometric(image: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    out = image.astype(np.float32)
    gain = 1.0 + float(rng.uniform(-cfg.contrast, cfg.contrast))
    bias = float(rng.uniform(-cfg.brightness, cfg.brightness)) * 255.0
    mean = float(out.mean())
    out = (out - mean) * gain + mean + bias
    if cfg.noise_std > 0 and rng.random() < cfg.noise_prob:
        out += rng.normal(0.0, cfg.noise_std, out.shape).astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def augment_pair(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig = AugmentConfig()):
    image, mask = geometric(image, mask, rng, cfg)
    return photometric(image, rng, cfg), mask
