from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def pairs(root: Path):
    images, masks = root / "images", root / "masks"
    if not images.is_dir() or not masks.is_dir():
        raise ValueError("Expected data/images and data/masks. See README.md")
    imap = {}
    for path in images.rglob("*"):
        if path.suffix.lower() in IMAGE_EXT:
            key = path.relative_to(images).with_suffix("").as_posix()
            if key in imap:
                raise ValueError(f"Duplicate image stem: {key}")
            imap[key] = path
    mmap = {}
    for path in masks.rglob("*"):
        if path.suffix.lower() in IMAGE_EXT:
            if path.suffix.lower() in (".jpg", ".jpeg"):
                raise ValueError(f"Lossy JPEG mask: {path}")
            key = path.relative_to(masks).with_suffix("").as_posix()
            if key in mmap:
                raise ValueError(f"Duplicate mask stem: {key}")
            mmap[key] = path
    if not imap or set(imap) != set(mmap):
        missing_masks = sorted(set(imap) - set(mmap))[:5]
        missing_images = sorted(set(mmap) - set(imap))[:5]
        raise ValueError(f"Unpaired images/masks: {len(imap)} images, {len(mmap)} masks; missing masks {missing_masks}; missing images {missing_images}")
    return [(key, imap[key], mmap[key]) for key in sorted(imap)]


def read_pair(image_path: Path, mask_path: Path, size=(200, 200)):
    with Image.open(image_path) as im, Image.open(mask_path) as m:
        image = np.asarray(im.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.uint8)
        mask_raw = np.asarray(m)
        if mask_raw.ndim != 2:
            raise ValueError(f"RGB mask needs conversion to class IDs: {mask_path}")
        unique = set(np.unique(mask_raw).tolist())
        if not unique <= {0, 1, 2, 3}:
            raise ValueError(f"Mask values {sorted(unique)[:15]} in {mask_path}; expected IDs 0..3")
        mask = np.asarray(m.resize(size, Image.Resampling.NEAREST), dtype=np.uint8)
    return image, mask


def split_records(root: Path, seed=42):
    entries = pairs(root)
    current = root / "splits.json"
    keys = [key for key, _, _ in entries]
    digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    if current.exists():
        split = json.loads(current.read_text())
        if split.get("dataset_digest") != digest:
            raise ValueError("Image list changed since split; remove data/splits.json only after checking for leakage")
        return split
    rng = random.Random(seed)
    grouped = {"label_1": [], "label_2": [], "label_3": [], "other": []}
    for key, image, mask in entries:
        _, arr = read_pair(image, mask)
        counts = np.bincount(arr.ravel(), minlength=4)[1:]
        group = f"label_{int(counts.argmax()) + 1}" if counts.sum() else "other"
        grouped[group].append(key)
    records = {"train": [], "val": [], "test": []}
    for group in grouped.values():
        rng.shuffle(group)
        n = len(group)
        explicit_test = [k for k in group if k.lower().startswith(("test/", "testing/"))]
        remaining = [k for k in group if k not in explicit_test]
        explicit_train = [k for k in remaining if k.lower().startswith(("train/", "training/", "val/", "validation/"))]
        if explicit_test and len(explicit_train) == len(remaining):
            val_count = max(1, round(len(remaining) * 0.2)) if len(remaining) >= 5 else 0
            records["test"].extend(explicit_test)
            records["val"].extend(remaining[:val_count])
            records["train"].extend(remaining[val_count:])
        else:
            train_end, val_end = round(n * 0.7), round(n * 0.85)
            records["train"].extend(group[:train_end])
            records["val"].extend(group[train_end:val_end])
            records["test"].extend(group[val_end:])
    if any(not records[x] for x in records):
        raise ValueError("Need enough paired images for a train/validation/test split")
    records["dataset_digest"] = digest
    records["seed"] = seed
    current.write_text(json.dumps(records, indent=2))
    return records


def batch_iter(root, keys, batch_size, shuffle=False, seed=42):
    import torch
    mapping = {key: (img, mask) for key, img, mask in pairs(root)}
    keys = list(keys)
    if shuffle:
        random.Random(seed).shuffle(keys)
    for start in range(0, len(keys), batch_size):
        batch = [read_pair(*mapping[k]) for k in keys[start:start + batch_size]]
        x = torch.from_numpy(np.stack([x for x, _ in batch]).copy()).float().unsqueeze(1) / 255
        y = torch.from_numpy(np.stack([y for _, y in batch]).copy()).long()
        yield x, y
