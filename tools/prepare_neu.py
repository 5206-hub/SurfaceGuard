"""Convert inspected NEU-SEG image/mask files into explicit class-ID pairs.

Run from the project root. Conversion is intentionally strict: unknown colors,
labels, unmatched files and ambiguous class names stop the import.
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surfaceguard.dataset import IMAGE_EXT  # noqa: E402

ALIASES = {"scratch": 1, "scratches": 1, "sc": 1, "patch": 2, "patches": 2, "pa": 2, "inclusion": 3, "inclusions": 3, "in": 3}
SPLITS = {"train": "train", "training": "train", "test": "test", "testing": "test", "val": "val", "validation": "val"}


def class_from_path(path):
    tokens = [part.lower().replace("-", "_").split("_") for part in path.parts]
    values = {ALIASES[token] for chunk in tokens for token in chunk if token in ALIASES}
    if len(values) > 1:
        raise ValueError(f"Ambiguous class in path {path}")
    return next(iter(values)) if values else None


def split_from_path(path):
    found = {SPLITS[x.lower()] for x in path.parts if x.lower() in SPLITS}
    return next(iter(found)) if len(found) == 1 else None


def parse_palette(spec):
    result = {}
    for entry in spec.split(",") if spec else []:
        color, label = entry.strip().split(":")
        label = ALIASES.get(label.lower())
        if label is None or len(color) != 7 or not color.startswith("#"):
            raise ValueError(f"Invalid palette entry {entry!r}; example #ff0000:scratch")
        result[tuple(bytes.fromhex(color[1:]))] = label
    return result


def convert_mask(path, fallback_class, palette):
    with Image.open(path) as opened:
        if opened.mode == "P" and not palette:
            raise ValueError(f"Paletted mask needs an inspected, verified --palette: {path}")
        values = np.asarray(opened.convert("RGB") if opened.mode in ("RGB", "RGBA", "P") and palette else opened)
    if values.ndim == 2:
        distinct = set(np.unique(values).tolist())
        if distinct <= {0, 1}:
            return values.astype(np.uint8)
        if distinct <= {0, 1, 2, 3} and distinct - {0, 1}:
            return values.astype(np.uint8)
        if distinct <= {0, 1, 255}:
            if fallback_class is None:
                raise ValueError(f"Binary mask has no identifiable defect class: {path}")
            return (values > 0).astype(np.uint8) * fallback_class
        raise ValueError(f"Unknown mask pixel IDs {sorted(distinct)[:15]} in {path}")
    if values.ndim == 3 and palette:
        result = np.zeros(values.shape[:2], dtype=np.uint8)
        known = set(palette) | {(0, 0, 0)}
        for color in np.unique(values[:, :, :3].reshape(-1, 3), axis=0):
            key = tuple(int(x) for x in color)
            if key not in known:
                raise ValueError(f"Unmapped color {key} in {path}; verify --palette")
            if key in palette:
                result[np.all(values[:, :, :3] == color, axis=2)] = palette[key]
        return result
    raise ValueError(f"Color mask requires a verified --palette: {path}")


def find_dir(root, names):
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    raise ValueError(f"Missing one of {names} in {root}; select a source containing images and annotations")


def key(path, directory):
    relative = path.relative_to(directory)
    split = split_from_path(relative) or "unspecified"
    category = class_from_path(relative)
    return split, category, path.stem.lower()


def main(args):
    src, dest = Path(args.source).resolve(), Path(args.output).resolve()
    if not src.is_dir() or src == dest or src in dest.parents:
        raise ValueError("Source must exist and output cannot be inside source")
    images_dir = find_dir(src, ["images", "Images"])
    masks_dir = find_dir(src, ["annotations", "masks", "Annotations", "Masks"])
    palette = parse_palette(args.palette)
    images = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT]
    masks = [p for p in masks_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT]
    if not images or not masks:
        raise ValueError("No images/masks found")
    lookup = {}
    for image in images:
        lookup.setdefault(key(image, images_dir), []).append(image)
    # Avoid silently overwriting a user's existing prepared dataset.
    if (dest / "images").exists() or (dest / "masks").exists():
        raise ValueError(f"Output {dest} already has images/masks; use a new directory")
    matched = []
    for mask in masks:
        split, category, stem = key(mask, masks_dir)
        candidate = lookup.get((split, category, stem), [])
        if not candidate:
            candidate = [img for (s, c, n), paths in lookup.items() if n == stem and (s == split or "unspecified" in (s, split)) and (c == category or None in (c, category)) for img in paths]
        if len(candidate) != 1:
            raise ValueError(f"Mask {mask} matches {len(candidate)} images. Ensure paired filenames and class/split folders.")
        matched.append((candidate[0], mask, split, category))
    if len({p[0] for p in matched}) != len(images) or len(matched) != len(images):
        raise ValueError(f"Expected one mask per image: {len(images)} images / {len(matched)} masks. Multi-file-per-image masks need manual combination.")
    audit = {"source": str(src), "mapping": {"0": "background", "1": "mask label 1", "2": "mask label 2", "3": "mask label 3"}, "images": len(images), "class_pixel_counts": Counter(), "splits": Counter()}
    converted = []
    for image, mask, split, category in matched:
        if mask.suffix.lower() in (".jpg", ".jpeg"):
            raise ValueError(f"JPEG ground truth mask loses pixel IDs: {mask}")
        data = convert_mask(mask, category or class_from_path(image.relative_to(images_dir)), palette)
        with Image.open(image) as opened:
            if opened.size != (data.shape[1], data.shape[0]):
                raise ValueError(f"Image/mask dimensions differ: {image} / {mask}")
        audit["class_pixel_counts"].update({str(c): int((data == c).sum()) for c in (1, 2, 3)})
        audit["splits"][split] += 1
        converted.append((image, data, split))
    for index, (image, data, split) in enumerate(converted):
        relative = Path(split) / f"sample_{index:05d}"
        im_path = dest / "images" / relative.with_suffix(image.suffix.lower())
        mask_path = dest / "masks" / relative.with_suffix(".png")
        im_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, im_path)
        Image.fromarray(data).save(mask_path)
    audit["class_pixel_counts"] = dict(audit["class_pixel_counts"])
    audit["splits"] = dict(audit["splits"])
    (dest / "mask_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print("Verify several original image/mask pairs visually before training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="data")
    parser.add_argument("--palette", default="", help="Verified RGB colors: '#ff0000:scratch,#0000ff:patch,#00ff00:inclusion'")
    main(parser.parse_args())
