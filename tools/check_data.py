import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surfaceguard.dataset import pairs, read_pair, split_records  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    data = Path(args.data)
    counts = Counter()
    items = pairs(data)
    for _, im_path, mask_path in items:
        _, mask = read_pair(im_path, mask_path)
        counts.update({int(c): int((mask == c).sum()) for c in (1, 2, 3)})
    print(f"Paired images: {len(items)}; pixel counts by mask ID: {dict(counts)}")
    if any(counts[x] == 0 for x in (1, 2, 3)):
        raise SystemExit("A mask ID is absent. Verify the mask labels and class mapping.")
    splits = split_records(data)
    print("Train/validation/test:", *(f"{x}={len(splits[x])}" for x in ("train", "val", "test")))
    print("Visually verify a sample of paired masks before training.")
