"""Phase 1 checks: augmentation consistency, split integrity, checkpoint safety.

Run from the project folder:  .venv\\Scripts\\python.exe tests\\test_phase1.py
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surfaceguard.augment import AugmentConfig, augment_pair, geometric  # noqa: E402
from surfaceguard.dataset import pairs, read_pair, split_records  # noqa: E402

DATA = ROOT / "data"
STEP, OFFSET = 60, 30  # encode class IDs 0..3 as intensities 30, 90, 150, 210


def encode(mask):
    return (mask.astype(np.uint8) * STEP + OFFSET).astype(np.uint8)


def decode(image):
    return np.clip(np.rint((image.astype(np.float32) - OFFSET) / STEP), 0, 3).astype(np.uint8)


def interior(mask):
    """Pixels whose 3x3 neighbourhood has a single class (bilinear blending can't change them)."""
    lo = cv2.erode(mask, np.ones((3, 3), np.uint8))
    hi = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return lo == hi


def real_samples(n=25):
    split = split_records(DATA)
    mapping = {k: (i, m) for k, i, m in pairs(DATA)}
    keys = split["train"][:: max(1, len(split["train"]) // n)][:n]
    return [read_pair(*mapping[k]) for k in keys]


def test_flips_exact():
    cfg = AugmentConfig(rotate_prob=0.0)
    for _, mask in real_samples(10):
        for seed in range(8):
            img_out, mask_out = geometric(encode(mask), mask, np.random.default_rng(seed), cfg)
            assert np.array_equal(decode(img_out), mask_out), "flip desynchronised image and mask"
    cfg90 = AugmentConfig(rotate_prob=0.0, rot90=True)
    for _, mask in real_samples(5):
        for seed in range(8):
            img_out, mask_out = geometric(encode(mask), mask, np.random.default_rng(seed), cfg90)
            assert np.array_equal(decode(img_out), mask_out), "rot90 desynchronised image and mask"


def test_rotation_aligned():
    cfg = AugmentConfig(hflip_prob=0.0, vflip_prob=0.0, rotate_prob=1.0, max_angle=15.0)
    agree_all, agree_interior = [], []
    for _, mask in real_samples(25):
        if not mask.any():
            continue
        for seed in range(4):
            img_out, mask_out = geometric(encode(mask), mask, np.random.default_rng(seed), cfg)
            dec = decode(img_out)
            inner = interior(mask_out)
            agree_all.append(float((dec == mask_out).mean()))
            agree_interior.append(float((dec[inner] == mask_out[inner]).mean()))
    assert min(agree_interior) > 0.995, f"rotation misaligned inside regions: {min(agree_interior):.4f}"
    assert np.mean(agree_all) > 0.98, f"rotation misaligned overall: {np.mean(agree_all):.4f}"
    return np.mean(agree_all), min(agree_interior)


def test_photometric_leaves_mask():
    cfg = AugmentConfig()
    for image, mask in real_samples(10):
        for seed in range(5):
            _, m_full = augment_pair(image, mask, np.random.default_rng(seed), cfg)
            _, m_geo = geometric(image, mask, np.random.default_rng(seed), cfg)
            assert np.array_equal(m_full, m_geo), "photometric step changed the mask"


def test_output_contract():
    cfg = AugmentConfig()
    rng = np.random.default_rng(0)
    changed = 0
    for image, mask in real_samples(25):
        img_out, mask_out = augment_pair(image, mask, rng, cfg)
        assert img_out.dtype == np.uint8 and mask_out.dtype == np.uint8
        assert img_out.shape == image.shape == mask_out.shape
        assert set(np.unique(mask_out).tolist()) <= {0, 1, 2, 3}
        assert set(np.unique(mask_out).tolist()) <= set(np.unique(mask).tolist()), "new class introduced"
        changed += int(not np.array_equal(img_out, image))
    assert changed >= 20, "augmentation rarely changes the image"


def test_deterministic():
    image, mask = real_samples(1)[0]
    a = augment_pair(image, mask, np.random.default_rng(7))
    b = augment_pair(image, mask, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_split_integrity():
    split = split_records(DATA)
    tr, va, te = set(split["train"]), set(split["val"]), set(split["test"])
    assert (len(tr), len(va), len(te)) == (2904, 726, 840), (len(tr), len(va), len(te))
    assert not (tr & va) and not (tr & te) and not (va & te), "splits overlap"
    assert all(k.lower().startswith("test/") for k in te), "test split is not the official NEU-SEG test folder"


def test_overwrite_guard():
    proc = subprocess.run(
        [sys.executable, "train.py", "--output", "models/best_perclass.pt", "--log", "models/_guard.log",
         "--history", "models/_guard.json", "--epochs", "1", "--limit", "20"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode != 0 and "Refusing to overwrite" in (proc.stdout + proc.stderr), proc.stdout + proc.stderr
    assert not (ROOT / "models/_guard.log").exists()


def save_preview(path):
    """Write a visual grid: original | augmented image | augmented mask overlay."""
    colors = {1: (75, 100, 255), 2: (255, 170, 66), 3: (83, 205, 255)}
    rng = np.random.default_rng(3)
    rows = []
    for image, mask in [s for s in real_samples(25) if s[1].any()][:6]:
        img_out, mask_out = augment_pair(image, mask, rng)
        over = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR).astype(np.float32)
        for label, color in colors.items():
            over[mask_out == label] = 0.5 * over[mask_out == label] + 0.5 * np.array(color)
        orig = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rows.append(np.hstack([orig, cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR), over.astype(np.uint8)]))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))


if __name__ == "__main__":
    tests = [test_flips_exact, test_rotation_aligned, test_photometric_leaves_mask, test_output_contract,
             test_deterministic, test_split_integrity, test_overwrite_guard]
    failed = 0
    for test in tests:
        try:
            extra = test()
            print(f"PASS {test.__name__}" + (f"  {extra}" if extra else ""))
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
    save_preview(ROOT / "models/phase1/augmentation_preview.png")
    print("preview: models/phase1/augmentation_preview.png")
    sys.exit(1 if failed else 0)
