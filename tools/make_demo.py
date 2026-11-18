"""Create visibly synthetic example images for interface smoke testing."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

destination = Path("demo")
destination.mkdir(exist_ok=True)
rng = np.random.default_rng(4)
for kind in ("scratch", "patch", "inclusion"):
    base = np.clip(rng.normal(135, 12, (200, 200)), 0, 255).astype(np.uint8)
    image = Image.fromarray(base).convert("RGB")
    brush = ImageDraw.Draw(image)
    if kind == "scratch":
        brush.line([(30, 151), (72, 119), (110, 93), (172, 28)], fill=(22, 22, 22), width=4)
    elif kind == "patch":
        brush.ellipse((54, 48, 148, 130), fill=(61, 61, 61))
    else:
        for x, y in [(36, 72), (104, 45), (135, 151), (87, 132)]:
            brush.ellipse((x, y, x + 9, y + 10), fill=(31, 31, 31))
    image.save(destination / f"demo_{kind}.png")
print("Synthetic samples saved to demo/; they are not model evaluation data.")
