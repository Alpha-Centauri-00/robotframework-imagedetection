"""
Generates a tiny synthetic dataset so the smoke test suite in this folder can run
end-to-end without needing real photos yet. Not meant to prove the model is
accurate - only that training, saving, loading, and prediction all work, and
(most importantly for GPU testing) that CUDA is actually being used.

Run once before the smoke test:
    uv run python atest/generate_sample_data.py
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw

random.seed(0)

HERE = Path(__file__).parent
DATA = HERE / "Data"

CLASSES = {
    "Red": (200, 40, 40),
    "Blue": (40, 40, 200),
    "Green": (40, 180, 40),
}


def make_image(path: Path, base_color: tuple[int, int, int]) -> None:
    img = Image.new("RGB", (64, 64), base_color)
    draw = ImageDraw.Draw(img)
    # A bit of random noise/shapes so the model has to actually learn the
    # class, rather than a single memorized flat color.
    for _ in range(15):
        x0, y0 = random.randint(0, 63), random.randint(0, 63)
        x1, y1 = x0 + random.randint(1, 10), y0 + random.randint(1, 10)
        jitter = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in base_color)
        draw.rectangle([x0, y0, x1, y1], fill=jitter)
    img.save(path)


def main() -> None:
    for split, count in [("training", 10), ("validation", 4)]:
        for class_name, color in CLASSES.items():
            class_dir = DATA / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                make_image(class_dir / f"{class_name.lower()}_{i}.png", color)

    test_photos = DATA / "test_photos"
    test_photos.mkdir(parents=True, exist_ok=True)
    for class_name, color in CLASSES.items():
        make_image(test_photos / f"{class_name.lower()}_sample1.png", color)
        make_image(test_photos / f"{class_name.lower()}_sample2.png", color)

    print(f"Sample dataset written to {DATA}")


if __name__ == "__main__":
    main()
