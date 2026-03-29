"""
Takes raw_organized/ and splits into train/val/test (70/15/15).
Creates datasets/train/, datasets/val/, datasets/test/.
"""

import os
import shutil
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORGANIZED_DIR = PROJECT_ROOT / "datasets" / "raw_organized"
OUTPUT_DIR = PROJECT_ROOT / "datasets"
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_dataset():
    random.seed(SEED)
    grand_total = {"train": 0, "val": 0, "test": 0}

    for cls_dir in sorted(ORGANIZED_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        print(f"\n[{cls_name.upper()}]")

        for sub_dir in sorted(cls_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            sub_name = sub_dir.name

            # Gather all image files
            imgs = [
                f
                for f in sub_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ]
            random.shuffle(imgs)

            n = len(imgs)
            if n == 0:
                print(f"  {sub_name}: EMPTY — skipping")
                continue

            n_train = int(n * SPLITS["train"])
            n_val = int(n * SPLITS["val"])
            # Rest goes to test (handles rounding)

            assignments = {
                "train": imgs[:n_train],
                "val": imgs[n_train : n_train + n_val],
                "test": imgs[n_train + n_val :],
            }

            for split_name, split_imgs in assignments.items():
                dst = OUTPUT_DIR / split_name / cls_name / sub_name
                dst.mkdir(parents=True, exist_ok=True)
                for img in split_imgs:
                    shutil.copy2(img, dst / img.name)
                grand_total[split_name] += len(split_imgs)

            print(
                f"  {sub_name}: "
                f"{len(assignments['train'])} train / "
                f"{len(assignments['val'])} val / "
                f"{len(assignments['test'])} test "
                f"(total: {n})"
            )

    print(f"\n{'=' * 50}")
    print(
        f"GRAND TOTAL: "
        f"{grand_total['train']} train / "
        f"{grand_total['val']} val / "
        f"{grand_total['test']} test"
    )


if __name__ == "__main__":
    # Clean existing splits
    for split in ["train", "val", "test"]:
        p = OUTPUT_DIR / split
        if p.exists():
            print(f"Removing existing {p}...")
            shutil.rmtree(p)

    print("Splitting raw_organized/ -> train/ val/ test/")
    split_dataset()
