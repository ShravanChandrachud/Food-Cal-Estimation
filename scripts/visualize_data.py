"""
Prints a clean table of image counts per class/subclass/split.
Screenshot this output for your submission.
"""

from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE = PROJECT_ROOT / "datasets"
SPLITS = ["train", "val", "test"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images():
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for split in SPLITS:
        split_dir = BASE / split
        if not split_dir.exists():
            print(f"WARNING: {split_dir} does not exist. Run split_data.py first.")
            continue

        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for sub_dir in sorted(cls_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue
                n = len(
                    [
                        f
                        for f in sub_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in VALID_EXT
                    ]
                )
                stats[cls_dir.name][sub_dir.name][split] = n

    if not stats:
        print(
            "No data found. Make sure you ran organize_data.py and split_data.py first."
        )
        return

    # Print table
    header = f"{'Class':<15} {'Subclass':<15} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}"
    print(header)
    print("-" * len(header))

    grand = {"train": 0, "val": 0, "test": 0}

    for cls in sorted(stats):
        for sub in sorted(stats[cls]):
            tr = stats[cls][sub].get("train", 0)
            va = stats[cls][sub].get("val", 0)
            te = stats[cls][sub].get("test", 0)
            total = tr + va + te
            grand["train"] += tr
            grand["val"] += va
            grand["test"] += te
            print(f"{cls:<15} {sub:<15} {tr:>7} {va:>7} {te:>7} {total:>7}")

    gt = grand["train"] + grand["val"] + grand["test"]
    print("-" * len(header))
    print(
        f"{'TOTAL':<31} {grand['train']:>7} {grand['val']:>7} {grand['test']:>7} {gt:>7}"
    )


if __name__ == "__main__":
    count_images()
