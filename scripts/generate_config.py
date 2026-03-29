"""
Scans the split directories and produces config/dataset_config.json.
All paths are stored RELATIVE to the project root.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "datasets"
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def to_relative(path):
    return str(path.relative_to(PROJECT_ROOT))


def count_in_dir(d):
    if not d.exists():
        return 0
    return len(
        [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXT]
    )


def generate():
    config = {}

    train_dir = BASE / "train"
    if not train_dir.exists():
        print("ERROR: datasets/train/ not found. Run split_data.py first.")
        return

    for cls_dir in sorted(train_dir.iterdir()):
        if not cls_dir.is_dir():
            continue

        subclasses = []
        for sub_dir in sorted(cls_dir.iterdir()):
            if not sub_dir.is_dir():
                continue

            counts = {}
            for split in ["train", "val", "test"]:
                p = BASE / split / cls_dir.name / sub_dir.name
                counts[split] = count_in_dir(p)

            subclasses.append(
                {
                    "name": sub_dir.name,
                    "train_path": to_relative(
                        BASE / "train" / cls_dir.name / sub_dir.name
                    ),
                    "val_path": to_relative(BASE / "val" / cls_dir.name / sub_dir.name),
                    "test_path": to_relative(
                        BASE / "test" / cls_dir.name / sub_dir.name
                    ),
                    "num_train": counts["train"],
                    "num_val": counts["val"],
                    "num_test": counts["test"],
                    "num_total": sum(counts.values()),
                }
            )

        config[cls_dir.name] = {
            "num_subclasses": len(subclasses),
            "subclasses": subclasses,
        }

    out_path = PROJECT_ROOT / "config" / "dataset_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Saved config to {out_path}")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    generate()
