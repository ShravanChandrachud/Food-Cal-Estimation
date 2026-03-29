"""
Copies only the subclasses we need from raw downloads into a flat
datasets/raw_organized/ structure. Run BEFORE split_data.py.
"""

import os
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION — All paths resolved relative to project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
ORGANIZED_DIR = PROJECT_ROOT / "datasets" / "raw_organized"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FRUITS_BASE = "fruits/fruits-360_100x100/fruits-360/Training"

CLASS_MAP = {
    "fruits": {
        "apple": [
            f"{FRUITS_BASE}/Apple 5",
            f"{FRUITS_BASE}/Apple 6",
            f"{FRUITS_BASE}/Apple 7",
            f"{FRUITS_BASE}/Apple 8",
            f"{FRUITS_BASE}/Apple Braeburn 1",
            f"{FRUITS_BASE}/Apple Golden 1",
            f"{FRUITS_BASE}/Apple Red 1",
        ],
        "banana": [
            f"{FRUITS_BASE}/Banana 1",
            f"{FRUITS_BASE}/Banana 3",
            f"{FRUITS_BASE}/Banana 4",
        ],
        "lemon": f"{FRUITS_BASE}/Lemon 1",
        "strawberry": [
            f"{FRUITS_BASE}/Strawberry 1",
            f"{FRUITS_BASE}/Strawberry 2",
            f"{FRUITS_BASE}/Strawberry 3",
        ],
        "orange": [
            f"{FRUITS_BASE}/Orange 1",
            f"{FRUITS_BASE}/Orange 2",
            f"{FRUITS_BASE}/Orange 3",
        ],
    },
    "vegetables": {
        "onion": "vegetables/veg200_images/onion",
        "tomato": "vegetables/veg200_images/tomato",
        "potato": "vegetables/veg200_images/potato",
        "carrot": "vegetables/veg200_images/carrot",
        "broccoli": "vegetables/veg200_images/broccoli",
    },
    "rice": {
        "arborio": "rice/Rice_Image_Dataset/Arborio",
        "basmati": "rice/Rice_Image_Dataset/Basmati",
        "ipsala": "rice/Rice_Image_Dataset/Ipsala",
        "jasmine": "rice/Rice_Image_Dataset/Jasmine",
        "karacadag": "rice/Rice_Image_Dataset/Karacadag",
    },
    "seafood": {
        "shrimp": "seafood/Seafood/Train/Shrimp",
        "salmon": "seafood/Seafood/Train/Salmon",
        "crab": "seafood/Seafood/Train/Crab",
    },
    "egg": {
        "egg": "egg/Eggs Classification/Not Damaged",
    },
    "oil": {
        "oil": [
            "oil/train",
            "oil/test",
            "oil/valid",
        ],
    },
}

# Per-class image limits. If a class is listed here, each of its
# subclasses will be capped at this many images.
CLASS_LIMITS = {
    "rice": 500,
}


def copy_images(src_path, dst_path, max_images=None):
    """
    Copy all valid image files from src_path into dst_path.
    Returns the count of images copied.
    """
    dst_path.mkdir(parents=True, exist_ok=True)
    count = 0

    if not src_path.exists():
        return 0

    for f in sorted(src_path.iterdir()):
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
            new_name = f"{src_path.name}_{f.name}"
            shutil.copy2(f, dst_path / new_name)
            count += 1
            if max_images and count >= max_images:
                break

    return count


def organize():
    total_all = 0

    for cls_name, subclasses in CLASS_MAP.items():
        print(f"\n[{cls_name.upper()}]")

        # Check if this class has a per-subclass image limit
        limit = CLASS_LIMITS.get(cls_name, None)

        for sub_name, src_paths in subclasses.items():
            dst = ORGANIZED_DIR / cls_name / sub_name

            if isinstance(src_paths, str):
                src_paths = [src_paths]

            total_sub = 0
            for sp in src_paths:
                full_src = RAW_DIR / sp
                if full_src.exists():
                    # If there's a limit, subtract what we've already copied
                    remaining = (limit - total_sub) if limit else None
                    if limit and remaining <= 0:
                        break
                    n = copy_images(full_src, dst, max_images=remaining)
                    total_sub += n
                else:
                    print(f"  WARNING: Path not found: {full_src}")

            if limit:
                print(f"  {sub_name}: {total_sub} images (capped at {limit})")
            else:
                print(f"  {sub_name}: {total_sub} images")
            total_all += total_sub

    print(f"\n{'=' * 50}")
    print(f"TOTAL: {total_all} images copied to {ORGANIZED_DIR}")


if __name__ == "__main__":
    if ORGANIZED_DIR.exists():
        print(f"Removing existing {ORGANIZED_DIR}...")
        shutil.rmtree(ORGANIZED_DIR)

    print("Organizing datasets from raw/ -> raw_organized/")
    organize()
