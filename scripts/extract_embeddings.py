"""
Extract CLIP ViT-L/14 embeddings for train/val/test splits.
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embedder import CLIPEmbedder
from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    EMBEDDINGS_TRAIN,
    EMBEDDINGS_VAL,
    EMBEDDINGS_TEST,
    DATA_DIR,
)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    embedder = CLIPEmbedder()

    for split_name, split_dir, out_path in [
        ("train", TRAIN_DIR, EMBEDDINGS_TRAIN),
        ("val", VAL_DIR, EMBEDDINGS_VAL),
        ("test", TEST_DIR, EMBEDDINGS_TEST),
    ]:
        print(f"\n{'=' * 50}")
        print(f"  Extracting {split_name} embeddings")
        print(f"{'=' * 50}")

        data = embedder.extract_dataset(split_dir)
        np.savez(
            out_path,
            embeddings=data["embeddings"],
            class_labels=data["class_labels"],
            subclass_labels=data["subclass_labels"],
            paths=data["paths"],
        )
        print(f"  Saved {data['embeddings'].shape} to {out_path}")

    embedder.unload()
    print("\nDone!")


if __name__ == "__main__":
    main()
