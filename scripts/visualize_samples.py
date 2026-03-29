"""
Generates a comprehensive visual overview of the entire dataset.

Creates 3 outputs:
  1. outputs/class_overview.png     — 5 random samples per CLASS (one row per class)
  2. outputs/subclass_grid.png      — 3 random samples per SUBCLASS (grouped by class)
  3. outputs/class_distribution.png — Bar chart of image counts per subclass
"""

import random
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "datasets" / "train"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

random.seed(SEED)
OUTPUT_DIR.mkdir(exist_ok=True)


def get_images(directory, max_images=None):
    """Get list of image paths from a directory."""
    imgs = [
        f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXT
    ]
    random.shuffle(imgs)
    if max_images:
        imgs = imgs[:max_images]
    return imgs


def get_dataset_structure():
    """Scan train/ and return {class: {subclass: [image_paths]}}."""
    structure = {}
    for cls_dir in sorted(TRAIN_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        structure[cls_dir.name] = {}
        for sub_dir in sorted(cls_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            imgs = [
                f
                for f in sub_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXT
            ]
            structure[cls_dir.name][sub_dir.name] = imgs
    return structure


def plot_class_overview(structure):
    classes = sorted(structure.keys())
    samples_per_class = 5
    n_classes = len(classes)

    fig, axes = plt.subplots(
        n_classes, samples_per_class, figsize=(3 * samples_per_class, 3 * n_classes)
    )
    fig.suptitle(
        "Dataset Overview — 5 Random Samples Per Class",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )

    for row, cls_name in enumerate(classes):
        all_imgs = []
        for sub_imgs in structure[cls_name].values():
            all_imgs.extend(sub_imgs)
        random.shuffle(all_imgs)
        samples = all_imgs[:samples_per_class]

        for col in range(samples_per_class):
            ax = axes[row][col] if n_classes > 1 else axes[col]

            if col < len(samples):
                try:
                    img = Image.open(samples[col]).convert("RGB")
                    ax.imshow(img)
                    sub_name = samples[col].parent.name
                    ax.set_xlabel(sub_name, fontsize=8, color="gray")
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha="center", va="center")
            else:
                ax.text(
                    0.5, 0.5, "N/A", ha="center", va="center", fontsize=12, color="gray"
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if col == 0:
                ax.set_ylabel(
                    cls_name.upper(),
                    fontsize=13,
                    fontweight="bold",
                    rotation=0,
                    labelpad=70,
                    va="center",
                )

    plt.tight_layout()
    path = OUTPUT_DIR / "class_overview.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_subclass_grid(structure):
    rows = []
    for cls_name in sorted(structure.keys()):
        for sub_name in sorted(structure[cls_name].keys()):
            rows.append((cls_name, sub_name, structure[cls_name][sub_name]))

    samples_per_sub = 3
    n_rows = len(rows)

    fig, axes = plt.subplots(
        n_rows,
        samples_per_sub + 1,
        figsize=(3 * (samples_per_sub + 1), 2.5 * n_rows),
        gridspec_kw={"width_ratios": [1.5] + [1] * samples_per_sub},
    )
    fig.suptitle(
        "All Subclasses — 3 Random Samples Each",
        fontsize=18,
        fontweight="bold",
        y=1.005,
    )

    class_names = sorted(structure.keys())
    cmap = plt.cm.Set3
    class_colors = {c: cmap(i / len(class_names)) for i, c in enumerate(class_names)}

    for row_idx, (cls_name, sub_name, imgs) in enumerate(rows):
        random.shuffle(imgs)
        samples = imgs[:samples_per_sub]

        ax_label = axes[row_idx][0]
        ax_label.set_facecolor(class_colors[cls_name])
        ax_label.text(
            0.5,
            0.5,
            f"{cls_name}\n{sub_name}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax_label.transAxes,
        )
        ax_label.set_xticks([])
        ax_label.set_yticks([])

        for col in range(samples_per_sub):
            ax = axes[row_idx][col + 1]
            if col < len(samples):
                try:
                    img = Image.open(samples[col]).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha="center", va="center")
            else:
                ax.text(
                    0.5, 0.5, "N/A", ha="center", va="center", fontsize=10, color="gray"
                )
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    path = OUTPUT_DIR / "subclass_grid.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_distribution(structure):
    labels = []
    counts = []
    colors = []

    class_names = sorted(structure.keys())
    cmap = plt.cm.Set3
    class_colors = {c: cmap(i / len(class_names)) for i, c in enumerate(class_names)}

    for cls_name in class_names:
        for sub_name in sorted(structure[cls_name].keys()):
            labels.append(f"{cls_name}/{sub_name}")
            counts.append(len(structure[cls_name][sub_name]))
            colors.append(class_colors[cls_name])

    fig, ax = plt.subplots(figsize=(14, max(6, len(labels) * 0.4)))
    bars = ax.barh(range(len(labels)), counts, color=colors, edgecolor="white")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Images (train split)", fontsize=12)
    ax.set_title("Image Count Per Subclass", fontsize=16, fontweight="bold")

    # Add legend for class colors
    from matplotlib.patches import Patch

    legend_patches = [Patch(facecolor=class_colors[c], label=c) for c in class_names]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    if not TRAIN_DIR.exists():
        print(f"ERROR: {TRAIN_DIR} not found.")
        print("Run organize_data.py and split_data.py first.")
        exit(1)

    print("Scanning dataset structure...")
    structure = get_dataset_structure()

    total_classes = len(structure)
    total_subs = sum(len(subs) for subs in structure.values())
    total_imgs = sum(len(imgs) for subs in structure.values() for imgs in subs.values())
    print(
        f"Found {total_classes} classes, {total_subs} subclasses, {total_imgs:,} images\n"
    )

    print("Generating class overview...")
    plot_class_overview(structure)

    print("Generating subclass grid...")
    plot_subclass_grid(structure)

    print("Generating distribution chart...")
    plot_distribution(structure)

    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")
