"""
Extracts CLIP embeddings from the organized dataset, runs K-Means
clustering at both class and subclass levels, evaluates with ARI/NMI,
and generates visualizations.
"""

import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings("ignore")

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path(".").resolve()
    if PROJECT_ROOT.name == "notebooks":
        PROJECT_ROOT = PROJECT_ROOT.parent

TRAIN_DIR = PROJECT_ROOT / "datasets" / "train"
OUTPUT_DIR = PROJECT_ROOT / "outputs/01_embedding_and_clustering"
OUTPUT_DIR.mkdir(exist_ok=True)

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Train directory: {TRAIN_DIR}")


print("Loading CLIP ViT-B/32...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
print("CLIP loaded successfully.")


def extract_embeddings(base_dir, model, preprocess, device, batch_size=64):
    """
    Walk through base_dir/class/subclass/images and extract CLIP embeddings.
    Uses batched inference for speed on GPU.

    Returns: embeddings (np.array), class_labels, subclass_labels, file_paths
    """
    all_paths = []
    all_class_labels = []
    all_sub_labels = []

    for cls_dir in sorted(base_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for sub_dir in sorted(cls_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            imgs = [
                f
                for f in sub_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXT
            ]
            for img_path in imgs:
                all_paths.append(img_path)
                all_class_labels.append(cls_dir.name)
                all_sub_labels.append(sub_dir.name)

    print(
        f"Found {len(all_paths)} images across "
        f"{len(set(all_class_labels))} classes, "
        f"{len(set(all_sub_labels))} subclasses"
    )

    all_embeddings = []
    valid_indices = []

    for i in tqdm(range(0, len(all_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = all_paths[i : i + batch_size]
        batch_images = []
        batch_valid = []

        for j, p in enumerate(batch_paths):
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                batch_images.append(img)
                batch_valid.append(i + j)
            except Exception as e:
                print(f"  Skipping {p.name}: {e}")

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy())
        valid_indices.extend(batch_valid)

    embeddings = np.concatenate(all_embeddings, axis=0)

    class_labels = np.array([all_class_labels[i] for i in valid_indices])
    sub_labels = np.array([all_sub_labels[i] for i in valid_indices])
    file_paths = np.array([str(all_paths[i]) for i in valid_indices])

    return embeddings, class_labels, sub_labels, file_paths


embeddings, class_labels, sub_labels, file_paths = extract_embeddings(
    TRAIN_DIR, model, preprocess, device, batch_size=64
)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Classes: {sorted(set(class_labels))}")
print(f"Subclasses: {sorted(set(sub_labels))}")

np.savez(
    OUTPUT_DIR / "train_embeddings.npz",
    embeddings=embeddings,
    class_labels=class_labels,
    subclass_labels=sub_labels,
    file_paths=file_paths,
)
print(f"Saved embeddings to {OUTPUT_DIR / 'train_embeddings.npz'}")


def cluster_and_evaluate(X, y_true, n_clusters, level_name="Class"):
    """
    Run K-Means, map clusters to labels via majority voting, evaluate.
    Returns: predicted labels, kmeans model
    """
    print(f"\n{'=' * 60}")
    print(f"  {level_name}-Level Clustering (k={n_clusters})")
    print(f"{'=' * 60}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    pred = kmeans.fit_predict(X)

    mapping = {}
    for c in range(n_clusters):
        mask = pred == c
        if mask.sum() > 0:
            most_common = Counter(y_true[mask]).most_common(1)[0][0]
            mapping[c] = most_common
        else:
            mapping[c] = "empty_cluster"

    pred_mapped = np.array([mapping[p] for p in pred])

    ari = adjusted_rand_score(y_true, pred)
    nmi = normalized_mutual_info_score(y_true, pred)
    accuracy = (pred_mapped == y_true).mean()

    print(f"\n  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")
    print(f"  Majority-Vote Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, pred_mapped, zero_division=0))

    return pred_mapped, kmeans, mapping


n_classes = len(set(class_labels))
pred_class, kmeans_class, class_mapping = cluster_and_evaluate(
    embeddings, class_labels, n_classes, "Class"
)


n_subclasses = len(set(sub_labels))
pred_sub, kmeans_sub, sub_mapping = cluster_and_evaluate(
    embeddings, sub_labels, n_subclasses, "Subclass"
)


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Plot and save a confusion matrix."""
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) * 0.8)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


plot_confusion_matrix(
    class_labels,
    pred_class,
    "Confusion Matrix — Class-Level Clustering",
    OUTPUT_DIR / "confusion_matrix_class.png",
)

plot_confusion_matrix(
    sub_labels,
    pred_sub,
    "Confusion Matrix — Subclass-Level Clustering",
    OUTPUT_DIR / "confusion_matrix_subclass.png",
)


def plot_embedding_space(X, labels, title, save_path, method="pca"):
    """Reduce to 2D and plot colored by labels."""
    unique_labels = sorted(set(labels))
    n = len(unique_labels)

    print(f"Reducing to 2D with {method.upper()}...")
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        explained = reducer.explained_variance_ratio_
        axis_labels = (f"PC1 ({explained[0]:.1%} var)", f"PC2 ({explained[1]:.1%} var)")
    else:
        X_pca = PCA(n_components=50, random_state=42).fit_transform(X)
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        X_2d = reducer.fit_transform(X_pca)
        axis_labels = ("t-SNE 1", "t-SNE 2")

    cmap = plt.cm.tab20 if n > 10 else plt.cm.tab10
    colors = {lbl: cmap(i / max(n - 1, 1)) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(12, 8))

    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1], c=[colors[lbl]], label=lbl, s=8, alpha=0.6
        )

    ax.set_xlabel(axis_labels[0], fontsize=11)
    ax.set_ylabel(axis_labels[1], fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(markerscale=4, fontsize=8, loc="best", ncol=2 if n > 8 else 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


plot_embedding_space(
    embeddings,
    class_labels,
    "CLIP Embedding Space — Ground Truth Classes (PCA)",
    OUTPUT_DIR / "pca_class_gt.png",
    method="pca",
)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(embeddings)
explained = pca.explained_variance_ratio_

unique_classes = sorted(set(class_labels))
cmap = plt.cm.tab10
colors = {
    lbl: cmap(i / max(len(unique_classes) - 1, 1))
    for i, lbl in enumerate(unique_classes)
}

for ax, (labels, title) in zip(
    axes,
    [
        (class_labels, "Ground Truth"),
        (pred_class, "K-Means Predicted"),
    ],
):
    for lbl in unique_classes:
        mask = labels == lbl
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1], c=[colors[lbl]], label=lbl, s=8, alpha=0.5
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.legend(markerscale=4, fontsize=8)

fig.suptitle(
    "Class-Level: Ground Truth vs K-Means Clusters (PCA)",
    fontsize=15,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_gt_vs_pred.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {OUTPUT_DIR / 'pca_gt_vs_pred.png'}")

plot_embedding_space(
    embeddings,
    class_labels,
    "CLIP Embedding Space — Ground Truth Classes (t-SNE)",
    OUTPUT_DIR / "tsne_class_gt.png",
    method="tsne",
)

plot_embedding_space(
    embeddings,
    sub_labels,
    "CLIP Embedding Space — Ground Truth Subclasses (t-SNE)",
    OUTPUT_DIR / "tsne_subclass_gt.png",
    method="tsne",
)


def analyze_cluster_purity(y_true, kmeans_labels, n_clusters):
    """Show what each cluster contains."""
    print(f"\n{'=' * 60}")
    print(f"  Cluster Purity Analysis")
    print(f"{'=' * 60}")

    for c in range(n_clusters):
        mask = kmeans_labels == c
        total = mask.sum()
        if total == 0:
            continue

        dist = Counter(y_true[mask])
        dominant = dist.most_common(1)[0]
        purity = dominant[1] / total

        print(f"\n  Cluster {c} ({total} images, purity: {purity:.2%}):")
        for label, count in dist.most_common():
            bar = "█" * int(count / total * 30)
            print(f"    {label:<20} {count:>5} ({count / total:>6.1%}) {bar}")


analyze_cluster_purity(class_labels, kmeans_class.labels_, n_classes)

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

ari_class = adjusted_rand_score(class_labels, kmeans_class.labels_)
nmi_class = normalized_mutual_info_score(class_labels, kmeans_class.labels_)
acc_class = (pred_class == class_labels).mean()

ari_sub = adjusted_rand_score(sub_labels, kmeans_sub.labels_)
nmi_sub = normalized_mutual_info_score(sub_labels, kmeans_sub.labels_)
acc_sub = (pred_sub == sub_labels).mean()

print(f"\n  Total images:     {len(embeddings):,}")
print(f"  Embedding dim:    {embeddings.shape[1]}")
print(f"  Classes:          {n_classes}")
print(f"  Subclasses:       {n_subclasses}")
print(f"\n  CLASS-LEVEL (k={n_classes}):")
print(f"    ARI:      {ari_class:.4f}")
print(f"    NMI:      {nmi_class:.4f}")
print(f"    Accuracy: {acc_class:.4f}")
print(f"\n  SUBCLASS-LEVEL (k={n_subclasses}):")
print(f"    ARI:      {ari_sub:.4f}")
print(f"    NMI:      {nmi_sub:.4f}")
print(f"    Accuracy: {acc_sub:.4f}")

summary = {
    "total_images": int(len(embeddings)),
    "embedding_dim": int(embeddings.shape[1]),
    "n_classes": int(n_classes),
    "n_subclasses": int(n_subclasses),
    "class_level": {
        "ARI": float(ari_class),
        "NMI": float(nmi_class),
        "accuracy": float(acc_class),
    },
    "subclass_level": {
        "ARI": float(ari_sub),
        "NMI": float(nmi_sub),
        "accuracy": float(acc_sub),
    },
}

import json

with open(OUTPUT_DIR / "clustering_results.json", "w") as f:
    json.dump(summary, f, indent=4)

print(f"\nResults saved to {OUTPUT_DIR / 'clustering_results.json'}")
print(f"\nGenerated visualizations in {OUTPUT_DIR}/:")
print(f"  - confusion_matrix_class.png")
print(f"  - confusion_matrix_subclass.png")
print(f"  - pca_class_gt.png")
print(f"  - pca_gt_vs_pred.png")
print(f"  - tsne_class_gt.png")
print(f"  - tsne_subclass_gt.png")
