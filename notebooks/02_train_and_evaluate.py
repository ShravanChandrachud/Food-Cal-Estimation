"""
Train the MLP multi-label classifier, evaluate on val/test,
generate all visualizations for the report.
"""

import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path(".").resolve()
    if ROOT.name == "notebooks":
        ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.config import EMBEDDINGS_TRAIN, EMBEDDINGS_VAL, EMBEDDINGS_TEST, OUTPUTS_DIR
from src.classifier import MultiLabelClassifier

OUT = OUTPUTS_DIR / "02_train_eval"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading embeddings...")
train = np.load(EMBEDDINGS_TRAIN, allow_pickle=True)
val = np.load(EMBEDDINGS_VAL, allow_pickle=True)
test = np.load(EMBEDDINGS_TEST, allow_pickle=True)

X_train, train_subs = (
    train["embeddings"],
    np.array([str(s) for s in train["subclass_labels"]]),
)
X_val, val_subs = val["embeddings"], np.array([str(s) for s in val["subclass_labels"]])
X_test, test_subs = (
    test["embeddings"],
    np.array([str(s) for s in test["subclass_labels"]]),
)

class_names = sorted(set(train_subs))
n_classes = len(class_names)
label_to_idx = {n: i for i, n in enumerate(class_names)}

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"Classes ({n_classes}): {class_names}")


def make_labels(subs):
    Y = np.zeros((len(subs), n_classes), dtype=np.float32)
    for i, s in enumerate(subs):
        if s in label_to_idx:
            Y[i, label_to_idx[s]] = 1.0
    return Y


Y_train = make_labels(train_subs)
Y_val = make_labels(val_subs)
Y_test = make_labels(test_subs)

print(
    f"\nLabel matrices: train={Y_train.shape}, val={Y_val.shape}, test={Y_test.shape}"
)
print(f"Train positives per class:")
for i, n in enumerate(class_names):
    print(f"  {n:20s}: {int(Y_train[:, i].sum()):>5d}")

print("\nGenerating PCA visualization of training embeddings...")
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_train)
ev = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(12, 8))
cmap = plt.cm.tab20
colors = {n: cmap(i / max(n_classes - 1, 1)) for i, n in enumerate(class_names)}
for n in class_names:
    mask = train_subs == n
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[n]], label=n, s=8, alpha=0.5)
ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
ax.set_title("CLIP ViT-L/14 Embeddings — PCA (train)", fontweight="bold")
ax.legend(markerscale=4, fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "pca_embeddings.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {OUT / 'pca_embeddings.png'}")

print(f"\n{'=' * 55}")
print("  Training MLP (768 → 256 → 20)")
print(f"{'=' * 55}")

clf = MultiLabelClassifier()
clf.train(X_train, Y_train, X_val, Y_val, class_names)
clf.save()

print(f"\n{'=' * 55}")
print("  Validation evaluation (tuned thresholds)")
print(f"{'=' * 55}")

clf.model.eval()
with torch.no_grad():
    val_logits = clf.model(torch.tensor(X_val, dtype=torch.float32))
    val_probs = torch.sigmoid(val_logits).numpy()

val_preds = np.zeros_like(val_probs)
for i, n in enumerate(class_names):
    t = clf.thresholds.get(n, 0.5)
    val_preds[:, i] = (val_probs[:, i] >= t).astype(int)

print(
    "\n"
    + classification_report(Y_val, val_preds, target_names=class_names, zero_division=0)
)

macro_f1 = f1_score(Y_val, val_preds, average="macro", zero_division=0)
micro_f1 = f1_score(Y_val, val_preds, average="micro", zero_division=0)
print(f"Macro F1: {macro_f1:.4f}")
print(f"Micro F1: {micro_f1:.4f}")

print(f"\n{'=' * 55}")
print("  Test evaluation (tuned thresholds)")
print(f"{'=' * 55}")

with torch.no_grad():
    test_logits = clf.model(torch.tensor(X_test, dtype=torch.float32))
    test_probs = torch.sigmoid(test_logits).numpy()

test_preds = np.zeros_like(test_probs)
for i, n in enumerate(class_names):
    t = clf.thresholds.get(n, 0.5)
    test_preds[:, i] = (test_probs[:, i] >= t).astype(int)

print(
    "\n"
    + classification_report(
        Y_test, test_preds, target_names=class_names, zero_division=0
    )
)

test_macro = f1_score(Y_test, test_preds, average="macro", zero_division=0)
test_micro = f1_score(Y_test, test_preds, average="micro", zero_division=0)
print(f"Test Macro F1: {test_macro:.4f}")
print(f"Test Micro F1: {test_micro:.4f}")

per_class_f1 = f1_score(Y_test, test_preds, average=None, zero_division=0)

fig, ax = plt.subplots(figsize=(12, max(5, n_classes * 0.35)))
sorted_idx = np.argsort(per_class_f1)
ax.barh([class_names[i] for i in sorted_idx], per_class_f1[sorted_idx], color="#534AB7")
for i, idx in enumerate(sorted_idx):
    ax.text(
        per_class_f1[idx] + 0.01, i, f"{per_class_f1[idx]:.3f}", va="center", fontsize=9
    )
ax.set_xlabel("F1 Score")
ax.set_title(f"Per-Class F1 on Test Set (Macro={test_macro:.3f})", fontweight="bold")
ax.set_xlim(0, 1.1)
plt.tight_layout()
plt.savefig(OUT / "per_class_f1.png", dpi=150, bbox_inches="tight")
plt.show()

true_single = Y_test.argmax(axis=1)
pred_single = test_probs.argmax(axis=1)

fig, ax = plt.subplots(figsize=(max(10, n_classes * 0.6), max(8, n_classes * 0.5)))
cm = confusion_matrix(true_single, pred_single, labels=range(n_classes))
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
ax.set_title("Confusion Matrix — Test Set (argmax)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix_test.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(12, max(5, n_classes * 0.35)))
thresh_vals = [clf.thresholds[n] for n in class_names]
sorted_idx = np.argsort(thresh_vals)
threshold_colors = [
    "#1D9E75" if thresh_vals[i] < 0.5 else "#D85A30" for i in sorted_idx
]
ax.barh(
    [class_names[i] for i in sorted_idx],
    [thresh_vals[i] for i in sorted_idx],
    color=threshold_colors,
)
for i, idx in enumerate(sorted_idx):
    ax.text(
        thresh_vals[idx] + 0.01, i, f"{thresh_vals[idx]:.2f}", va="center", fontsize=9
    )
ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Optimal Threshold")
ax.set_title("Per-Class Tuned Thresholds", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "thresholds.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nGenerating t-SNE visualization (this takes ~1 min)...")
X_pca50 = PCA(n_components=50, random_state=42).fit_transform(X_test)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_pca50)

fig, ax = plt.subplots(figsize=(12, 8))
for n in class_names:
    mask = test_subs == n
    ax.scatter(
        X_tsne[mask, 0], X_tsne[mask, 1], c=[colors[n]], label=n, s=10, alpha=0.6
    )
ax.set_title("CLIP ViT-L/14 Embeddings — t-SNE (test)", fontweight="bold")
ax.legend(markerscale=4, fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "tsne_embeddings.png", dpi=150, bbox_inches="tight")
plt.show()

metrics = {
    "model": "MLP 768→256→20",
    "clip_model": "ViT-L-14 (laion2b_s32b_b82k)",
    "n_classes": n_classes,
    "class_names": class_names,
    "train_size": int(X_train.shape[0]),
    "val_size": int(X_val.shape[0]),
    "test_size": int(X_test.shape[0]),
    "val_macro_f1": round(float(macro_f1), 4),
    "val_micro_f1": round(float(micro_f1), 4),
    "test_macro_f1": round(float(test_macro), 4),
    "test_micro_f1": round(float(test_micro), 4),
    "per_class_f1_test": {
        n: round(float(per_class_f1[i]), 4) for i, n in enumerate(class_names)
    },
    "thresholds": clf.thresholds,
}
with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"\n{'=' * 55}")
print("  SUMMARY")
print(f"{'=' * 55}")
print(f"  Val  Macro F1: {macro_f1:.4f}")
print(f"  Test Macro F1: {test_macro:.4f}")
print(f"  Test Micro F1: {test_micro:.4f}")
print(f"\n  All outputs → {OUT}/")
print("Done!")
