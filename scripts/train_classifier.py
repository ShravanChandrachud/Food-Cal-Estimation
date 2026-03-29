"""
Multi-label MLP classifier head.
Trained on frozen CLIP ViT-L/14 embeddings.
Uses sigmoid + BCEWithLogitsLoss for independent per-class predictions.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from src.config import (
    MLP_HIDDEN,
    MLP_DROPOUT,
    MLP_LR,
    MLP_EPOCHS,
    MLP_BATCH,
    CLIP_EMBED_DIM,
    MLP_MODEL_PATH,
    THRESHOLDS_PATH,
    CLASS_NAMES_PATH,
)


class IngredientMLP(nn.Module):
    """768 → 256 → N_classes with sigmoid output."""

    def __init__(
        self,
        n_classes,
        embed_dim=CLIP_EMBED_DIM,
        hidden=MLP_HIDDEN,
        dropout=MLP_DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)  # raw logits — apply sigmoid at inference


class MultiLabelClassifier:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.thresholds = {}  # {class_name: float}

    # ── Training ─────────────────────────────────────────────
    def train(
        self,
        X_train,
        Y_train,
        X_val,
        Y_val,
        class_names,
        epochs=None,
        lr=None,
        batch_size=None,
    ):
        """
        Train the MLP on precomputed embeddings.

        Args:
            X_train: np.ndarray (N, 768) — CLIP embeddings
            Y_train: np.ndarray (N, C) — binary multi-label matrix
            X_val:   np.ndarray (M, 768)
            Y_val:   np.ndarray (M, C)
            class_names: list of C class name strings
        """
        epochs = epochs or MLP_EPOCHS
        lr = lr or MLP_LR
        batch_size = batch_size or MLP_BATCH
        n_classes = len(class_names)
        self.class_names = class_names

        self.model = IngredientMLP(n_classes)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Class weights for imbalanced data
        pos_count = Y_train.sum(axis=0).astype(float)
        neg_count = Y_train.shape[0] - pos_count
        pos_weight = torch.tensor(neg_count / (pos_count + 1e-6), dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / len(train_ds)
                val_f1 = self._quick_f1(X_val, Y_val, threshold=0.5)
                print(
                    f"  Epoch {epoch + 1:3d}/{epochs}: "
                    f"loss={avg_loss:.4f}  val_macro_f1={val_f1:.4f}"
                )

        # Tune thresholds on validation set
        print("\n[Classifier] Tuning per-class thresholds on validation set...")
        self.thresholds = self._tune_thresholds(X_val, Y_val)
        for name, t in self.thresholds.items():
            print(f"  {name:20s} → threshold={t:.2f}")

        return self

    def _quick_f1(self, X, Y, threshold=0.5):
        """Compute macro F1 for a quick progress check."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy()
        preds = (probs >= threshold).astype(int)
        self.model.train()

        from sklearn.metrics import f1_score

        return f1_score(Y, preds, average="macro", zero_division=0)

    def _tune_thresholds(self, X_val, Y_val, min_threshold=0.75):
        """
        Sweep thresholds per class on validation set, pick best F1.
        Enforces a minimum threshold floor to reduce false positives
        on complex scenes (cooking videos) vs isolated training images.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_val, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy()

        from sklearn.metrics import f1_score

        thresholds = {}
        for i, name in enumerate(self.class_names):
            best_t, best_f1 = min_threshold, 0.0
            for t in np.arange(min_threshold, 0.96, 0.01):
                pred_col = (probs[:, i] >= t).astype(int)
                f1 = f1_score(Y_val[:, i], pred_col, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            thresholds[name] = round(float(best_t), 2)
        return thresholds

    # ── Inference ────────────────────────────────────────────
    def predict(self, embeddings):
        """
        Multi-label predict on precomputed embeddings.
        Args: embeddings — np.ndarray (N, 768) or (768,)
        Returns: list of list of (class_name, probability) tuples
        """
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(embeddings, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy()

        results = []
        for row in probs:
            detected = []
            for i, name in enumerate(self.class_names):
                t = self.thresholds.get(name, 0.5)
                if row[i] >= t:
                    detected.append((name, round(float(row[i]), 4)))
            detected.sort(key=lambda x: x[1], reverse=True)
            results.append(detected)
        return results

    def predict_single(self, embedding):
        """Predict for a single embedding vector. Returns [(name, prob), ...]"""
        return self.predict(embedding)[0]

    # ── Save / Load ──────────────────────────────────────────
    def save(self, model_path=None, thresh_path=None, names_path=None):
        model_path = Path(model_path or MLP_MODEL_PATH)
        thresh_path = Path(thresh_path or THRESHOLDS_PATH)
        names_path = Path(names_path or CLASS_NAMES_PATH)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        with open(thresh_path, "w", encoding="utf-8") as f:
            json.dump(self.thresholds, f, indent=2)
        with open(names_path, "w", encoding="utf-8") as f:
            json.dump(self.class_names, f, indent=2)
        print(f"[Classifier] Saved model to {model_path}")

    def load(self, model_path=None, thresh_path=None, names_path=None):
        model_path = Path(model_path or MLP_MODEL_PATH)
        thresh_path = Path(thresh_path or THRESHOLDS_PATH)
        names_path = Path(names_path or CLASS_NAMES_PATH)

        with open(names_path, "r") as f:
            self.class_names = json.load(f)
        with open(thresh_path, "r") as f:
            self.thresholds = json.load(f)

        self.model = IngredientMLP(len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        print(f"[Classifier] Loaded {len(self.class_names)} classes from {model_path}")
        return self
