"""
CLIP ViT-L/14 embedding extraction. Frozen encoder — no training.
Extracts 768D L2-normalized embeddings from images.
"""

import torch
import numpy as np
import open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from src.config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    CLIP_EMBED_DIM,
    CLIP_BATCH_SIZE,
    VALID_EXT,
)


class CLIPEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Embedder] Loading {CLIP_MODEL_NAME} on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        print("[Embedder] Ready.")

    def embed_images(self, image_paths, batch_size=None):
        """
        Encode a list of image paths into L2-normalized embeddings.
        Returns: np.ndarray of shape (N, 768)
        """
        bs = batch_size or CLIP_BATCH_SIZE
        all_emb = []

        for i in tqdm(range(0, len(image_paths), bs), desc="Embedding"):
            batch_paths = image_paths[i : i + bs]
            tensors = []
            for p in batch_paths:
                try:
                    img = self.preprocess(Image.open(p).convert("RGB"))
                    tensors.append(img)
                except Exception as e:
                    print(f"  Skip {p}: {e}")
                    tensors.append(self.preprocess(Image.new("RGB", (224, 224))))

            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                feats = self.model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_emb.append(feats.cpu().float().numpy())

        return np.concatenate(all_emb, axis=0)

    def embed_texts(self, texts):
        """
        Encode a list of text strings into L2-normalized embeddings.
        Returns: np.ndarray of shape (N, 768)
        """
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    def embed_single(self, image_input):
        """Embed a single image (path, Path, or PIL Image)."""
        if isinstance(image_input, (str, Path)):
            image_input = Image.open(image_input).convert("RGB")
        tensor = self.preprocess(image_input).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            feat = self.model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().float().numpy()

    def extract_dataset(self, base_dir):
        """
        Walk base_dir/class/subclass/ and extract embeddings + labels.
        Returns: embeddings (N,768), class_labels (N,), subclass_labels (N,), paths (N,)
        """
        base = Path(base_dir)
        paths, cls_labels, sub_labels = [], [], []

        for cls_dir in sorted(base.iterdir()):
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
                for img in imgs:
                    paths.append(str(img))
                    cls_labels.append(cls_dir.name)
                    sub_labels.append(sub_dir.name)

        print(f"[Embedder] Found {len(paths)} images in {base}")
        embeddings = self.embed_images(paths)

        return {
            "embeddings": embeddings,
            "class_labels": np.array(cls_labels),
            "subclass_labels": np.array(sub_labels),
            "paths": np.array(paths),
        }

    def unload(self):
        self.model = self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Embedder] GPU freed.")
