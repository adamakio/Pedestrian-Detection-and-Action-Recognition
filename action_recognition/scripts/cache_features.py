#!/usr/bin/env python3
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader

# Import model builder and dataset
from train import build_model
from _dataset import TubeDataset, CATS

"""
Extract backbone features for fast dev experiments.
Run:
    python dev_scripts/cache_features.py
"""

# === CONFIG ===
SPLIT_DIR = Path("action_det/data/splits_pct_20_T8_S4")
TRAIN_JSONL = SPLIT_DIR / "train_dev.jsonl"
VAL_JSONL   = SPLIT_DIR / "val_dev.jsonl"
LABEL_SPACE_JSON = SPLIT_DIR / "label_space.json"

# pretrained full model
CKPT = Path("action_det/runs/r3d18_pct20_T8_S4_img224_b24/best.pt")

OUT_DIR = Path("action_recognition/cached_feats")
OUT_DIR.mkdir(exist_ok=True)

TRAIN_FEATS = OUT_DIR / "train_feats.pt"
TRAIN_LABS  = OUT_DIR / "train_labels.pt"
VAL_FEATS   = OUT_DIR / "val_feats.pt"
VAL_LABS    = OUT_DIR / "val_labels.pt"


def load_label_space():
    js = json.loads(LABEL_SPACE_JSON.read_text())
    return js["label_space"]


def load_trained_backbone():
    """Loads your main model and returns only the backbone."""
    label_space = load_label_space()
    model = build_model(label_space)

    ckpt = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("[OK] Loaded pretrained model -> using backbone for feature extraction")
    return model.bb, label_space


def extract(jsonl_path, feats_path, labels_path):
    dataset = TubeDataset(jsonl_path, LABEL_SPACE_JSON)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    backbone, _ = load_trained_backbone()

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (tubes, labels) in enumerate(loader):
            feat = backbone(tubes).cpu()  # shape (B,512)
            all_feats.append(feat)
            all_labels.append(labels.cpu())

            if batch_idx % 50 == 0:
                print(f"  batch {batch_idx}: {feat.shape}")

    all_feats  = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    torch.save(all_feats, feats_path)
    torch.save(all_labels, labels_path)

    print(f"[OK] Saved: {feats_path}")
    print(f"[OK] Saved: {labels_path}")


def main():
    print("[INFO] Extracting train_dev features…")
    extract(TRAIN_JSONL, TRAIN_FEATS, TRAIN_LABS)

    print("[INFO] Extracting val_dev features…")
    extract(VAL_JSONL, VAL_FEATS, VAL_LABS)


if __name__ == "__main__":
    main()
