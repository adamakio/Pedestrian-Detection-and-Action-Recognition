#!/usr/bin/env python3
import torch
import numpy as np
import json
from pathlib import Path
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# import real model & loss code
from train import build_model, build_ce_by_head, multitask_loss, _average_precision, _softmax_np
from _dataset import CATS

"""
Train classification heads ONLY using cached backbone features.
Run:
    python dev_scripts/train_heads_on_cache.py
"""

DEVICE = "mps"
LR = 1e-3
EPOCHS = 5
BATCH = 64

SPLIT_DIR = Path("action_det/data/splits_pct_20_T8_S4")
LABEL_SPACE_JSON = SPLIT_DIR / "label_space.json"
TRAIN_LABEL_REPORT = SPLIT_DIR / "train_label_report.json"

FEAT_DIR = Path("cached_feats")
TRAIN_FEATS = FEAT_DIR / "train_feats.pt"
TRAIN_LABS  = FEAT_DIR / "train_labels.pt"
VAL_FEATS   = FEAT_DIR / "val_feats.pt"
VAL_LABS    = FEAT_DIR / "val_labels.pt"


# ───────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────
def load_label_space():
    js = json.loads(LABEL_SPACE_JSON.read_text())
    return js["label_space"]

def load_label_report():
    js = json.loads(TRAIN_LABEL_REPORT.read_text())
    return js

def patch_forward_from_features(model):
    """Adds model.forward_from_features for cached features."""
    def forward_feats(self, feats):
        return {h: self.hd[h](feats) for h in self.hd.keys()}
    model.forward_from_features = forward_feats.__get__(model, model.__class__)
    return model


def make_loader(feats_path, labels_path):
    feats = torch.load(feats_path).float()
    labels = torch.load(labels_path).long()
    ds = TensorDataset(feats, labels)
    return DataLoader(ds, batch_size=BATCH, shuffle=True)


# ───────────────────────────────────────────────────────────────
# Evaluation (same AP as 04_train)
# ───────────────────────────────────────────────────────────────
def eval_model(model, dl, label_space, ce_cfg):
    all_logits = {h: [] for h in CATS}
    all_targets = {h: [] for h in CATS}

    model.eval()
    with torch.no_grad():
        for feats, labels in dl:
            feats = feats.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model.forward_from_features(feats)

            # collect logits per head
            for i, h in enumerate(CATS):
                all_logits[h].append(out[h].cpu().numpy())
                all_targets[h].append(labels[:, i].cpu().numpy())

    metrics = {}

    for h in CATS:
        logits = np.concatenate(all_logits[h], axis=0)
        targets = np.concatenate(all_targets[h], axis=0).astype(int)
        classes = label_space[h]

        idx_none = classes.index("none")
        cls_indices = [i for i in range(len(classes)) if i != idx_none]

        probs = _softmax_np(logits)

        aps = []
        for ci in cls_indices:
            y_bin = (targets == ci).astype(int)
            if y_bin.sum() == 0:
                continue
            aps.append(_average_precision(y_bin, probs[:, ci]))

        metrics[h] = float(np.mean(aps)) if aps else 0.0

    return metrics


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────
def main():
    label_space = load_label_space()
    train_label_report = load_label_report()

    train_loader = make_loader(TRAIN_FEATS, TRAIN_LABS)
    val_loader   = make_loader(VAL_FEATS,  VAL_LABS)

    # build real model and remove backbone
    model = build_model(label_space).to(DEVICE)

    # Freeze the backbone entirely
    for p in model.bb.parameters():
        p.requires_grad = False

    # Add helper for cached features
    patch_forward_from_features(model)

    ce_cfg = build_ce_by_head(train_label_report, label_smooth=0.1)

    # Only train heads + s params
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR
    )

    print("[INFO] Training heads only on cached features…")
    for epoch in range(1, EPOCHS+1):
        model.train()
        total = 0
        for i, (feats, labels) in enumerate(train_loader):
            feats = feats.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model.forward_from_features(feats)
            loss, _ = multitask_loss(out, labels, model.s, ce_cfg)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += loss.item()

        print(f"Epoch {epoch}: loss={total/len(train_loader):.4f}")

    print("\n[INFO] Learned uncertainty weights (s):")
    print({h: float(model.s[h].detach()) for h in CATS})


    print("\n[INFO] Evaluating model on dev-val subset…")
    metrics = eval_model(model, val_loader, label_space, ce_cfg)

    print("\n===== Dev mAP =====")
    for h, m in metrics.items():
        print(f"{h:20s}: {m:.4f}")
    print("===================")


if __name__ == "__main__":
    main()
