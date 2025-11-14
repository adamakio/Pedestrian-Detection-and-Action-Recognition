#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from tqdm import tqdm

# --- plotting (non-interactive) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _dataset import TubeDataset, CATS  # fixed order of heads

# --- CONSTANTS ---
NUM_VIS = 20  # default number of samples to visualize

# ---------- helpers ----------
def softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

def pr_curve_points(y_true_bin: np.ndarray, y_score: np.ndarray):
    """
    y_true_bin: (N,) in {0,1}
    y_score: (N,) float scores
    Returns (precision[], recall[], thresholds[])
    """
    order = np.argsort(-y_score)
    y = y_true_bin[order]
    s = y_score[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    denom = np.maximum(tp + fp, 1)
    precision = tp / denom
    recall = tp / max(int((y == 1).sum()), 1)

    # thresholds correspond to sorted scores
    thresholds = s

    # add endpoints for nicer plots
    precision = np.concatenate(([1.0], precision, [0.0]))
    recall    = np.concatenate(([0.0], recall,    [1.0]))
    thresholds= np.concatenate(([thresholds[0] if len(thresholds) else 1.0], thresholds, [0.0]))
    return precision, recall, thresholds

def average_precision(y_true_bin, y_score):
    P, R, _ = pr_curve_points(y_true_bin, y_score)
    # Monotone precision envelope
    for i in range(P.size - 1, 0, -1):
        P[i-1] = max(P[i-1], P[i])
    # Trapezoidal integral over PR
    idx = np.where(R[1:] != R[:-1])[0]
    ap = np.sum((R[idx + 1] - R[idx]) * P[idx + 1])
    return float(ap)

def build_model(label_space):
    backbone = r3d_18(weights="KINETICS400_V1")
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    heads = nn.ModuleDict({cat: nn.Linear(feat_dim, len(lbls))
                           for cat, lbls in label_space.items()})
    class MultiHead(nn.Module):
        def __init__(self, bb, hd):
            super().__init__()
            self.bb = bb
            self.hd = hd
            self.s  = nn.ParameterDict({k: nn.Parameter(torch.zeros(1)) for k in hd.keys()})
        def forward(self, x):
            f = self.bb(x)
            return {k: self.hd[k](f) for k in self.hd.keys()}
    return MultiHead(backbone, heads)

def ensure_xywh_per_frame(item):
    T = len(item.get("frames", []))
    def cxcywh_to_xywh(b):
        cx, cy, w, h = map(float, b)
        return [cx - w/2.0, cy - h/2.0, w, h]
    if "bboxes_xywh" in item:      return [list(map(float, b)) for b in item["bboxes_xywh"]]
    if "bboxes_cxcywh" in item:    return [cxcywh_to_xywh(b) for b in item["bboxes_cxcywh"]]
    if "bbox_xywh" in item:        return [list(map(float, item["bbox_xywh"]))] * T
    if "bbox" in item:             return [list(map(float, item["bbox"]))] * T
    if "bbox_cxcywh" in item:      return [cxcywh_to_xywh(item["bbox_cxcywh"])] * T
    for k in ("boxes","bboxes"):
        if k in item and len(item[k]) and len(item[k][0]) == 4:
            return [list(map(float, b)) for b in item[k]]
    raise KeyError("No bbox-like field for visualization.")

def draw_box_and_label(img_path, box_xywh, text, out_path):
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    x, y, w, h = box_xywh
    tlx = int(max(0, math.floor(x)))
    tly = int(max(0, math.floor(y)))
    brx = int(min(W, math.ceil(x + w)))
    bry = int(min(H, math.ceil(y + h)))
    draw = ImageDraw.Draw(im)
    draw.rectangle([tlx, tly, brx, bry], outline=(0, 255, 0), width=3)
    try: font = ImageFont.truetype("Arial.ttf", 16)
    except: font = ImageFont.load_default()
    draw.text((tlx, max(0, tly - 18)), text, fill=(255, 255, 0), font=font)
    im.save(out_path)

def plot_pr_curve(cat, cname, P, R, out_png):
    plt.figure(figsize=(4,4))
    plt.plot(R, P, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR: {cat} / {cname}")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_confusion_png(cm, classes, title, out_png):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    # annotate counts
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            color = "white" if v > thresh else "black"
            plt.text(j, i, str(v), ha="center", va="center", color=color, fontsize=8)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--device", type=str, default="mps")
    args = ap.parse_args()

    run_dir = Path(f"action_det/runs/r3d18_pct{int(args.pct*100)}_T{args.T}_S{args.stride}_img{args.imgsz}_b{args.batch}")
    ckpt = run_dir / "best.pt"
    split_dir = Path(f"action_det/data/splits_pct_{int(args.pct*100)}_T{args.T}_S{args.stride}")
    test_jsonl = split_dir / "test.jsonl"
    label_space_json = split_dir / "label_space.json"

    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    assert test_jsonl.exists(), f"Missing test index: {test_jsonl}"
    assert label_space_json.exists(), "Run 02_build_label_space.py first"

    out_root = Path(f"action_det/eval/r3d18_pct{int(args.pct*100)}_T{args.T}_S{args.stride}_img{args.imgsz}_b{args.batch}")
    (out_root / "confusion").mkdir(parents=True, exist_ok=True)
    (out_root / "confusion_png").mkdir(parents=True, exist_ok=True)
    (out_root / "pr_curves").mkdir(parents=True, exist_ok=True)
    (out_root / "vis_frames").mkdir(parents=True, exist_ok=True)

    cache_npz = out_root / "test_logits_targets_cache.npz"
    recompute = False  # set True if you want to force re-run

    meta = json.loads(label_space_json.read_text())
    label_space = meta["label_space"]
    print(f"[INFO] Heads/classes:")
    for cat in CATS:
        print(f"  - {cat}: {len(label_space[cat])} classes -> {label_space[cat]}")

    device = torch.device(args.device if
                          (args.device == "cuda" and torch.cuda.is_available()) or
                          (args.device == "mps"  and torch.backends.mps.is_available())
                          else "cpu")
    print(f"[INFO] Using device: {device}")

    model = build_model(label_space).to(device)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    ds_te = TubeDataset(test_jsonl, label_space_json, img_size=args.imgsz)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)

    if cache_npz.exists() and not recompute:
        print(f"[INFO] Loading cached logits/targets from {cache_npz}")
        data = np.load(cache_npz, allow_pickle=True)
        all_logits = {k: data[f"logits_{k}"] for k in CATS}
        all_targets = {k: data[f"targets_{k}"] for k in CATS}
    else:
        all_logits = {cat: [] for cat in CATS}
        all_targets = {cat: [] for cat in CATS}

        print("[INFO] Running inference on test set…")
        for xb, yb in tqdm(dl_te, desc="Test batches", leave=True):
            xb = xb.to(device)
            with torch.no_grad():
                out = model(xb)
            for cat in CATS:
                all_logits[cat].append(out[cat].cpu().numpy())
                all_targets[cat].append(yb[:, CATS.index(cat)].cpu().numpy())

        for cat in CATS:
            all_logits[cat]  = np.concatenate(all_logits[cat],  axis=0)
            all_targets[cat] = np.concatenate(all_targets[cat], axis=0).astype(int)

        # save cache for fast resume
        save_dict = {}
        for cat in CATS:
            save_dict[f"logits_{cat}"]  = all_logits[cat]
            save_dict[f"targets_{cat}"] = all_targets[cat]
        np.savez_compressed(cache_npz, **save_dict)
        print(f"[INFO] Cached logits/targets to {cache_npz}")

    metrics = {"per_head": {}, "overall": {}}
    csv_rows = [["category", "metric", "value"]]
    per_head_maps, per_head_acc = [], []

    print("[INFO] Computing per-head metrics + PR curves…")
    for cat in tqdm(CATS, desc="Heads", leave=False):
        classes = label_space[cat]
        k = len(classes)
        logits = all_logits[cat]
        probs  = softmax_np(logits)
        y_true = all_targets[cat]

        # Accuracy
        preds = probs.argmax(axis=1)
        acc = float((preds == y_true).mean())
        per_head_acc.append(acc)

        # AP per class + PR CSV/PNG
        ap_per_class = {}
        idx_none = classes.index("none") if "none" in classes else -1
        class_indices = list(range(k))
        if idx_none >= 0:
            class_indices = [i for i in class_indices if i != idx_none]

        for ci in class_indices:
            cname = classes[ci]
            y_bin = (y_true == ci).astype(int)
            score = probs[:, ci]
            P, R, T = pr_curve_points(y_bin, score)
            ap = average_precision(y_bin, score)
            ap_per_class[cname] = ap

            # save PR CSV
            pr_csv = out_root / "pr_curves" / f"{cat}_{cname}.csv"
            with pr_csv.open("w") as f:
                f.write("recall,precision,threshold\n")
                for r, p, t in zip(R, P, T):
                    f.write(f"{r:.6f},{p:.6f},{t:.6f}\n")

            # save PR PNG
            pr_png = out_root / "pr_curves" / f"{cat}_{cname}.png"
            plot_pr_curve(cat, cname, P, R, pr_png)

        head_map = float(np.mean(list(ap_per_class.values())) if ap_per_class else 0.0)
        per_head_maps.append(head_map)

        metrics["per_head"][cat] = {
            "accuracy": acc,
            "mAP": head_map,
            "AP_per_class": ap_per_class,
            "num_classes": k,
            "classes": classes,
        }
        csv_rows.append([cat, "accuracy", f"{acc:.6f}"])
        csv_rows.append([cat, "mAP"    , f"{head_map:.6f}"])
        for cname, apv in ap_per_class.items():
            csv_rows.append([cat, f"AP[{cname}]", f"{apv:.6f}"])

    overall_map = float(np.mean(per_head_maps) if per_head_maps else 0.0)
    overall_acc = float(np.mean(per_head_acc) if per_head_acc else 0.0)
    metrics["overall"] = {"mAP": overall_map, "accuracy": overall_acc}
    print(f"[RESULTS] Overall mAP: {overall_map:.4f} | Overall acc: {overall_acc:.4f}")

    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out_root / "metrics.csv").open("w") as f:
        for row in csv_rows:
            f.write(",".join(row) + "\n")

    # Confusion matrices: save CSV + colored PNG
    print("[INFO] Saving confusion matrices…")
    for cat in tqdm(CATS, desc="Confusion", leave=False):
        classes = label_space[cat]
        probs   = softmax_np(all_logits[cat])
        preds   = probs.argmax(axis=1)
        y_true  = all_targets[cat]
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(y_true, preds):
            cm[t, p] += 1
        # CSV
        out_csv = out_root / "confusion" / f"{cat}_confusion.csv"
        with out_csv.open("w") as f:
            f.write("," + ",".join(classes) + "\n")
            for i, cname in enumerate(classes):
                f.write(cname + "," + ",".join(str(x) for x in cm[i]) + "\n")
        # PNG heatmap
        out_png = out_root / "confusion_png" / f"{cat}_confusion.png"
        plot_confusion_png(cm, classes, f"Confusion: {cat}", out_png)

    # A few last-frame visualizations
    print("[INFO] Saving a few visualization frames…")
    items = [json.loads(l) for l in test_jsonl.read_text().splitlines()]
    vis_n = min(NUM_VIS, len(items))
    order_preds = {cat: softmax_np(all_logits[cat]).argmax(axis=1) for cat in CATS}

    for i in tqdm(range(vis_n), desc="Vis frames", leave=False):
        it = items[i]
        frames = it.get("frames", [])
        if not frames:
            continue
        try:
            bboxes = ensure_xywh_per_frame(it)
            last_xywh = bboxes[-1]
        except Exception:
            continue
        last_frame = frames[-1]
        label_strs = []
        for cat in CATS:
            classes = label_space[cat]
            pred_idx = int(order_preds[cat][i])
            label_strs.append(f"{cat}:{classes[pred_idx]}")
        text = " | ".join(label_strs)
        out_img = out_root / "vis_frames" / f"sample_{i:03d}.jpg"
        draw_box_and_label(last_frame, last_xywh, text, out_img)

    print(f"[OK] Wrote outputs to: {out_root}")

if __name__ == "__main__":
    main()
