#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
RUN_DIR  = Path("runs/detect/titan_person_11n_0.1_merged")
CSV_PATH = RUN_DIR / "results.csv"
OUT_DIR  = RUN_DIR / "plots"
EPOCH    = "epoch"

# column names from your header:
COLS = {
    "train_box": "train/box_loss",
    "train_cls": "train/cls_loss",
    "train_dfl": "train/dfl_loss",
    "val_box":   "val/box_loss",
    "val_cls":   "val/cls_loss",
    "val_dfl":   "val/dfl_loss",
    "prec":      "metrics/precision(B)",
    "rec":       "metrics/recall(B)",
    "map50":     "metrics/mAP50(B)",
    "map5095":   "metrics/mAP50-95(B)",
    "lr0":       "lr/pg0",
    "lr1":       "lr/pg1",
    "lr2":       "lr/pg2",
}

def safe_plot_two(df, x, y1, y2, y1_label, y2_label, title, ylabel, outpath):
    if y1 not in df.columns and y2 not in df.columns:
        return False
    plt.figure()
    if y1 in df.columns:
        plt.plot(df[x], df[y1], label=y1_label)
    if y2 in df.columns:
        plt.plot(df[x], df[y2], label=y2_label)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
    return True

def safe_plot_one(df, x, y, title, ylabel, outpath):
    if y not in df.columns:
        return False
    plt.figure()
    plt.plot(df[x], df[y], label=y)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
    return True

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    df = pd.read_csv(CSV_PATH)
    if EPOCH not in df.columns:
        df.insert(0, EPOCH, range(len(df)))

    # 1) Losses: train vs val on the same figure
    safe_plot_two(
        df, EPOCH, COLS["train_box"], COLS["val_box"],
        "train/box_loss", "val/box_loss",
        "Box Loss (train vs val)", "loss",
        OUT_DIR / "loss_box.png"
    )
    safe_plot_two(
        df, EPOCH, COLS["train_cls"], COLS["val_cls"],
        "train/cls_loss", "val/cls_loss",
        "Cls Loss (train vs val)", "loss",
        OUT_DIR / "loss_cls.png"
    )
    safe_plot_two(
        df, EPOCH, COLS["train_dfl"], COLS["val_dfl"],
        "train/dfl_loss", "val/dfl_loss",
        "DFL Loss (train vs val)", "loss",
        OUT_DIR / "loss_dfl.png"
    )

    # 2) Validation-only metrics (no train counterpart in CSV)
    safe_plot_one(
        df, EPOCH, COLS["prec"],
        "Validation Precision (B)", "precision",
        OUT_DIR / "val_precision_B.png"
    )
    safe_plot_one(
        df, EPOCH, COLS["rec"],
        "Validation Recall (B)", "recall",
        OUT_DIR / "val_recall_B.png"
    )
    safe_plot_one(
        df, EPOCH, COLS["map50"],
        "Validation mAP@0.50 (B)", "mAP@0.50",
        OUT_DIR / "val_mAP50_B.png"
    )
    safe_plot_one(
        df, EPOCH, COLS["map5095"],
        "Validation mAP@0.50:0.95 (B)", "mAP@0.50:0.95",
        OUT_DIR / "val_mAP50-95_B.png"
    )

    # 3) Learning rate schedules
    lr_cols_present = [c for c in (COLS["lr0"], COLS["lr1"], COLS["lr2"]) if c in df.columns]
    if lr_cols_present:
        plt.figure()
        for c in lr_cols_present:
            plt.plot(df[EPOCH], df[c], label=c)
        plt.xlabel("epoch")
        plt.ylabel("learning rate")
        plt.title("Learning Rate Schedules")
        plt.legend()
        plt.tight_layout()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUT_DIR / "lr_schedules.png", dpi=200)
        plt.close()

    print(f"[OK] Plots written to: {OUT_DIR}")

if __name__ == "__main__":
    main()
