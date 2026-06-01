#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt

def run_dir_from_args(args):
    tag = f"r3d18_pct{int(args.pct*100)}_T{args.T}_S{args.stride}_img{args.imgsz}_b{args.batch}"
    # primary location (your current runs live here)
    rd = Path("runs") / tag
    if rd.exists():
        return rd
    # fallback if you ever moved runs under action_det/runs
    rd2 = Path("action_det") / "runs" / tag
    return rd2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--out", type=str, default=None, help="Optional custom output dir")
    args = ap.parse_args()

    rd = run_dir_from_args(args)
    results_csv = rd / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found at {results_csv}")

    epochs, train_loss, val_loss = [], [], []
    # Be tolerant to column naming variations
    with results_csv.open() as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        # common names in our trainer
        tr_keys = [c for c in cols if c.lower() in ("train_loss","train/loss","loss_train")]
        va_keys = [c for c in cols if c.lower() in ("val_loss","val/loss","loss_val","valid_loss")]
        if not tr_keys or not va_keys:
            raise ValueError(f"Cannot find train/val loss columns in {cols}")
        tr_k, va_k = tr_keys[0], va_keys[0]
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row[tr_k]))
            val_loss.append(float(row[va_k]))

    out_root = Path(args.out) if args.out else (rd / "plots")
    out_root.mkdir(parents=True, exist_ok=True)
    out_png = out_root / "loss_train_val.png"

    plt.figure(figsize=(7,5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss,   label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train vs Val Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    print(f"[OK] Saved {out_png}")

if __name__ == "__main__":
    main()
