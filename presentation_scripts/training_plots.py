import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---------------------------------------------------------
run_dir = Path("action_det/runs/r3d18_pct20_T16_S4_img112_b24_rare2.5_no_s")
csv_path = run_dir / "results.csv"
out_dir = Path("presentation_outputs/training_plots")
out_dir.mkdir(parents=True, exist_ok=True)

# --- Load results --------------------------------------------------
df = pd.read_csv(csv_path)
epochs = df["epoch"]

# Friendly names for heads
head_long_names = {
    "atomic": "Atomic",
    "simple-context": "Simple-context",
    "complex-context": "Complex-context",
    "communicative": "Communicative",
    "transportive": "Transportive",
}

# ------------------------------------------------------------------
# 1) Loss curves: train vs val
# ------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(epochs, df["train_loss"], label="Train loss")
plt.plot(epochs, df["val_loss"], label="Val loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "loss_curves.png", dpi=300)
plt.close()

# ------------------------------------------------------------------
# 2) mAP curves: macro + per-head mAP
# ------------------------------------------------------------------
plt.figure(figsize=(6, 4))

plt.plot(epochs, df["macro_mAP"], linewidth=2, label="Macro mAP")

for key, nice in head_long_names.items():
    col = f"{key}_mAP"
    if col in df.columns:
        plt.plot(epochs, df[col], linestyle="--", label=f"{nice} mAP")

plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.ylim(0, 1.0)
plt.title("mAP over training")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "map_curves.png", dpi=300)
plt.close()

# ------------------------------------------------------------------
# 3) Accuracy curves: per-head accuracy
# ------------------------------------------------------------------
plt.figure(figsize=(6, 4))

for key, nice in head_long_names.items():
    col = f"{key}_acc"
    if col in df.columns:
        plt.plot(epochs, df[col], linestyle="--", label=f"{nice} acc.")

# Overall avg accuracy if present
if "avg_acc" in df.columns:
    plt.plot(epochs, df["avg_acc"], linewidth=2, label="Average acc.")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.title("Head accuracies over training")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "acc_curves.png", dpi=300)
plt.close()

# ------------------------------------------------------------------
# 4) Loss curves (first 10 epochs only)
# ------------------------------------------------------------------
# Adjust this if your epochs start at 0 instead of 1
df_10 = df[df["epoch"] <= 10].copy()

plt.figure(figsize=(6, 4))
plt.plot(df_10["epoch"], df_10["train_loss"], label="Train loss")
plt.plot(df_10["epoch"], df_10["val_loss"], label="Val loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss (first 10 epochs)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "loss_curves_first10.png", dpi=300)
plt.close()


print(f"Saved plots to: {out_dir.resolve()}")
