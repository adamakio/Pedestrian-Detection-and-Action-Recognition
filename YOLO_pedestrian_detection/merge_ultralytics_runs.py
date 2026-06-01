import csv
import shutil
from pathlib import Path
import pandas as pd
import yaml

# === EDIT THESE THREE PATHS ===
RUN1 = Path("runs/detect/titan_person_11n_0.1")
RUN2 = Path("runs/detect/titan_person_11n_0.1v2")
OUT  = Path("runs/detect/titan_person_11n_0.1_merged")

# --- helpers ---
def load_results_csv(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"[ERR] {csv_path} not found")
    df = pd.read_csv(csv_path)
    # ensure there is an epoch column; if not, synthesize from index
    if "epoch" not in df.columns:
        df.insert(0, "epoch", range(len(df)))
    return df

def pick_best_metric_columns(df: pd.DataFrame):
    # Prefer mAP50-95, fall back to mAP50
    candidates = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95",
        "metrics/mAP50(B)",
        "metrics/mAP50",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def best_row(df: pd.DataFrame):
    key = pick_best_metric_columns(df)
    if key is None:
        # If no known metric present, just take last row
        return df.iloc[-1], None
    return df.loc[df[key].idxmax()], key

def load_args(run_dir: Path) -> dict:
    y = run_dir / "args.yaml"
    if not y.exists():
        return {}
    with open(y, "r") as f:
        return yaml.safe_load(f)

def dump_args(args: dict, out_path: Path):
    with open(out_path, "w") as f:
        yaml.safe_dump(args, f, sort_keys=False)

def safe_copy(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)

def main():
    print("[INFO] Merging two Ultralytics runs")
    print(f"[INFO] Run1: {RUN1}")
    print(f"[INFO] Run2: {RUN2}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weights").mkdir(parents=True, exist_ok=True)

    # 1) Load result histories
    df1 = load_results_csv(RUN1)
    df2 = load_results_csv(RUN2)

    # 2) Align columns: keep intersection so concatenation is safe
    common_cols = [c for c in df1.columns if c in df2.columns]
    if "epoch" not in common_cols:
        common_cols = ["epoch"] + [c for c in common_cols if c != "epoch"]
    df1c = df1[common_cols].copy()
    df2c = df2[common_cols].copy()

    # 3) Offset epochs for run2 so the timeline is continuous
    last_epoch_run1 = int(df1c["epoch"].max())
    df2c["epoch"] = df2c["epoch"] + last_epoch_run1

    # 4) Concatenate and save merged results.csv
    merged = pd.concat([df1c, df2c], ignore_index=True)
    merged_csv = OUT / "results.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"[OK] Wrote merged results to {merged_csv}")

    # 5) Choose best.pt by best validation metric across runs
    r1_best, key1 = best_row(df1)
    r2_best, key2 = best_row(df2)
    print(f"[INFO] Run1 best metric column: {key1}")
    print(f"[INFO] Run2 best metric column: {key2}")

    def metric_value(row, key):
        if key is None:
            return float("-inf")
        try:
            return float(row[key])
        except Exception:
            return float("-inf")

    v1 = metric_value(r1_best, key1)
    v2 = metric_value(r2_best, key2)
    better = "run1" if v1 >= v2 else "run2"
    print(f"[INFO] Chosen best comes from {better} "
          f"({v1 if v1 != float('-inf') else 'NA'} vs {v2 if v2 != float('-inf') else 'NA'})")

    # 6) Copy weights
    run1_best = RUN1 / "weights" / "best.pt"
    run1_last = RUN1 / "weights" / "last.pt"
    run2_best = RUN2 / "weights" / "best.pt"
    run2_last = RUN2 / "weights" / "last.pt"

    # Keep originals named
    safe_copy(run1_best, OUT / "weights" / "best_run1.pt")
    safe_copy(run2_best, OUT / "weights" / "best_run2.pt")

    # Set merged best.pt
    if better == "run1" and run1_best.exists():
        safe_copy(run1_best, OUT / "weights" / "best.pt")
    elif run2_best.exists():
        safe_copy(run2_best, OUT / "weights" / "best.pt")

    # For last.pt, take the newest (run2 is your continuation)
    safe_copy(run2_last if run2_last.exists() else run1_last, OUT / "weights" / "last.pt")

    print(f"[OK] Weights saved under {OUT/'weights'}")

    # 7) Merge args.yaml (very light merge with a provenance note)
    a1 = load_args(RUN1)
    a2 = load_args(RUN2)
    merged_args = {
        "note": "Merged args from two runs; metrics history concatenated with epoch offset.",
        "source_runs": [str(RUN1), str(RUN2)],
        "run1_args": a1,
        "run2_args": a2,
    }
    dump_args(merged_args, OUT / "args.yaml")
    print(f"[OK] Wrote merged args.yaml")

    # 8) Copy any helpful artifacts if present (optional)
    for art in ["confusion_matrix.png", "labels.jpg", "results.png"]:
        for src in [(RUN2 / art), (RUN1 / art)]:
            if src.exists():
                safe_copy(src, OUT / art)
                break

    print(f"\n[DONE] Merged run at: {OUT}\n"
          f"- results.csv = concatenated history with continuous epochs\n"
          f"- weights/best.pt = best of both runs\n"
          f"- weights/last.pt = final checkpoint from the continuation (run2)\n"
          f"- weights/best_run1.pt & best_run2.pt retained for audit\n")

if __name__ == "__main__":
    main()
