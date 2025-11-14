#!/usr/bin/env python3
import json, argparse
from pathlib import Path

CATS = ["atomic", "simple-context", "complex-context", "communicative", "transportive"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, required=True, help="e.g., 0.20")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()

    report_p = Path(f"action_det/data/splits_pct_{int(args.pct*100)}_T{args.T}_S{args.stride}/train_label_report.json")
    assert report_p.exists(), f"Missing {report_p}"
    report = json.loads(report_p.read_text())

    label_space = {}
    for cat in CATS:
        counts = report.get(cat, {})
        labels = [k for k, v in counts.items() if v > 0]
        # guarantee 'none' exists and is first for each head
        if "none" not in labels:
            labels = ["none"] + labels
        else:
            labels = ["none"] + [x for x in labels if x != "none"]
        label_space[cat] = labels

    out_dir = Path(f"action_det/data/splits_pct_{int(args.pct*100)}_T{args.T}_S{args.stride}")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "pct": args.pct,
        "T": args.T,
        "stride": args.stride,
        "label_space": label_space
    }
    (out_dir / "label_space.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] Wrote {out_dir/'label_space.json'}")

if __name__ == "__main__":
    main()
