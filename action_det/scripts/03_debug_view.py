# action_det/scripts/04_debug_view.py
import argparse, json, imageio
from pathlib import Path
from _dataset import TubeDataset
import numpy as np
import torch

MEAN = [0.43216, 0.394666, 0.37645]
STD  = [0.22803, 0.22145, 0.216989]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", type=str, default="action_det/debug_viz")
    ap.add_argument("--num", type=int, default=3)
    args = ap.parse_args()

    split_dir = Path(f"action_det/data/splits_pct_{int(args.pct*100)}_T{args.T}_S{args.stride}")
    jsonl = split_dir / f"{args.split}.jsonl"
    label_space_json = split_dir / "label_space.json"

    # train=False so we don’t apply rare-class aug, but normalization still happens
    ds = TubeDataset(jsonl, label_space_json, img_size=args.img_size, train=False)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.num, len(ds))):
        tube, targets = ds[i]      # tube: (C,T,H,W), normalized

        # --- un-normalize back to [0,1] ---
        tube_vis = tube.clone()
        for c in range(3):
            tube_vis[c] = tube_vis[c] * STD[c] + MEAN[c]
        tube_vis.clamp_(0.0, 1.0)

        # (C,T,H,W) -> (T,H,W,C), uint8
        tube_vis = tube_vis.permute(1, 2, 3, 0).cpu().numpy()  # (T,H,W,C)
        frames = (tube_vis * 255).astype(np.uint8)

        savep = outdir / f"sample_{i:03d}.mp4"
        imageio.mimsave(savep, list(frames), fps=8)
        print(f"[OK] wrote {savep}")

if __name__ == "__main__":
    main()
