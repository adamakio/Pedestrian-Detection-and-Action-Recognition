# action_det/scripts/04_debug_view.py
import argparse, json, imageio
from pathlib import Path
from _dataset import TubeDataset

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

    ds = TubeDataset(jsonl, label_space_json, img_size=args.img_size)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.num, len(ds))):
        tube, targets = ds[i]  # (C,T,H,W)
        C,T,H,W = tube.shape
        frames = (tube.permute(1,0,2,3) * 255).byte().numpy()  # (T,C,H,W) uint8
        frames = [frames[t].transpose(1,2,0) for t in range(T)] # list of HxWxC
        savep = outdir / f"sample_{i:03d}.mp4"
        imageio.mimsave(savep, frames, fps=8)
        print(f"[OK] wrote {savep}")

if __name__ == "__main__":
    main()
