# action_det/scripts/01_build_index.py
import json, csv, math, re
from pathlib import Path
from collections import defaultdict, Counter
from heads import HEADS, COL2HEAD  # HEADS: dict[head] -> list[str]; COL2HEAD: csv_col -> head

DATASET = Path("dataset")
IMG_ROOT = DATASET / "images_anonymized"
CSV_ROOT = DATASET / "titan_0_4"
SPLIT_TXT = {
    "train": DATASET / "train_set.txt",
    "val":   DATASET / "val_set.txt",
    "test":  DATASET / "test_set.txt",
}

def load_clip_list(txt_path, pct: float):
    clips = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
    k = max(1, math.ceil(len(clips) * pct))
    return clips[:k]

def list_frames(clip_dir: Path):
    exts = ("*.jpg", "*.png", "*.jpeg")
    imgs = []
    for pat in exts:
        imgs += sorted((clip_dir / "images").glob(pat))
    if not imgs:
        for pat in exts:
            imgs += sorted((clip_dir / "images" / "test").glob(pat))
    return imgs

def normalize_stem(s: str) -> str | None:
    m = re.search(r"(\d+)", str(s)) if s else None
    return f"{int(m.group(1)):06d}" if m else None

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def build_clip_samples(clip: str, T: int, STRIDE: int):
    csv_path = CSV_ROOT / f"{clip}.csv"
    img_dir  = IMG_ROOT / clip
    frames = list_frames(img_dir)
    if not frames or not csv_path.exists():
        return []

    # map both original stems and normalized 000123 to paths
    idx_by_stem = {p.stem: p for p in frames}
    for p in frames:
        ns = normalize_stem(p.name)
        if ns and ns not in idx_by_stem:
            idx_by_stem[ns] = p

    by_track = defaultdict(list)
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("label","").strip().lower() != "person"):
                continue
            stem = normalize_stem(row.get("frames"))
            if not stem:
                continue
            p = idx_by_stem.get(stem)
            if not p:
                continue
            by_track[row.get("obj_track_id","")].append((stem, row))

    samples = []
    for tid, items in by_track.items():
        if not tid:
            continue
        items.sort(key=lambda x: int(x[0]))
        stems = [s for s,_ in items]
        rows  = [r for _,r in items]

        # sliding windows
        for start in range(0, len(rows) - T + 1, STRIDE):
            sub_rows  = rows[start:start+T]
            sub_stems = stems[start:start+T]

            # labels from last frame of window
            labels = {}
            last = sub_rows[-1]
            for col, head in COL2HEAD.items():
                raw = (last.get(col) or "").strip()
                labels[head] = raw if raw in HEADS[head] else "none"

            # per-frame center boxes (cx, cy, w, h) in pixels
            bboxes_cxcywh, ok = [], True
            for r in sub_rows:
                L = safe_float(r.get("left"))
                Tp= safe_float(r.get("top"))
                W = safe_float(r.get("width"))
                H = safe_float(r.get("height"))
                if None in (L, Tp, W, H):
                    ok = False
                    break
                cx, cy = L + W/2.0, Tp + H/2.0
                bboxes_cxcywh.append([cx, cy, W, H])
            if not ok:
                continue

            # resolve absolute frame paths
            frame_paths = []
            for s in sub_stems:
                p = idx_by_stem.get(s)
                if not p:
                    frame_paths = []
                    break
                frame_paths.append(str(p.resolve()))
            if not frame_paths:
                continue

            samples.append({
                "clip": clip,
                "track_id": tid,
                "frames": frame_paths,            # list[str], length T
                "bboxes_cxcywh": bboxes_cxcywh,   # list[list[float]], length T
                "labels": labels,                 # dict[head] -> str
                "T": T,
                "stride": STRIDE,
            })
    return samples

def main(pct: float, T: int, STRIDE: int):
    tag = f"splits_pct_{int(pct*100)}_T{T}_S{STRIDE}"
    out_root = Path("action_det/data") / tag
    out_root.mkdir(parents=True, exist_ok=True)

    # write meta
    meta = {
        "pct": pct, "T": T, "stride": STRIDE,
        "dataset_root": str(DATASET.resolve()),
        "img_root": str(IMG_ROOT.resolve()),
        "csv_root": str(CSV_ROOT.resolve()),
        "heads": HEADS,
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))

    # aggregate per-split; also accumulate train labels to build label_space.json
    train_label_counts = {h: Counter() for h in HEADS}
    for split, txt in SPLIT_TXT.items():
        clips = load_clip_list(txt, pct)
        all_samples = []
        label_counts = {h: Counter() for h in HEADS}

        for c in clips:
            ss = build_clip_samples(c, T, STRIDE)
            all_samples.extend(ss)
            for s in ss:
                for h, lab in s["labels"].items():
                    label_counts[h][lab] += 1
                    if split == "train":
                        train_label_counts[h][lab] += 1

        idx_path = out_root / f"{split}.jsonl"
        with idx_path.open("w") as f:
            for s in all_samples:
                f.write(json.dumps(s) + "\n")

        report = {h: dict(cnt) for h, cnt in label_counts.items()}
        (out_root / f"{split}_label_report.json").write_text(json.dumps(report, indent=2))

        print(f"[{split}] clips={len(clips)} windows={len(all_samples)} -> {idx_path}")
        for h in HEADS:
            present = [k for k,v in label_counts[h].items() if v>0]
            print(f"  - {h}: {len(present)} labels present")

    # Build label_space.json from TRAIN present labels (+ ensure 'none' exists and is first)
    label_space = {}
    for h, cnt in train_label_counts.items():
        present = sorted([k for k,v in cnt.items() if v>0 and k != "none"])
        label_space[h] = ["none"] + present
    label_space_payload = {"label_space": label_space, "T": T, "stride": STRIDE}
    (out_root / "label_space.json").write_text(json.dumps(label_space_payload, indent=2))
    print(f"[ok] wrote {out_root/'label_space.json'}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, default=0.1, help="fraction of clips to use per split (0-1]")
    ap.add_argument("--T", type=int, default=16, help="window length (frames)")
    ap.add_argument("--stride", type=int, default=8, help="sliding window stride (frames)")
    args = ap.parse_args()
    assert 0 < args.pct <= 1.0
    assert args.T > 0 and args.stride > 0
    main(args.pct, args.T, args.stride)
