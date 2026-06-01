# action_det/scripts/01_build_index.py
import json, csv, math, re
from pathlib import Path
from collections import defaultdict, Counter
from heads import HEADS, COL2HEAD  # HEADS: dict[head] -> list[str]; COL2HEAD: csv_col -> head

# Map raw TITAN strings to our canonical HEADS labels
RAW2CANON = {
    "atomic": {
        "none of the above": "none",
    },
    "simple-context": {
        "none of the above": "none",
        "cleaning an object": "cleaning",
        "crossing a street at pedestrian crossing": "crossing legally",
        "jaywalking (illegally crossing NOT at pedestrian crossing)": "jaywalking",
        "waiting to cross street": "waiting to cross",
        "walking along the side of the road": "walking on the side",
    },
    "complex-context": {
        "none of the above": "none",
        "getting in 4 wheel vehicle": "getting in 4wv",
        "getting off 2 wheel vehicle": "getting off 2wv",
        "getting on 2 wheel vehicle": "getting-on 2wv",
        "getting out of 4 wheel vehicle": "getting-out 4wv",
    },
    "communicative": {
        "none of the above": "none",
        "looking into phone": "looking at phone",
    },
    "transportive": {
        "none of the above": "none",
        "carrying with both hands": "carrying",
    },
}




DATASET = Path("dataset")
IMG_ROOT = DATASET / "images_anonymized"
CSV_ROOT = DATASET / "titan_0_4"
SPLIT_TXT = {
    "train": DATASET / "train_set.txt",
    "val":   DATASET / "val_set.txt",
    "test":  DATASET / "test_set.txt",
}
RARE_STRIDE = 1


def normalize_label(head: str, raw: str) -> str | None:
    """
    Map a raw TITAN label string to our canonical HEADS label.
    Returns None if the label is empty / unusable.
    """
    if not raw:
        return None
    raw = raw.strip()
    # exact mapping first
    canon_map = RAW2CANON.get(head, {})
    if raw in canon_map:
        return canon_map[raw]
    return raw  # fall back to raw; will be checked against HEADS later


def frame_has_action(row):
    """
    Return True if this annotation row has any non-'none' label in any head.
    """
    for col, head in COL2HEAD.items():
        raw = (row.get(col) or "").strip()
        if raw and raw.lower() != "none of the above" and raw.lower() != "walking" and raw.lower() != "walking on the road":
            # print(raw)
            return True
    return False

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

        # Decide if this track is "rare" (has any action frames)
        is_rare_track = any(frame_has_action(r) for r in rows)
        stride_for_track = RARE_STRIDE if is_rare_track else STRIDE

        # sliding windows
                # sliding windows
        for start in range(0, len(rows) - T + 1, stride_for_track):
            sub_rows  = rows[start:start+T]
            sub_stems = stems[start:start+T]

            # labels by majority vote over *normalized* labels
            labels = {}
            for col, head in COL2HEAD.items():
                per_frame = []
                for r in sub_rows:
                    raw = (r.get(col) or "").strip()
                    canon = normalize_label(head, raw)
                    if canon and canon in HEADS[head]:
                        per_frame.append(canon)

                if per_frame:
                    counts = Counter(per_frame)
                    majority_label, _ = counts.most_common(1)[0]
                    labels[head] = majority_label
                else:
                    labels[head] = "none"


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
