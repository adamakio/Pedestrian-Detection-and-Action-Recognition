import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# --------- CONFIG ---------
# Assuming this file is at titan_data/scripts/verify_titan_annotations.py
ROOT = Path(__file__).resolve().parents[1]        # titan_data/
DATASET = ROOT / "dataset"
IMAGES_ROOT = DATASET / "images_anonymized"
ANN_ROOT = DATASET / "titan_0_4"                  # One CSV per clip
OUT_ROOT = ROOT / "debug"

TRAIN_SPLIT = DATASET / "train_set.txt"

MAX_FRAMES_TO_DRAW = 20                           # safety limit for visualization
PEDESTRIAN_LABEL = "person"                       # Only keep this class
# --------------------------


def read_clip_list(split_path: Path):
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    clips = [
        ln.strip()
        for ln in split_path.read_text().splitlines()
        if ln.strip()
    ]
    print(f"[INFO] Loaded {len(clips)} clips from {split_path.name}")
    if len(clips) > 0:
        print(f"[INFO] First 5 clips: {clips[:5]}")
    return clips


def pick_first_clip():
    clips = read_clip_list(TRAIN_SPLIT)
    if not clips:
        raise RuntimeError("[ERROR] No clips found in train_set.txt")
    clip = clips[0]
    print(f"[INFO] Using first train clip for verification: {clip}")
    return clip


def load_annotations(clip: str) -> pd.DataFrame:
    csv_path = ANN_ROOT / f"{clip}.csv"
    print(f"[INFO] Looking for annotation CSV at: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"[ERROR] Annotation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    expected_cols = [
        "frames",
        "label",
        "obj_track_id",
        "top",
        "left",
        "height",
        "width",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"[ERROR] Missing expected columns in {csv_path.name}: {missing}\n"
            f"Columns present: {list(df.columns)}"
        )

    print(f"[INFO] Loaded {len(df)} rows from {csv_path.name}")
    print(f"[INFO] Unique labels: {df['label'].unique()}")

    # Show sample rows
    print("[DEBUG] First 5 rows of raw annotations:")
    print(df.head(5).to_string(index=False))

    return df


def filter_pedestrians(df: pd.DataFrame) -> pd.DataFrame:
    pdf = df[df["label"] == PEDESTRIAN_LABEL].copy()
    print(f"[INFO] Filtered to label == '{PEDESTRIAN_LABEL}': {len(pdf)} rows")

    if pdf.empty:
        print("[WARN] No 'person' label rows found in this clip.")
        return pdf

    # Some quick bbox sanity checks
    print("[DEBUG] Bounding box stats for 'person':")
    print(f"  top:    min={pdf['top'].min()},    max={pdf['top'].max()}")
    print(f"  left:   min={pdf['left'].min()},   max={pdf['left'].max()}")
    print(f"  height: min={pdf['height'].min()}, max={pdf['height'].max()}")
    print(f"  width:  min={pdf['width'].min()},  max={pdf['width'].max()}")

    print("[DEBUG] Sample 5 'person' rows:")
    print(
        pdf[["frames", "label", "obj_track_id", "top", "left", "height", "width"]]
        .head(5)
        .to_string(index=False)
    )

    return pdf


def draw_bboxes_for_clip(clip: str, pdf: pd.DataFrame):
    if pdf.empty:
        print("[INFO] Nothing to draw (no person rows).")
        return

    out_dir = OUT_ROOT / f"verify_{clip}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving visualizations to: {out_dir}")

    frames_drawn = 0
    unique_frames = sorted(pdf["frames"].unique())

    print(f"[INFO] Unique frames with persons in this clip: {len(unique_frames)}")
    print(f"[INFO] First 10 such frames: {unique_frames[:10]}")

    # Group by frame ID so all boxes for that frame are drawn together
    for frame_id, group in tqdm(
        pdf.groupby("frames"),
        desc="[INFO] Rendering frames with person boxes"
    ):
        # frame_id is expected to correspond to file name like 000006.png
        frame_str = str(frame_id)              # e.g. "000006.png" or "000006"
        frame_base = frame_str.split(".")[0]   # -> "000006"
        img_name = f"{frame_base}.png"         # -> "000006.png"
        img_path = IMAGES_ROOT / clip / "images" / img_name

        if not img_path.exists():
            print(f"[WARN] Missing image for frame {frame_id}: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        H, W = img.shape[:2]

        # Draw all bboxes for this frame
        for _, row in group.iterrows():
            x = int(row["left"])
            y = int(row["top"])
            w = int(row["width"])
            h = int(row["height"])

            # Clamp box inside image just in case
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img,
                "person",
                (x, max(y - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        out_file = out_dir / img_name
        cv2.imwrite(str(out_file), img)
        frames_drawn += 1

        # Print a few detailed examples then quiet down
        if frames_drawn <= 3:
            print(f"[DEBUG] Drew {len(group)} boxes on {img_name}, saved -> {out_file}")

        if frames_drawn >= MAX_FRAMES_TO_DRAW:
            break

    print(f"[INFO] Total frames rendered with person boxes: {frames_drawn}")
    if frames_drawn == 0:
        print("[WARN] No frames were rendered. Check label filter or paths.")


def main():
    print("========== TITAN Annotation Verification ==========")
    print(f"[INFO] ROOT directory:       {ROOT}")
    print(f"[INFO] DATASET directory:    {DATASET}")
    print(f"[INFO] IMAGES_ROOT:          {IMAGES_ROOT}")
    print(f"[INFO] ANN_ROOT:             {ANN_ROOT}")
    print(f"[INFO] OUT_ROOT:             {OUT_ROOT}")
    print("===================================================")

    clip = pick_first_clip()
    df = load_annotations(clip)
    pdf = filter_pedestrians(df)
    draw_bboxes_for_clip(clip, pdf)

    print("[INFO] Verification script complete.")
    print("[INFO] Open the images in debug/ to visually confirm boxes align with pedestrians.")


if __name__ == "__main__":
    main()
