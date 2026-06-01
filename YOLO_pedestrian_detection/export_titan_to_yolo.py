import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ================= CONFIG =================
ROOT = Path(__file__).resolve().parents[1]        # titan_data/
DATASET = ROOT / "dataset"
IMAGES_ROOT = DATASET / "images_anonymized"
ANN_ROOT = DATASET / "titan_0_4"

SPLIT_FILES = {
    "train": DATASET / "train_set.txt",
    "val": DATASET / "val_set.txt",
    "test": DATASET / "test_set.txt",
}

YOLO_ROOT = ROOT / "yolo"
TRACKS_ROOT = ROOT / "tracks"

CLASS_NAME = "person"
CLASS_ID = 0  # YOLO class id for 'person'

# If True, script will verify that each image exists and can be read.
STRICT_IMAGE_CHECK = True
# ==========================================


def read_split(split_name: str):
    split_path = SPLIT_FILES[split_name]
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    clips = [
        ln.strip()
        for ln in split_path.read_text().splitlines()
        if ln.strip()
    ]
    print(f"[INFO] {split_name}: loaded {len(clips)} clips from {split_path.name}")
    return clips


def ensure_dirs():
    for split in SPLIT_FILES.keys():
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
        (TRACKS_ROOT / split).mkdir(parents=True, exist_ok=True)


def load_annotations_for_clip(clip: str) -> pd.DataFrame:
    csv_path = ANN_ROOT / f"{clip}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"[ERROR] Missing annotation CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    expected = ["frames", "label", "obj_track_id", "top", "left", "height", "width"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(
            f"[ERROR] {csv_path.name} missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Filter to 'person' only
    df = df[df["label"] == CLASS_NAME].copy()

    return df


def frame_to_filename(frame_value) -> str:
    """
    frames column is already like '000006.png' or '000006'.
    Normalize to '000006.png'.
    """
    s = str(frame_value)
    base = s.split(".")[0]
    return f"{base}.png"


def export_split(split: str):
    clips = read_split(split)

    # images_dir = YOLO_ROOT / "images" / split
    labels_dir = YOLO_ROOT / "labels" / split
    tracks_split_dir = TRACKS_ROOT / split

    total_boxes = 0
    total_images_with_boxes = 0
    total_person_rows = 0

    print(f"[INFO] ===== Exporting split: {split} =====")

    for clip in tqdm(clips, desc=f"[{split}] Clips"):
        try:
            df = load_annotations_for_clip(clip)
        except FileNotFoundError as e:
            print(e)
            continue

        if df.empty:
            # No persons in this clip, skip for detection
            continue

        total_person_rows += len(df)

        # For tracking: write one MOT-style file per clip
        # Format: frame, id, x, y, w, h, conf, -1, -1, -1
        # Using 1-based frame index for MOT; here we derive from filename.
        track_file = tracks_split_dir / f"{clip}.txt"
        with track_file.open("w") as tf:
            # Group by frame so we can both write YOLO labels and MOT lines
            for frame_value, g in df.groupby("frames"):
                img_name = frame_to_filename(frame_value)
                img_rel = f"{clip}/images/{img_name}"
                img_path = IMAGES_ROOT / img_rel

                if not img_path.exists():
                    print(f"[WARN] Missing image {img_path}, skipping frame.")
                    continue

                img = cv2.imread(str(img_path)) if STRICT_IMAGE_CHECK else None
                if STRICT_IMAGE_CHECK and img is None:
                    print(f"[WARN] Failed to read {img_path}, skipping frame.")
                    continue

                if img is not None:
                    H, W = img.shape[:2]
                else:
                    # If STRICT_IMAGE_CHECK is False, we would need a different way
                    raise RuntimeError(
                        "Image size unknown and STRICT_IMAGE_CHECK is False; "
                        "enable it or add image-size metadata."
                    )

                # YOLO label file path (same name as image, .txt extension)
                yolo_label_path = labels_dir / f"{clip}_{img_name.replace('.png', '.txt')}"
                has_box_for_image = False

                # Convert frame to numeric index for MOT
                frame_idx = int(img_name.split(".")[0])  # e.g. '000006' -> 6

                for _, row in g.iterrows():
                    x = float(row["left"])
                    y = float(row["top"])
                    w = float(row["width"])
                    h = float(row["height"])
                    track_id = int(row["obj_track_id"])

                    # Clamp bbox inside image
                    if w <= 0 or h <= 0:
                        continue
                    if x >= W or y >= H:
                        continue

                    x = max(0.0, min(x, W - 1.0))
                    y = max(0.0, min(y, H - 1.0))
                    w = max(1.0, min(w, W - x))
                    h = max(1.0, min(h, H - y))

                    # ---- YOLO format (normalized cx, cy, w, h) ----
                    cx = (x + w / 2.0) / W
                    cy = (y + h / 2.0) / H
                    nw = w / W
                    nh = h / H

                    # Safety: ensure in [0,1]
                    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                        continue
                    if nw <= 0 or nh <= 0:
                        continue

                    with yolo_label_path.open("a") as lf:
                        lf.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                    has_box_for_image = True
                    total_boxes += 1

                    # ---- MOT-style line using original pixel coords ----
                    # frame, id, x, y, w, h, conf, -1, -1, -1
                    tf.write(f"{frame_idx},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

                # If at least one box for this image, record the corresponding image path
                if has_box_for_image:
                    total_images_with_boxes += 1

                    # Optionally: write a simple txt list of image paths for training frameworks
                    # Many YOLO trainers instead glob paths; we can rely on that.
                    # Use a deterministic name: copy/symlink is optional & separate.

        # Small debug summary per clip (only if it had persons)
        if not df.empty:
            print(
                f"[DEBUG] {split} / {clip}: {len(df)} person rows -> "
                f"MOT: {track_file.name}"
            )

    print(f"[INFO] ===== Done {split} =====")
    print(f"[INFO] Total 'person' rows:           {total_person_rows}")
    print(f"[INFO] Total YOLO boxes written:      {total_boxes}")
    print(f"[INFO] Total images with boxes:       {total_images_with_boxes}")


def main():
    print("========== TITAN → YOLO + MOT Export ==========")
    print(f"[INFO] ROOT:       {ROOT}")
    print(f"[INFO] DATASET:    {DATASET}")
    print(f"[INFO] IMAGES:     {IMAGES_ROOT}")
    print(f"[INFO] ANN:        {ANN_ROOT}")
    print(f"[INFO] YOLO_ROOT:  {YOLO_ROOT}")
    print(f"[INFO] TRACKS_ROOT:{TRACKS_ROOT}")
    print("===============================================")

    ensure_dirs()

    for split in ["train", "val", "test"]:
        export_split(split)

    print("[INFO] Export complete.")
    print("[INFO] Next: point your detector to yolo/images/* and yolo/labels/*.")


if __name__ == "__main__":
    main()
