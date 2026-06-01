#!/usr/bin/env python3
"""
Generate three videos of person tracks for the data preprocessing slide:

1) tracks_original.mp4  - original crops (no resize, no normalization)
2) tracks_resized_112.mp4  - crops resized to 112x112
3) tracks_normalized_112.mp4  - resized + normalized with Kinetics stats,
   then rescaled to [0, 255] for visualization.

Each video shows several 16-frame person tubes stitched sequentially.
"""

import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# -------------------------------------------------------------------
# PATHS / CONSTANTS (adapt if needed)
# -------------------------------------------------------------------
CLIP_PATH_TEMPLATE = "dataset/images_anonymized/{}/images/"
LABEL_PATH_TEMPLATE = "dataset/titan_0_4/{}.csv"
TRAIN_CLIPS_PATH = "dataset/train_set.txt"

OUTPUT_DIR = Path("presentation_outputs/preprocessing_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ORIG_PATH = OUTPUT_DIR / "tracks_original.mp4"
OUT_RESIZED_PATH = OUTPUT_DIR / "tracks_resized_112.mp4"
OUT_NORM_PATH = OUTPUT_DIR / "tracks_normalized_112.mp4"

FPS = 10
T = 16            # tube length
N_TUBES = 40       # how many tubes you want to show

PERSON_LABEL = "person"

KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def load_clip_ids(n_clips: int = 20):
    """Read some clip IDs from train_set.txt (e.g., 'clip_306')."""
    clip_ids = []
    with open(TRAIN_CLIPS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            clip_ids.append(line)
            if len(clip_ids) >= n_clips:
                break
    if not clip_ids:
        raise RuntimeError(f"No clip IDs found in {TRAIN_CLIPS_PATH}")
    return clip_ids


def load_person_tracks_for_clip(clip_id: str):
    """
    Returns a dict: track_id -> list of (frame_name, bbox)
    bbox is (left, top, width, height), all floats.
    """
    csv_path = Path(LABEL_PATH_TEMPLATE.format(clip_id))
    if not csv_path.exists():
        print(f"[WARN] missing CSV for {clip_id}: {csv_path}")
        return {}

    track_to_frames = defaultdict(list)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label", "").strip() != PERSON_LABEL:
                continue

            frame_name = Path(row["frames"]).name
            raw_id = row["obj_track_id"].strip()
            try:
                track_id = int(raw_id)
            except ValueError:
                track_id = int(float(raw_id))  # handle "1.0"

            top = float(row["top"])
            left = float(row["left"])
            height = float(row["height"])
            width = float(row["width"])
            bbox = (left, top, width, height)

            track_to_frames[track_id].append((frame_name, bbox))

    # sort frames for each track by frame_name
    for tid in track_to_frames:
        track_to_frames[tid].sort(key=lambda x: x[0])

    return track_to_frames


def normalize_for_visualization(frame_bgr_112):
    """
    Apply Kinetics normalization to a BGR uint8 frame of size 112x112,
    then re-scale to [0,255] for visualization.
    """
    # BGR -> RGB, [0,1]
    rgb = cv2.cvtColor(frame_bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - KINETICS_MEAN) / KINETICS_STD  # normalized

    # rescale to [0, 1] for display
    vmin = rgb.min()
    vmax = rgb.max()
    rgb_vis = (rgb - vmin) / (vmax - vmin + 1e-8)
    rgb_vis = np.clip(rgb_vis * 255.0, 0, 255).astype(np.uint8)

    # back to BGR for OpenCV VideoWriter
    bgr_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
    return bgr_vis


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    clip_ids = load_clip_ids(n_clips=50)

    tubes_orig = []   # list of list of original crops
    tubes_resz = []   # list of list of 112x112
    tubes_norm = []   # list of list of normalized visualizations

    max_h = 0
    max_w = 0

    # ---------------------------------------------------------------
    # 1) Collect T-frame tubes from various clips/track_ids
    # ---------------------------------------------------------------
    for clip_id in clip_ids:
        images_dir = Path(CLIP_PATH_TEMPLATE.format(clip_id))
        if not images_dir.exists():
            print(f"[WARN] images dir missing for {clip_id}: {images_dir}")
            continue

        track_to_frames = load_person_tracks_for_clip(clip_id)
        if not track_to_frames:
            continue

        # map frame_name -> full path once for efficiency
        frame_paths = {p.name: p for p in images_dir.iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}

        for track_id, frames in track_to_frames.items():
            if len(frames) < T:
                continue

            # take the first contiguous T frames
            frames_slice = frames[:T]
            tube_orig_frames = []
            tube_resz_frames = []
            tube_norm_frames = []

            for frame_name, bbox in frames_slice:
                if frame_name not in frame_paths:
                    continue
                img = cv2.imread(str(frame_paths[frame_name]))
                if img is None:
                    continue

                h, w = img.shape[:2]
                left, top, width, height = bbox
                x1 = int(max(0, min(w - 1, left)))
                y1 = int(max(0, min(h - 1, top)))
                x2 = int(max(0, min(w, left + width)))
                y2 = int(max(0, min(h, top + height)))

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                ch, cw = crop.shape[:2]
                max_h = max(max_h, ch)
                max_w = max(max_w, cw)

                # resized & normalized versions
                resz = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
                norm_vis = normalize_for_visualization(resz)

                tube_orig_frames.append(crop)
                tube_resz_frames.append(resz)
                tube_norm_frames.append(norm_vis)

            if len(tube_orig_frames) == T:
                tubes_orig.append(tube_orig_frames)
                tubes_resz.append(tube_resz_frames)
                tubes_norm.append(tube_norm_frames)

            if len(tubes_orig) >= N_TUBES:
                break
        if len(tubes_orig) >= N_TUBES:
            break

    if not tubes_orig:
        raise RuntimeError("No tubes found; check paths and annotations.")

    print(f"[INFO] Collected {len(tubes_orig)} tubes of length {T}.")
    print(f"[INFO] Original crop max size: {max_w}x{max_h}")

    # ---------------------------------------------------------------
    # 2) Prepare VideoWriters
    # ---------------------------------------------------------------
    orig_size = (max_w, max_h)         # (width, height)
    resz_size = (112, 112)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_orig = cv2.VideoWriter(str(OUT_ORIG_PATH), fourcc, FPS, orig_size)
    vw_resz = cv2.VideoWriter(str(OUT_RESIZED_PATH), fourcc, FPS, resz_size)
    vw_norm = cv2.VideoWriter(str(OUT_NORM_PATH), fourcc, FPS, resz_size)

    # ---------------------------------------------------------------
    # 3) Write frames: tubes concatenated one after another
    # ---------------------------------------------------------------
    def pad_to_size(img, size):
        """Pad img to (H, W) with black borders, without resizing."""
        target_w, target_h = size
        h, w = img.shape[:2]
        top = (target_h - h) // 2
        bottom = target_h - h - top
        left = (target_w - w) // 2
        right = target_w - w - left
        return cv2.copyMakeBorder(img, top, bottom, left, right,
                                  borderType=cv2.BORDER_CONSTANT,
                                  value=(255, 255, 255))

    for t_idx in range(len(tubes_orig)):
        for f_idx in range(T):
            orig = tubes_orig[t_idx][f_idx]
            resz = tubes_resz[t_idx][f_idx]
            norm = tubes_norm[t_idx][f_idx]

            orig_padded = pad_to_size(orig, orig_size)

            vw_orig.write(orig_padded)
            vw_resz.write(resz)
            vw_norm.write(norm)


    vw_orig.release()
    vw_resz.release()
    vw_norm.release()

    print(f"[OK] Wrote {OUT_ORIG_PATH}")
    print(f"[OK] Wrote {OUT_RESIZED_PATH}")
    print(f"[OK] Wrote {OUT_NORM_PATH}")


if __name__ == "__main__":
    main()
