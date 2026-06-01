#!/usr/bin/env python3
"""
Generate a stitched TITAN video for the presentation:

- Uses the first 10 clip IDs from dataset/train_set.txt (20% train split)
- For each frame:
    * draws person bounding boxes + track IDs on the left
    * shows a white panel on the right listing, for each VISIBLE ID:
        - atomic
        - simple-context
        - complex-context
        - communicative
        - transportive

Output:
    presentation_outputs/titan_videos_with_labels/titan_10clips_person_labels_with_side_panel.mp4
"""

import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# -------------------------------------------------------------
# CONSTANTS FROM YOUR SETUP
# -------------------------------------------------------------
CLIP_PATH_TEMPLATE = "dataset/images_anonymized/{}/images/"
LABEL_PATH_TEMPLATE = "dataset/titan_0_4/{}.csv"
TRAIN_CLIPS_PATH = "dataset/train_set.txt"  # first 20% of rows

PERSON_LABEL = "person"
NONE_LABEL = "none of the above"

OUTPUT_DIR = Path("presentation_outputs/titan_videos_with_labels/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "titan_10clips_person_labels_with_side_panel.mp4"

FPS = 10  # 10 Hz for presentation video

# Map CSV columns to friendly head names
HEAD_MAP = {
    "attributes.Atomic Actions": "atomic",
    "attributes.Simple Context": "simple-context",
    "attributes.Complex Contextual": "complex-context",
    "attributes.Communicative": "communicative",
    "attributes.Transporting": "transportive",
}
HEADS_IN_ORDER = list(HEAD_MAP.values())


# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def load_clip_ids(n_clips: int = 10):
    """Read the first n_clips lines from train_set.txt (e.g., 'clip_306')."""
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


def load_person_annotations(clip_id: str):
    """
    Load annotations for a single clip.

    Returns
    -------
    frame_to_anns : dict[str, list[dict]]
        frame_name -> list of {track_id, bbox}
    id_to_labels_by_head : dict[int, dict[str, str]]
        track_id -> {head_name -> label or 'none'}
    track_ids : list[int]
        sorted list of track IDs in this clip
    """
    csv_path = Path(LABEL_PATH_TEMPLATE.format(clip_id))
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    frame_to_anns = defaultdict(list)
    track_ids = set()
    id_to_labels_by_head: dict[int, dict[str, str]] = defaultdict(dict)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label", "").strip() != PERSON_LABEL:
                continue

            frame_name = Path(row["frames"]).name  # e.g. '000012.png'

            raw_id = row["obj_track_id"].strip()
            try:
                track_id = int(raw_id)
            except ValueError:
                track_id = int(float(raw_id))  # handles "1.0"

            top = float(row["top"])
            left = float(row["left"])
            height = float(row["height"])
            width = float(row["width"])

            frame_to_anns[frame_name].append(
                {"track_id": track_id, "bbox": (left, top, width, height)}
            )
            track_ids.add(track_id)

            # record per-head labels, keeping first non-none we see
            for col, head_name in HEAD_MAP.items():
                if col not in row:
                    continue
                val = row[col].strip()
                if val and val.lower() != NONE_LABEL.lower():
                    if head_name not in id_to_labels_by_head[track_id]:
                        id_to_labels_by_head[track_id][head_name] = val

    # fill 'none' where we have no label for that head
    for tid in track_ids:
        if tid not in id_to_labels_by_head:
            id_to_labels_by_head[tid] = {}
        for head_name in HEADS_IN_ORDER:
            if head_name not in id_to_labels_by_head[tid]:
                id_to_labels_by_head[tid][head_name] = "none"

    return frame_to_anns, id_to_labels_by_head, sorted(track_ids)


def color_for_track(track_id: int):
    """Deterministic bright color for each track ID (BGR)."""
    rng = np.random.RandomState(track_id * 17 + 3)
    return tuple(int(c) for c in rng.randint(50, 255, size=3))


def draw_boxes_with_ids(frame, anns_for_frame):
    """
    Draw bounding boxes and IDs on the frame.

    Parameters
    ----------
    frame : np.ndarray
    anns_for_frame : list[dict]
    """
    h, w = frame.shape[:2]
    for ann in anns_for_frame:
        tid = ann["track_id"]
        left, top, width, height = ann["bbox"]

        x1 = int(left)
        y1 = int(top)
        x2 = int(left + width)
        y2 = int(top + height)

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        color = color_for_track(tid)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {tid}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame

def wrap_text(text, max_width_px, font, font_scale, thickness):
    """
    Split `text` into multiple lines so that each line is <= max_width_px.
    Returns a list of lines.
    """
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]

    for w in words[1:]:
        test = current + " " + w
        (test_w, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if test_w <= max_width_px:
            current = test
        else:
            lines.append(current)
            current = w

    lines.append(current)
    return lines


def make_label_panel(height, clip_id, visible_ids, id_to_labels_by_head):
    panel_w = 520   # or whatever you chose
    panel = np.full((height, panel_w, 3), 255, dtype=np.uint8)

    x0 = 24
    y = 44
    dy_id = 38
    dy_label = 30

    # Title
    cv2.putText(
        panel,
        f"{clip_id}",
        (x0, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    y += int(1.8 * dy_id)

    font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = 0.8
    label_thickness = 2
    max_label_width = panel_w - (x0 + 28) - 20  # margin on the right

    for tid in visible_ids:
        labels_by_head = id_to_labels_by_head.get(tid, {})

        # collect non-none labels
        labels = []
        for head_name in HEADS_IN_ORDER:
            lab = labels_by_head.get(head_name)
            if lab and lab.lower() != 'none':
                if '(' in lab:
                    lab = lab.split('(')[0][:-1]
                labels.append(lab)
                
        if not labels:
            labels = ["no annotated actions"]

        # ID line
        color = color_for_track(tid)
        cv2.rectangle(panel, (x0 - 4, y - 24), (x0 + 18, y - 4), color, -1)
        cv2.putText(
            panel,
            f"ID {tid}",
            (x0 + 28, y),
            font,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += dy_id

        # wrapped labels
        for lab in labels:
            lines = wrap_text(lab, max_label_width, font, label_scale, label_thickness)
            for line in lines:
                cv2.putText(
                    panel,
                    line,
                    (x0 + 28, y),
                    font,
                    label_scale,
                    (60, 60, 60),
                    label_thickness,
                    cv2.LINE_AA,
                )
                y += dy_label

        y += dy_label  # extra gap between IDs

        if y > height - 40:
            break

    return panel


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    clip_ids = load_clip_ids(n_clips=5)
    print(f"[INFO] Using clip IDs: {clip_ids}")

    writer = None
    out_size = None

    for clip_id in clip_ids:
        images_dir = Path(CLIP_PATH_TEMPLATE.format(clip_id))
        if not images_dir.exists():
            print(f"[WARN] images dir missing for clip {clip_id}: {images_dir}")
            continue

        frame_to_anns, id_to_labels_by_head, track_ids = load_person_annotations(
            clip_id
        )
        print(
            f"[INFO] Clip {clip_id}: {len(track_ids)} person tracks, "
            f"{len(frame_to_anns)} frames with persons"
        )

        image_paths = sorted(
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not image_paths:
            print(f"[WARN] no images found in {images_dir}")
            continue

        # Initialize writer using first frame of the first clip
        if writer is None:
            sample = cv2.imread(str(image_paths[0]))
            if sample is None:
                raise RuntimeError(f"Could not read {image_paths[0]}")
            h, w = sample.shape[:2]
            # temporary panel to get width
            dummy_panel = make_label_panel(h, clip_id, [], id_to_labels_by_head)
            out_w = w + dummy_panel.shape[1]
            out_h = h
            out_size = (out_w, out_h)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, out_size)
            print(
                f"[INFO] Video writer opened at {OUTPUT_PATH} ({out_w}x{out_h} @ {FPS}fps)"
            )

        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"[WARN] could not read {img_path}")
                continue

            frame_name = img_path.name
            anns = frame_to_anns.get(frame_name, [])

            # draw boxes + IDs
            frame = draw_boxes_with_ids(frame, anns)

            # which IDs are currently visible in this frame?
            visible_ids = sorted({ann["track_id"] for ann in anns})

            # build panel dynamically for current frame
            h, w = frame.shape[:2]
            panel = make_label_panel(h, clip_id, visible_ids, id_to_labels_by_head)

            # concat horizontally
            frame_out = np.concatenate([frame, panel], axis=1)
            writer.write(frame_out)

    if writer is not None:
        writer.release()
        print(f"[OK] Wrote stitched video to {OUTPUT_PATH}")
    else:
        print("[ERROR] No frames were written; check paths and file names.")


if __name__ == "__main__":
    main()
