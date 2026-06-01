#!/usr/bin/env python
"""
Generate a single-frame overlay comparison for TITAN test:
GT vs YOLO11n-full vs YOLO11n-0.1 vs YOLO11L on the SAME image.

Run from the project root (directory that contains 'dataset/' and 'runs/').
"""

from pathlib import Path
import csv
import random

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("dataset")
RUNS_ROOT = Path("runs/detect")

# Models used in the report (edit paths if needed)
MODEL_PATHS = {
    "YOLO11n-full": RUNS_ROOT / "titan_person_11n" / "weights" / "best.pt",
    "YOLO11n-0.1": RUNS_ROOT / "titan_person_11n_0.1_merged" / "weights" / "best.pt",
    # YOLO11L: if you have a local checkpoint path, put it here.
    # If it's in a cache or default ultralytics location, "yolo11l.pt" is fine.
    "YOLO11L": "yolo11l.pt",
}

OUTPUT_FIG = Path("figs/detector_overlay_gt_y11n_y11n01_y11l.png")
IMG_SIZE = 992       # match your eval img size
CONF_THRES = 0.25
IOU_THRES = 0.7
DEVICE = "mps"       # change to "cpu" or "cuda" if needed


# Colors (BGR for cv2 drawing)
COLOR_GT        = (0, 255, 0)     # green
COLOR_Y11N_FULL = (0, 0, 255)     # red
COLOR_Y11N_01   = (255, 0, 0)     # blue
COLOR_Y11L      = (0, 165, 255)   # orange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
import random  # put this at the top of the file

def pick_test_frame():
    """Pick a random clip and a random frame from the test set."""
    test_list = DATASET_ROOT / "test_set.txt"
    with test_list.open() as f:
        clip_ids = [line.strip() for line in f if line.strip()]
    if not clip_ids:
        raise RuntimeError("No clip IDs found in dataset/test_set.txt")

    # random clip from test set
    clip_id = random.choice(clip_ids)

    # adjust this if your frames are not under 'images/'
    img_dir = DATASET_ROOT / "images_anonymized" / clip_id / "images"
    frame_paths = sorted(img_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No PNG frames found in {img_dir}")

    # random frame from that clip
    frame_path = random.choice(frame_paths)

    return clip_id, frame_path

def load_gt_boxes(clip_id, frame_path):
    """
    Load GT person boxes for a given clip and frame.

    Uses dataset/titan_0_4/<clip_id>.csv and filters rows where:
      - label == "person"
      - frames column corresponds to the current frame file name
    Returns a list of (left, top, width, height) in pixel coordinates.
    """
    csv_path = DATASET_ROOT / "titan_0_4" / f"{clip_id}.csv"
    frame_name = frame_path.name

    boxes = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label") != "person":
                continue
            # frames may contain a path like "images/000123.png" or just "000123.png"
            if Path(row["frames"]).name != frame_name:
                continue
            left = float(row["left"])
            top = float(row["top"])
            width = float(row["width"])
            height = float(row["height"])
            boxes.append((left, top, width, height))
    return boxes


def get_pred_boxes(model, frame_path):
    """
    Run a YOLO model on a single frame and return list of predicted boxes.

    Returns a list of (x1, y1, x2, y2) in pixel coordinates.
    """
    results = model.predict(
        source=str(frame_path),
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        workers=0,
        verbose=False,
        classes=[0]
    )
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    # xyxy are in original image coordinates
    xyxy = r.boxes.xyxy.cpu().numpy()
    boxes = []
    for x1, y1, x2, y2 in xyxy:
        boxes.append((float(x1), float(y1), float(x2), float(y2)))
    return boxes


def draw_overlay(img_bgr, gt_boxes, boxes_y11n_full, boxes_y11n_01, boxes_y11l):
    """Draw all boxes on a copy of the image and return RGB image."""
    img = img_bgr.copy()

    # Draw GT (green)
    for (left, top, width, height) in gt_boxes:
        pt1 = (int(left), int(top))
        pt2 = (int(left + width), int(top + height))
        cv2.rectangle(img, pt1, pt2, COLOR_GT, thickness=2)

    # Draw YOLO11n-full (red)
    for (x1, y1, x2, y2) in boxes_y11n_full:
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(img, pt1, pt2, COLOR_Y11N_FULL, thickness=2)

    # Draw YOLO11n-0.1 (blue)
    for (x1, y1, x2, y2) in boxes_y11n_01:
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(img, pt1, pt2, COLOR_Y11N_01, thickness=2)

    # Draw YOLO11L (orange)
    for (x1, y1, x2, y2) in boxes_y11l:
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(img, pt1, pt2, COLOR_Y11L, thickness=2)

    # Convert to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Pick a test frame
    clip_id, frame_path = pick_test_frame()
    print(f"Using clip: {clip_id}")
    print(f"Frame: {frame_path}")

    # Load raw image and GT boxes
    img_bgr = cv2.imread(str(frame_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {frame_path}")
    gt_boxes = load_gt_boxes(clip_id, frame_path)
    print(f"Found {len(gt_boxes)} GT person boxes")

    # Load models
    models = {}
    for name, path in MODEL_PATHS.items():
        print(f"Loading model '{name}' from {path}")
        models[name] = YOLO(str(path))

    # Run models and collect boxes
    print("Running YOLO11n-full...")
    boxes_y11n_full = get_pred_boxes(models["YOLO11n-full"], frame_path)
    print(f"YOLO11n-full predictions: {len(boxes_y11n_full)} boxes")

    print("Running YOLO11n-0.1...")
    boxes_y11n_01 = get_pred_boxes(models["YOLO11n-0.1"], frame_path)
    print(f"YOLO11n-0.1 predictions: {len(boxes_y11n_01)} boxes")

    print("Running YOLO11L...")
    boxes_y11l = get_pred_boxes(models["YOLO11L"], frame_path)
    print(f"YOLO11L predictions: {len(boxes_y11l)} boxes")

    # Draw a single overlay image
    overlay_rgb = draw_overlay(
        img_bgr,
        gt_boxes,
        boxes_y11n_full,
        boxes_y11n_01,
        boxes_y11l,
    )

    # Prepare output directory
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    # Plot with legend
    plt.figure(figsize=(8, 6))
    plt.imshow(overlay_rgb)
    plt.axis("off")
    plt.title(f"TITAN test frame: {clip_id} / {frame_path.name}")

    legend_handles = [
        Patch(color="lime",   label="Ground truth"),
        Patch(color="red",    label="YOLO11n-full"),
        Patch(color="blue",   label="YOLO11n-0.1"),
        Patch(color="orange", label="YOLO11L"),
    ]
    plt.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=200)
    print(f"Saved overlay figure to {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
