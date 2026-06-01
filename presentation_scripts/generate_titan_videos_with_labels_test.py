import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights

# ----------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------
CLIP_PATH_TEMPLATE = "dataset/images_anonymized/{}/images/"
LABEL_PATH_TEMPLATE = "dataset/titan_0_4/{}.csv"
TEST_CLIPS_PATH = "dataset/test_set.txt"

MODEL_PATH = (
    "action_det/runs/r3d18_pct20_T16_S4_img112_b24_rare2.5_no_s/best.pt"
)

SPLIT_DIR = Path("action_det/data/splits_pct_20_T16_S4")
LABEL_SPACE_JSON = SPLIT_DIR / "label_space.json"
meta = json.loads(LABEL_SPACE_JSON.read_text())
LABEL_SPACE: Dict[str, List[str]] = meta["label_space"]

OUTPUT_DIR = Path("presentation_outputs/titan_videos_with_labels_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO = OUTPUT_DIR / "titan_test_predictions.mp4"

FPS = 10
TUBE_LEN = 16
RESIZE_HW = 112

KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

# ---- Ground-truth label parsing (same convention as your GT script) ----
NONE_LABEL = "none of the above"
HEAD_MAP = {
    "attributes.Atomic Actions": "atomic",
    "attributes.Simple Context": "simple-context",
    "attributes.Complex Contextual": "complex-context",
    "attributes.Communicative": "communicative",
    "attributes.Transporting": "transportive",
}

HEAD_ORDER = list(HEAD_MAP.values())
HEADS_IN_ORDER = HEAD_ORDER  # for reuse of the original panel code


# ----------------------------------------------------------------------
# Model definition & loading  (UNCHANGED)
# ----------------------------------------------------------------------
def build_model(label_space):
    backbone = r3d_18(weights="KINETICS400_V1")  # torchvision >= 0.15
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    hidden_dim = 256
    heads = nn.ModuleDict({
        cat: nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, len(labels)),
        )
        for cat, labels in label_space.items()
    })

    class MultiHead(nn.Module):
        def __init__(self, bb, hd):
            super().__init__()
            self.bb = bb
            self.hd = hd
        def forward(self, x):
            f = self.bb(x)  # (N, feat)
            return {k: self.hd[k](f) for k in self.hd.keys()}

    return MultiHead(backbone, heads)


def load_trained_model() -> nn.Module:
    model = build_model(LABEL_SPACE)
    state = torch.load(MODEL_PATH, map_location="cpu")

    # Handle common checkpoint formats
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # If keys are prefixed (e.g., "model."), try to strip once
    sample_key = next(iter(state.keys()))
    if sample_key.startswith("model."):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ----------------------------------------------------------------------
# Data loading helpers
# ----------------------------------------------------------------------
def read_split_file(path: str, max_clips: int = 6) -> List[str]:
    clip_ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # lines can be "clip_306" or "clip_306,something"
            clip_id = line.split(",")[0].strip()
            if clip_id:
                clip_ids.append(clip_id)
            if len(clip_ids) >= max_clips:
                break
    return clip_ids


def load_person_annotations(
    clip_id: str,
) -> Tuple[
    Dict[str, List[dict]],
    Dict[int, Dict[str, Tuple[int, int, int, int]]],
    Dict[int, Dict[str, str]],
]:
    """
    Returns:
      frame_to_anns: frame_name -> list of {track_id, bbox}
      track_to_frames: track_id -> {frame_name -> bbox}
      gt_labels_by_track: track_id -> {head -> gt_label}
    """
    csv_path = Path(LABEL_PATH_TEMPLATE.format(clip_id))
    frame_to_anns: Dict[str, List[dict]] = {}
    track_to_frames: Dict[int, Dict[str, Tuple[int, int, int, int]]] = {}
    gt_labels_by_track: Dict[int, Dict[str, str]] = {}

    if not csv_path.exists():
        print(f"[WARN] Missing CSV for clip {clip_id}: {csv_path}")
        return frame_to_anns, track_to_frames, gt_labels_by_track

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # keep only person tracks
            if row.get("label", "").strip() != "person":
                continue

            # frame name
            frame_name = row["frames"].split("/")[-1]

            # track id (robust to "1.0" style)
            raw_id = row["obj_track_id"].strip()
            try:
                track_id = int(raw_id)
            except ValueError:
                track_id = int(float(raw_id))

            left = int(float(row["left"]))
            top = int(float(row["top"]))
            width = int(float(row["width"]))
            height = int(float(row["height"]))
            bbox = (left, top, width, height)

            ann = {"track_id": track_id, "bbox": bbox}
            frame_to_anns.setdefault(frame_name, []).append(ann)
            track_to_frames.setdefault(track_id, {})[frame_name] = bbox

            # per-head GT labels: keep the first non-"none of the above" seen
            track_gt = gt_labels_by_track.setdefault(track_id, {})
            for col, head_name in HEAD_MAP.items():
                if col not in row:
                    continue
                val = row[col].strip()
                if not val:
                    continue
                if val.lower() == NONE_LABEL.lower():
                    continue
                if head_name not in track_gt:
                    track_gt[head_name] = val

    # fill in "none" where we never saw a non-none label
    for tid, head_dict in gt_labels_by_track.items():
        for head_name in HEADS_IN_ORDER:
            if head_name not in head_dict:
                head_dict[head_name] = "none"

    return frame_to_anns, track_to_frames, gt_labels_by_track


# ----------------------------------------------------------------------
# Utility: build a tube, run model, get predictions  (UNCHANGED)
# ----------------------------------------------------------------------
def build_tube_for_track(
    images_dir: Path,
    track_id: int,
    frame_to_bbox: Dict[str, Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Returns a (T, H, W, 3) uint8 clip for the given track,
    using a TUBE_LEN window centered on the track.
    """
    frame_names = sorted(frame_to_bbox.keys())
    if not frame_names:
        raise ValueError("Track has no frames")

    num_frames = len(frame_names)
    if num_frames >= TUBE_LEN:
        mid = num_frames // 2
        start = max(0, mid - TUBE_LEN // 2)
        end = start + TUBE_LEN
        frame_seq = frame_names[start:end]
    else:
        frame_seq = frame_names + [frame_names[-1]] * (TUBE_LEN - num_frames)

    clip_imgs: List[np.ndarray] = []

    last_valid = None
    for fname in frame_seq:
        img_path = images_dir / fname
        img = cv2.imread(str(img_path))
        if img is None:
            # fall back to last valid frame if possible
            if last_valid is not None:
                img = last_valid.copy()
            else:
                continue

        bbox = frame_to_bbox.get(fname)
        if bbox is None:
            # use last bbox if missing
            bbox = frame_to_bbox[frame_names[-1]]

        x, y, w, h = bbox
        x2, y2 = x + w, y + h
        x, y = max(0, x), max(0, y)
        crop = img[y:y2, x:x2]
        if crop.size == 0:
            crop = img  # degenerate box; just use full frame

        crop = cv2.resize(crop, (RESIZE_HW, RESIZE_HW))
        clip_imgs.append(crop)
        last_valid = crop

    if not clip_imgs:
        raise ValueError("Could not build tube for track")

    while len(clip_imgs) < TUBE_LEN:
        clip_imgs.append(clip_imgs[-1])

    clip = np.stack(clip_imgs, axis=0)  # (T, H, W, 3)
    return clip


def tube_to_tensor(clip: np.ndarray) -> torch.Tensor:
    """
    clip: (T, H, W, 3) uint8 in [0,255]
    -> tensor (1, 3, T, H, W) normalized with Kinetics stats
    """
    clip = clip.astype(np.float32) / 255.0
    mean = np.array(KINETICS_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(KINETICS_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    clip = (clip - mean) / std  # (T, H, W, 3)

    clip = np.transpose(clip, (3, 0, 1, 2))  # (3, T, H, W)
    tensor = torch.from_numpy(clip).unsqueeze(0)  # (1, 3, T, H, W)
    return tensor


def get_predictions_for_clip(
    model: nn.Module,
    clip_id: str,
    frame_to_anns: Dict[str, List[dict]],
    track_to_frames: Dict[int, Dict[str, Tuple[int, int, int, int]]],
) -> Dict[int, Dict[str, str]]:
    """
    Returns: track_id -> {head -> predicted_label}
    """
    images_dir = Path(CLIP_PATH_TEMPLATE.format(clip_id))
    preds: Dict[int, Dict[str, str]] = {}

    with torch.no_grad():
        for track_id, frame_dict in track_to_frames.items():
            try:
                tube_np = build_tube_for_track(images_dir, track_id, frame_dict)
            except Exception as e:
                print(f"[WARN] Skipping track {track_id} in {clip_id}: {e}")
                continue

            x = tube_to_tensor(tube_np)
            outputs = model(x)  # dict head -> (1, C)

            head_preds: Dict[str, str] = {}
            for head, logits in outputs.items():
                probs = F.softmax(logits, dim=1)[0]
                c_idx = int(torch.argmax(probs))
                label = LABEL_SPACE[head][c_idx]
                head_preds[head] = label

            preds[track_id] = head_preds

    return preds


# ----------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------
def color_for_track(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.RandomState(track_id * 9973 + 123)
    r, g, b = rng.randint(60, 255, size=3)
    return int(b), int(g), int(r)  # BGR for cv2


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


def _is_none_like(lab: str) -> bool:
    if lab is None:
        return True
    s = lab.strip().lower()
    return s == "" or s == "none" or s == NONE_LABEL.lower()


def _clean_label_text(lab: str) -> str:
    if lab is None:
        return ""
    lab = lab.strip()
    if "(" in lab:
        lab = lab.split("(")[0].rstrip()
    return lab


def make_label_panel(
    height: int,
    clip_id: str,
    visible_ids: List[int],
    id_to_pred_by_head: Dict[int, Dict[str, str]],
    id_to_gt_by_head: Dict[int, Dict[str, str]],
):
    panel_w = 520
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
        pred_by_head = id_to_pred_by_head.get(tid, {})
        gt_by_head = id_to_gt_by_head.get(tid, {})

        texts = []
        for head_name in HEADS_IN_ORDER:
            pred_lab = pred_by_head.get(head_name, "none")
            gt_lab = gt_by_head.get(head_name, "none")

            has_pred = not _is_none_like(pred_lab)
            has_gt = not _is_none_like(gt_lab)

            if not (has_pred or has_gt):
                continue

            pred_clean = _clean_label_text(pred_lab)
            gt_clean = _clean_label_text(gt_lab)

            txt = ""
            if has_gt or has_pred:
                if not has_pred:
                    txt = f"None (GT: {gt_clean})"
                elif not has_gt:
                    txt = f"{pred_clean} (GT: None)"
                else:
                    txt = f"{pred_clean} (GT: {gt_clean})"
            if txt:
                texts.append(txt)

        if not texts:
            texts = ["no annotated actions"]

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
        for lab in texts:
            lines = wrap_text(
                lab,
                max_label_width,
                font,
                label_scale,
                label_thickness,
            )
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


def draw_boxes_with_ids(
    frame: np.ndarray, anns: List[dict]
) -> Tuple[np.ndarray, List[int]]:
    """
    Draws boxes & IDs on a frame. Returns the frame and list of visible IDs.
    """
    out = frame.copy()
    visible_ids: List[int] = []

    for ann in anns:
        tid = ann["track_id"]
        x, y, w, h = ann["bbox"]
        x2, y2 = x + w, y + h
        color = color_for_track(tid)

        cv2.rectangle(out, (x, y), (x2, y2), color, 2)
        label = f"ID {tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, y - th - 4), (x + tw + 4, y), color, -1)
        cv2.putText(
            out,
            label,
            (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        visible_ids.append(tid)

    return out, visible_ids


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    model = load_trained_model()
    print("[INFO] Loaded model")

    clip_ids = read_split_file(TEST_CLIPS_PATH, max_clips=2)
    print(f"[INFO] Using test clips: {clip_ids}")

    writer = None
    out_size = None

    for clip_id in clip_ids:
        images_dir = Path(CLIP_PATH_TEMPLATE.format(clip_id))
        if not images_dir.exists():
            print(f"[WARN] Missing images for clip {clip_id}: {images_dir}")
            continue

        frame_to_anns, track_to_frames, gt_labels_by_track = load_person_annotations(
            clip_id
        )
        if not frame_to_anns:
            print(f"[WARN] No person annotations for clip {clip_id}")
            continue

        preds_by_track = get_predictions_for_clip(
            model, clip_id, frame_to_anns, track_to_frames
        )
        print(
            f"[INFO] Clip {clip_id}: {len(track_to_frames)} tracks, "
            f"{len(preds_by_track)} with predictions"
        )

        frame_files = sorted(frame_to_anns.keys())
        for frame_name in frame_files:
            img_path = images_dir / frame_name
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            anns = frame_to_anns.get(frame_name, [])
            frame_with_boxes, visible_ids = draw_boxes_with_ids(frame, anns)
            panel = make_label_panel(
                frame.shape[0],
                clip_id,
                visible_ids,
                preds_by_track,
                gt_labels_by_track,
            )

            combined = np.hstack([frame_with_boxes, panel])

            if writer is None:
                h, w, _ = combined.shape
                out_size = (w, h)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, FPS, out_size)
                print(f"[INFO] Writing video to {OUTPUT_VIDEO}")

            writer.write(combined)

    if writer is not None:
        writer.release()
        print("[INFO] Done.")
    else:
        print("[WARN] No video written (no valid clips?).")


if __name__ == "__main__":
    main()
