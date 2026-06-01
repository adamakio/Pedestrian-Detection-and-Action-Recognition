#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a single qualitative visualization frame for the action recognizer.

For one chosen test frame, we:
  - find all tubes ending at that frame (one per person),
  - compute a crop region that contains all bounding boxes while maintaining
    the original aspect ratio,
  - draw differently coloured bounding boxes for each person on the cropped
    image, and
  - add a colour-coded, *wrapped* legend on the right with GT and Pred labels
    for each non-"none" head.

Assumes that evaluation has already been run and that
test_logits_targets_cache.npz exists under the run's eval directory.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _dataset import CATS  # fixed order of heads


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_xywh_per_frame(item):
    """Ensure list of [x, y, w, h] per frame."""
    T = len(item.get("frames", []))

    def cxcywh_to_xywh(b):
        cx, cy, w, h = map(float, b)
        return [cx - w / 2.0, cy - h / 2.0, w, h]

    if "bboxes_xywh" in item:
        return [list(map(float, b)) for b in item["bboxes_xywh"]]
    if "bboxes_cxcywh" in item:
        return [cxcywh_to_xywh(b) for b in item["bboxes_cxcywh"]]
    if "bbox_xywh" in item:
        return [list(map(float, item["bbox_xywh"]))] * T
    if "bbox" in item:
        return [list(map(float, item["bbox"]))] * T
    if "bbox_cxcywh" in item:
        return [cxcywh_to_xywh(item["bbox_cxcywh"])] * T
    for k in ("boxes", "bboxes"):
        if k in item and len(item[k]) and len(item[k][0]) == 4:
            return [list(map(float, b)) for b in item[k]]
    raise KeyError("No bbox-like field for visualization.")


def softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def line_height(font: ImageFont.ImageFont) -> int:
    """Estimate single-line height for layout."""
    if hasattr(font, "getbbox"):
        x0, y0, x1, y1 = font.getbbox("Ag")
        return y1 - y0
    elif hasattr(font, "getsize"):
        _, h = font.getsize("Ag")
        return h
    else:
        return 20


def text_width(text: str, font: ImageFont.ImageFont) -> int:
    """Measure text width with the font."""
    if hasattr(font, "getbbox"):
        x0, y0, x1, y1 = font.getbbox(text)
        return x1 - x0
    elif hasattr(font, "getsize"):
        w, _ = font.getsize(text)
        return w
    else:
        return len(text) * 10


def wrap_text_line(text: str, font: ImageFont.ImageFont, max_width: int):
    """
    Wrap a single logical line into multiple lines so that each fits
    within max_width. Simple word-based wrapping.
    """
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for w in words[1:]:
        candidate = current + " " + w
        if text_width(candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    args = ap.parse_args()

    pct_int = int(args.pct * 100)
    split_dir = Path(f"action_det/data/splits_pct_{pct_int}_T{args.T}_S{args.stride}")
    eval_dir = Path(
        f"action_det/eval/r3d18_pct{pct_int}_T{args.T}_S{args.stride}_img{args.imgsz}_b{args.batch}_rare_boost"
    )

    test_jsonl = split_dir / "test.jsonl"
    label_space_json = split_dir / "label_space.json"
    cache_npz = eval_dir / "test_logits_targets_cache.npz"

    assert test_jsonl.exists(), f"Missing test index: {test_jsonl}"
    assert label_space_json.exists(), f"Missing label space: {label_space_json}"
    assert cache_npz.exists(), f"Missing logits cache: {cache_npz} (run eval script first)"

    out_vis_dir = eval_dir / "vis_frames"
    out_vis_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_vis_dir / "sample_multi_gt_vs_pred_legend.jpg"

    # Load label space
    meta = json.loads(label_space_json.read_text())
    label_space = meta["label_space"]

    # Load test items
    items = [json.loads(l) for l in test_jsonl.read_text().splitlines()]
    N = len(items)
    if N == 0:
        raise RuntimeError("No items in test.jsonl")

    # Load logits and targets
    data = np.load(cache_npz, allow_pickle=True)
    all_logits = {k: data[f"logits_{k}"] for k in CATS}
    all_targets = {k: data[f"targets_{k}"] for k in CATS}
    all_probs = {k: softmax_np(all_logits[k]) for k in CATS}

    # ------------------------------------------------------------------
    # 1) Pick an index whose frame we will visualize:
    #    first sample with at least one non-"none" GT label.
    # ------------------------------------------------------------------
    chosen_idx = None
    for i in range(N):
        has_non_none = False
        for cat in CATS:
            classes = label_space[cat]
            gt_idx = int(all_targets[cat][i])
            if classes[gt_idx] != "none":
                has_non_none = True
                break
        if has_non_none and items[i].get("frames"):
            chosen_idx = i
            break

    if chosen_idx is None:
        raise RuntimeError("Could not find any test sample with non-'none' labels.")

    ref_item = items[chosen_idx]
    ref_frames = ref_item["frames"]
    ref_last_frame = ref_frames[-1]
    print(f"[INFO] Using reference frame from sample index {chosen_idx}: {ref_last_frame}")

    # ------------------------------------------------------------------
    # 2) Collect all tubes whose last frame matches this image.
    # ------------------------------------------------------------------
    same_frame_indices = []
    for i, it in enumerate(items):
        frames = it.get("frames", [])
        if not frames:
            continue
        if frames[-1] == ref_last_frame:
            same_frame_indices.append(i)

    if not same_frame_indices:
        raise RuntimeError("No tubes share the chosen frame; something is wrong.")

    print(f"[INFO] Found {len(same_frame_indices)} tubes ending at this frame.")

    # Load the underlying image
    base_im = Image.open(ref_last_frame).convert("RGB")
    W, H = base_im.size

    # Font setup: larger and more visible
    try:
        font = ImageFont.truetype("Arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    lh = line_height(font)

    # Colour palette for boxes / legend
    COLORS = [
        (255, 0, 0),      # red
        (0, 255, 0),      # green
        (0, 128, 255),    # blue-ish
        (255, 215, 0),    # gold
        (255, 105, 180),  # hot pink
        (0, 255, 255),    # cyan
        (255, 140, 0),    # orange
    ]

    # Collect bounding boxes and legend entries BEFORE drawing
    people = []         # each: dict with box + color + logical_lines
    all_boxes = []      # list of (tlx, tly, brx, bry)

    for j, idx in enumerate(same_frame_indices):
        it = items[idx]
        try:
            bboxes = ensure_xywh_per_frame(it)
            x, y, w, h = bboxes[-1]
        except Exception:
            continue

        # Clamp bbox to image bounds
        tlx = int(max(0, math.floor(x)))
        tly = int(max(0, math.floor(y)))
        brx = int(min(W, math.ceil(x + w)))
        bry = int(min(H, math.ceil(y + h)))

        color = COLORS[j % len(COLORS)]

        # Build logical lines for legend (only heads with non-"none" GT)
        lines = [f"Person {j+1}"]
        for cat in CATS:
            classes = label_space[cat]
            gt_idx = int(all_targets[cat][idx])
            gt_label = classes[gt_idx]
            if gt_label == "none":
                continue
            pred_idx = int(all_probs[cat][idx].argmax())
            pred_label = classes[pred_idx]
            lines.append(f"{cat} GT: {gt_label}")
            lines.append(f"{cat} Pred: {pred_label}")

        people.append(
            {
                "box": (tlx, tly, brx, bry),
                "color": color,
                "logical_lines": lines,
            }
        )
        all_boxes.append((tlx, tly, brx, bry))

    if not people:
        raise RuntimeError("No valid people with boxes found for this frame.")

    # ------------------------------------------------------------------
    # 3) Compute crop that contains all boxes, preserving aspect ratio.
    # ------------------------------------------------------------------
    min_x = min(b[0] for b in all_boxes)
    min_y = min(b[1] for b in all_boxes)
    max_x = max(b[2] for b in all_boxes)
    max_y = max(b[3] for b in all_boxes)

    union_w = max_x - min_x
    union_h = max_y - min_y

    # Add padding around union box
    pad_x = int(0.05 * union_w)
    pad_y = int(0.05 * union_h)
    min_x = max(0, min_x - pad_x)
    min_y = max(0, min_y - pad_y)
    max_x = min(W, max_x + pad_x)
    max_y = min(H, max_y + pad_y)

    union_w = max_x - min_x
    union_h = max_y - min_y

    aspect = W / H

    if union_w <= 0 or union_h <= 0:
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, W, H
    else:
        crop_w = float(union_w)
        crop_h = float(union_h)
        current_aspect = crop_w / crop_h

        if current_aspect < aspect:
            crop_w = aspect * crop_h
        elif current_aspect > aspect:
            crop_h = crop_w / aspect

        crop_w = min(crop_w, W)
        crop_h = min(crop_h, H)

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        crop_x1 = int(round(center_x - crop_w / 2.0))
        crop_y1 = int(round(center_y - crop_h / 2.0))

        crop_x1 = max(0, min(crop_x1, W - int(crop_w)))
        crop_y1 = max(0, min(crop_y1, H - int(crop_h)))

        crop_x2 = int(crop_x1 + crop_w)
        crop_y2 = int(crop_y1 + crop_h)

    # Crop image
    cropped_im = base_im.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    Wc, Hc = cropped_im.size

    # Shift boxes to cropped coordinates
    for person in people:
        tlx, tly, brx, bry = person["box"]
        person["box"] = (tlx - crop_x1, tly - crop_y1, brx - crop_x1, bry - crop_y1)

    # ------------------------------------------------------------------
    # 4) Draw boxes on cropped image.
    # ------------------------------------------------------------------
    draw_c = ImageDraw.Draw(cropped_im)
    for person in people:
        tlx, tly, brx, bry = person["box"]
        color = person["color"]
        draw_c.rectangle([tlx, tly, brx, bry], outline=color, width=4)

    # ------------------------------------------------------------------
    # 5) Wrap legend text and compute legend height.
    # ------------------------------------------------------------------
    legend_width = 480
    LEGEND_LEFT_MARGIN = 20
    LEGEND_RIGHT_MARGIN = 20
    SWATCH_W = 28
    TEXT_LEFT_PAD = 10
    spacing = lh // 2
    margin_top_bottom = 40

    # Available width for text inside the legend panel
    text_max_width = legend_width - (
        LEGEND_LEFT_MARGIN + SWATCH_W + TEXT_LEFT_PAD + LEGEND_RIGHT_MARGIN
    )

    # Wrap logical lines into visual lines per person
    total_legend_lines = 0
    n_people_with_lines = 0
    for person in people:
        logical = person["logical_lines"]
        wrapped = []
        for line in logical:
            wrapped.extend(wrap_text_line(line, font, text_max_width))
        person["wrapped_lines"] = wrapped
        if wrapped:
            total_legend_lines += len(wrapped)
            n_people_with_lines += 1

    total_legend_height = (
        total_legend_lines * lh + max(0, n_people_with_lines - 1) * spacing
    )
    H_needed = total_legend_height + 2 * margin_top_bottom
    H_new = max(Hc, H_needed)

    new_im = Image.new("RGB", (Wc + legend_width, H_new), color=(0, 0, 0))

    # Paste cropped image vertically centered if canvas is taller
    offset_y_img = (H_new - Hc) // 2
    new_im.paste(cropped_im, (0, offset_y_img))

    # Draw legend, vertically centered
    legend_draw = ImageDraw.Draw(new_im)
    x0 = Wc + LEGEND_LEFT_MARGIN
    y0 = (H_new - total_legend_height) // 2 if total_legend_height > 0 else 20

    current_y = y0
    for person in people:
        color = person["color"]
        wrapped = person["wrapped_lines"]
        if not wrapped:
            continue

        # Colour swatch
        sw_h = lh
        legend_draw.rectangle(
            [x0, current_y, x0 + SWATCH_W, current_y + sw_h], fill=color
        )

        text_x = x0 + SWATCH_W + TEXT_LEFT_PAD
        text_y = current_y

        for line in wrapped:
            legend_draw.text((text_x, text_y), line, font=font, fill=color)
            text_y += lh

        current_y = text_y + spacing

    new_im.save(out_img_path)
    print(f"[OK] Saved cropped multi-person visualization with wrapped legend to: {out_img_path}")


if __name__ == "__main__":
    main()
