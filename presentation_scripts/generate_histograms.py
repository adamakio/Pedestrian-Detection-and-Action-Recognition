#!/usr/bin/env python3
"""
Generate animated histograms of label distributions for each head.

- Reads train.jsonl from the 20% split (T=16, S=4).
- For each head in label_space:
    * counts number of tubes per label
    * computes percentage per label
    * saves an animated bar chart as MP4 where bars grow from 0
      and the count / percentage text above each bar counts up.

Output directory:
    presentation_scripts/histograms
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# -------------------------------------------------------------------
# CONSTANTS (from your message)
# -------------------------------------------------------------------
DATA_PATH = Path("action_det/data/splits_pct_20_T16_S4")

LABEL_SPACE = {
    "atomic": [
        "none",
        "walking",
        "standing",
        "sitting",
        "bending",
        "running",
        "squatting",
    ],
    "simple-context": [
        "none",
        "walking on the road",
        "biking",
        "entering a building",
        "opening",
        "exiting a building",
        "motorcycling",
        "closing",
    ],
    "complex-context": [
        "none",
        "unloading",
        "loading",
    ],
    "communicative": [
        "none",
        "talking in group",
        "talking on phone",
    ],
    "transportive": [
        "none",
        "pulling",
        "pushing",
    ],
}

TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
TEST_FILE = "test.jsonl"
LABELS_KEY = "labels"

OUTPUT_DIR = Path("presentation_outputs/histograms")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# where to save the final frame (static PNG)
IMG_OUTPUT_DIR = Path("presentation_outputs/histogram_images")
IMG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# -------------------------------------------------------------------
# DATA LOADING / COUNTING
# -------------------------------------------------------------------
def load_labels_from_jsonl(path: Path):
    """Yield the 'labels' dict for each example in a JSONL file."""
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj.get(LABELS_KEY, {})


def compute_counts_for_head(head: str, jsonl_path: Path):
    """
    Count occurrences of each label in LABEL_SPACE[head] within jsonl_path.

    Returns
    -------
    labels_ordered : list[str]
    counts : np.ndarray
    """
    labels_ordered = LABEL_SPACE[head]
    idx = {lab: i for i, lab in enumerate(labels_ordered)}
    counts = np.zeros(len(labels_ordered), dtype=np.int64)

    for labels_dict in load_labels_from_jsonl(jsonl_path):
        lab = labels_dict.get(head, "none")
        if lab not in idx:
            # unseen label, you can either skip or add
            # here we skip but warn once
            # print(f"[WARN] Unknown label for head {head}: {lab}")
            continue
        counts[idx[lab]] += 1

    return labels_ordered, counts


# -------------------------------------------------------------------
# ANIMATION
# -------------------------------------------------------------------
def animate_head_distribution(head: str, labels, counts, save_path: Path):
    """
    Create an animated bar chart where bars grow from 0 to counts.

    Text above each bar shows:
        count
        percentage (of total for this head)
    and both count and percentage "count up" with the animation.
    """
    total = counts.sum()
    if total == 0:
        print(f"[WARN] Total count is 0 for head '{head}', skipping.")
        return

    percentages = counts / total * 100.0
    n_labels = len(labels)

    x = np.arange(n_labels)
    max_count = counts.max()
    if max_count == 0:
        max_count = 1

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, np.zeros_like(counts), color="tab:blue")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"{head} head – label distribution (train, 20% subset)")
    ax.set_ylim(0, max_count * 1.15)
    fig.subplots_adjust(bottom=0.30) 
    
    # Text above each bar
    texts = []
    for rect in bars:
        tx = rect.get_x() + rect.get_width() / 2.0
        ty = 0.0
        txt = ax.text(
            tx,
            ty,
            "",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )
        texts.append(txt)

    n_frames = 60

    def update(frame_idx):
        # progress from 0..1
        prog = frame_idx / (n_frames - 1)
        for rect, target_count, pct, txt in zip(bars, counts, percentages, texts):
            h = target_count * prog
            rect.set_height(h)
            txt.set_y(h + max_count * 0.02)

            cur_count = int(round(target_count * prog))
            cur_pct = pct * prog
            txt.set_text(f"{cur_count}\n{cur_pct:.1f}%")

        return list(bars) + texts

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=40,
        blit=False,
    )

    writer = FFMpegWriter(fps=30, bitrate=2000)
    anim.save(save_path, writer=writer, dpi=200)

    # save the final frame as a PNG for the slides
    png_path = IMG_OUTPUT_DIR / f"{head}_distribution.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved animated histogram for '{head}' to {save_path}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    train_path = DATA_PATH / TRAIN_FILE
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    print(f"[INFO] Reading train data from {train_path}")

    for head in LABEL_SPACE.keys():
        labels, counts = compute_counts_for_head(head, train_path)
        out_path = OUTPUT_DIR / f"{head}_distribution.mp4"
        animate_head_distribution(head, labels, counts, out_path)


if __name__ == "__main__":
    main()
