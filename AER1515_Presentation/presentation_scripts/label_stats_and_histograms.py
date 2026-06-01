#!/usr/bin/env python3
"""
Label stats + histograms for TITAN action heads (20% subset, T=16, S=4).

Outputs:
- Console:
    * #tubes per split (train / val / test)
    * %none for each head in train / val / test / all

- Figures (train only, excluding 'none'):
    * presentation_outputs/histogram_images/<head>_train_distribution.png
    * presentation_outputs/histogram_images/all_heads_train_distribution.png
"""

from pathlib import Path
from collections import Counter
import json

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------
SPLIT_TAG = "splits_pct_20_T16_S4"
DATA_PATH = Path("action_recognition/data") / SPLIT_TAG

JSONL_FILES = {
    "train": DATA_PATH / "train.jsonl",
    "val":   DATA_PATH / "val.jsonl",
    "test":  DATA_PATH / "test.jsonl",
}

REPORT_FILES = {
    "train": DATA_PATH / "train_label_report.json",
    "val":   DATA_PATH / "val_label_report.json",
    "test":  DATA_PATH / "test_label_report.json",
}

IMG_OUTPUT_DIR = Path("AER1515_Presentation/presentation_outputs/histogram_images")
IMG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_HEADS_IMG = IMG_OUTPUT_DIR / "all_heads_train_distribution.png"

# Fixed head order (matches dataset / training code)
CATS = ["atomic", "simple-context", "complex-context", "communicative", "transportive"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_label_space() -> dict:
    """Return dict[head] -> list[label] from label_space.json."""
    js = json.loads((DATA_PATH / "label_space.json").read_text())
    return js["label_space"]


def load_split_reports() -> dict:
    """Return dict[split] -> dict[head] -> {label: count}."""
    reports = {}
    for split, path in REPORT_FILES.items():
        reports[split] = json.loads(path.read_text())
    return reports


def count_tubes_from_jsonl(path: Path) -> int:
    """Number of tubes == number of lines in the split JSONL."""
    n = 0
    with path.open("r") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def pct_none(counts) -> float:
    """Percentage of 'none' labels in a counts dict."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return 100.0 * counts.get("none", 0) / total


# ---------------------------------------------------------------------
# 1) Tube counts per split
# ---------------------------------------------------------------------
def print_tube_counts_per_split():
    print("Split, #tubes")
    for split, path in JSONL_FILES.items():
        n = count_tubes_from_jsonl(path)
        print(f"{split:5s} {n:7d}")
    print()  # blank line


# ---------------------------------------------------------------------
# 2) Table: %none per split + combined
# ---------------------------------------------------------------------
def print_none_percentage_table(reports):
    train_rep = reports["train"]
    val_rep   = reports["val"]
    test_rep  = reports["test"]

    print("Head, %none_train, %none_val, %none_test, %none_all")
    for head in CATS:
        tr_counts = train_rep[head]
        va_counts = val_rep[head]
        te_counts = test_rep[head]

        comb_counts = Counter(tr_counts) + Counter(va_counts) + Counter(te_counts)

        p_tr  = pct_none(tr_counts)
        p_va  = pct_none(va_counts)
        p_te  = pct_none(te_counts)
        p_all = pct_none(comb_counts)

        print(f"{head:15s} {p_tr:8.1f} {p_va:11.1f} {p_te:11.1f} {p_all:11.1f}")
    print()  # blank line


# ---------------------------------------------------------------------
# 3) Per-head train histograms (static, EXCLUDING 'none')
# ---------------------------------------------------------------------
def save_per_head_train_histograms(label_space, train_rep):
    for head in CATS:
        # exclude 'none' from the histogram
        labels = [lbl for lbl in label_space[head] if lbl != "none"]
        if not labels:
            continue  # nothing to plot

        counts = np.array(
            [train_rep[head].get(lbl, 0) for lbl in labels],
            dtype=np.int64,
        )

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x, counts)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Tube count (train)")
        # no title; you'll add it in LaTeX if needed
        fig.tight_layout()

        out_path = IMG_OUTPUT_DIR / f"{head}_train_distribution.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[OK] Saved {out_path}")


# ---------------------------------------------------------------------
# 4) Single color-coded histogram across all heads (train only, EXCLUDING 'none')
# ---------------------------------------------------------------------
def plot_all_heads_histogram_train(label_space, train_rep):
    heads = CATS

    # One color per head from Matplotlib's default cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    head_colors = {
        head: color_cycle[i % len(color_cycle)]
        for i, head in enumerate(heads)
    }

    x_labels = []   # class names (without 'none')
    heights = []    # counts
    colors = []     # bar colors by head
    boundaries = [] # x positions for vertical separators

    cur = 0
    for head in heads:
        # per-head labels excluding 'none'
        head_labels = [lbl for lbl in label_space[head] if lbl != "none"]
        if not head_labels:
            continue

        for lbl in head_labels:
            x_labels.append(lbl)
            heights.append(train_rep[head].get(lbl, 0))
            colors.append(head_colors[head])

        cur += len(head_labels)
        boundaries.append(cur - 0.5)  # separator after this head's last bar

    if not x_labels:
        print("[WARN] No non-'none' labels found; skipping all-heads histogram.")
        return

    x = np.arange(len(x_labels))
    heights = np.asarray(heights, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(x, heights, color=colors)

    for rect, h in zip(bars, heights):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            h,
            str(int(h)),
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=0,   # vertical so it fits
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Tube count (train)")
    ax.set_yscale("log")
    # no title; handled in LaTeX

    # Vertical dashed lines to separate heads (skip last far-right boundary)
    for b in boundaries[:-1]:
        ax.axvline(
            b,
            linestyle="--",
            color="gray",
            linewidth=0.8,
            alpha=0.7,
        )

    # Make room at the bottom for ticks + legend
    fig.subplots_adjust(bottom=0.30, top=0.95)

    # Legend: one entry per head, single row below the plot
    handles = [
        plt.Line2D([0], [0], color=head_colors[h], lw=4, label=h)
        for h in heads
    ]
    ax.legend(
        handles=handles,
        title="Head",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),  # slightly below the axis
        ncol=len(heads),
        frameon=False,
    )



    fig.savefig(ALL_HEADS_IMG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {ALL_HEADS_IMG}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    label_space = load_label_space()
    reports = load_split_reports()
    train_rep = reports["train"]

    # 1) #tubes per split (from JSONL)
    print_tube_counts_per_split()

    # 2) %none per head (train / val / test / all)
    print_none_percentage_table(reports)

    # 4) All-heads color-coded histogram (train only, no 'none')
    plot_all_heads_histogram_train(label_space, train_rep)


if __name__ == "__main__":
    main()
