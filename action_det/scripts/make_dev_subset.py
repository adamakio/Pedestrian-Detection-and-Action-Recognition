#!/usr/bin/env python3
import json
import random
from pathlib import Path
from collections import defaultdict

"""
Creates a small, balanced dev subset:
- data/splits_pct_20_T8_S4/train_dev.jsonl (~5k samples)
- data/splits_pct_20_T8_S4/val_dev.jsonl   (~1k samples)

Assumes the standard AER1515 JSONL format produced by 01_build_index.py
"""

# Which split to use as source (you can change this)
SPLIT_DIR = Path("action_det/data/splits_pct_20_T8_S4")

TRAIN_SRC = SPLIT_DIR / "train.jsonl"
VAL_SRC   = SPLIT_DIR / "val.jsonl"

OUT_TRAIN = SPLIT_DIR / "train_dev.jsonl"
OUT_VAL   = SPLIT_DIR / "val_dev.jsonl"

# Set desired subset sizes
TARGET_TRAIN = 5000
TARGET_VAL   = 1000

# The 5 heads (must match CATS in your project)
HEADS = [
    "atomic",
    "simple-context",
    "complex-context",
    "communicative",
    "transportive",
]


def load_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def is_non_none(entry):
    """Return True if ANY head has a non-'none' label."""
    for h in HEADS:
        if entry["labels"].get(h, "none") != "none":
            return True
    return False


def build_subset(src_items, target_count, out_path):
    # Split source into rare vs. easy
    rare = [e for e in src_items if is_non_none(e)]
    easy = [e for e in src_items if not is_non_none(e)]

    print(f"[INFO] Total entries = {len(src_items)}")
    print(f"[INFO] Rare entries  = {len(rare)}")
    print(f"[INFO] Easy entries  = {len(easy)}")

    # Oversample rare (70%) + easy (30%)
    take_rare = int(target_count * 0.7)
    take_easy = target_count - take_rare

    subset = []

    # If not enough rare, sample with replacement
    if len(rare) >= take_rare:
        subset.extend(random.sample(rare, take_rare))
    else:
        subset.extend(random.choices(rare, k=take_rare))

    # Add easy examples
    if len(easy) >= take_easy:
        subset.extend(random.sample(easy, take_easy))
    else:
        subset.extend(random.choices(easy, k=take_easy))

    random.shuffle(subset)

    # Write output
    with open(out_path, "w") as f:
        for e in subset:
            f.write(json.dumps(e) + "\n")

    print(f"[OK] Wrote {out_path} with {len(subset)} entries.")


def main():
    print("[INFO] Loading source JSONLs…")
    train_items = load_jsonl(TRAIN_SRC)
    val_items   = load_jsonl(VAL_SRC)

    print("[INFO] Building train_dev.jsonl")
    build_subset(train_items, TARGET_TRAIN, OUT_TRAIN)

    print("[INFO] Building val_dev.jsonl")
    build_subset(val_items, TARGET_VAL, OUT_VAL)


if __name__ == "__main__":
    main()
