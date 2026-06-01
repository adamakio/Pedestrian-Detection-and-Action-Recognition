#!/usr/bin/env python3
from pathlib import Path
import csv
from collections import defaultdict

from heads import HEADS, COL2HEAD

CSV_ROOT = Path("dataset/titan_0_4")

# Collect all distinct labels we actually see in the CSVs
labels_by_col = defaultdict(set)

csv_files = sorted(CSV_ROOT.glob("clip_*.csv"))
print(f"Found {len(csv_files)} CSV files")

for csv_path in csv_files:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in COL2HEAD.keys():
                raw = (row.get(col) or "").strip()
                if raw:
                    labels_by_col[col].add(raw)

# Compare per head
for col, head in COL2HEAD.items():
    csv_labels = labels_by_col[col]
    heads_labels = set(HEADS[head])

    only_in_csv   = sorted(csv_labels - heads_labels)
    only_in_heads = sorted(heads_labels - csv_labels)

    print(f"\n=== {head} ({col}) ===")
    print("CSV labels:")
    for v in sorted(csv_labels):
        print("  -", repr(v))

    print("Only in CSV (missing from HEADS):")
    for v in only_in_csv:
        print("  -", repr(v))

    print("Only in HEADS (never seen in this subset):")
    for v in only_in_heads:
        print("  -", repr(v))
