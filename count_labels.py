import json
from collections import Counter
from pathlib import Path

root = Path("action_det/data/splits_pct_20_T8_S4")
splits = ["train", "val", "test"]
heads = ["atomic", "simple-context", "complex-context",
         "communicative", "transportive"]

# aggregate per-head counts across splits
agg = {h: Counter() for h in heads}

for split in splits:
    with open(root / f"{split}_label_report.json") as f:
        report = json.load(f)
    for h in heads:
        agg[h].update(report[h])

summary = {}
for h in heads:
    total = sum(agg[h].values())
    none = agg[h].get("none", 0)
    non_none = total - none
    pct_none = 100.0 * none / total if total > 0 else 0.0
    summary[h] = {
        "total": total,
        "none": none,
        "non_none": non_none,
        "pct_none": pct_none,
    }

print(summary)
