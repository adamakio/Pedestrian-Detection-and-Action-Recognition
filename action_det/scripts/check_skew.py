import torch
from pathlib import Path

lbls = torch.load("cached_feats/train_labels.pt")  # (N, 5)

from _dataset import CATS  # or just hardcode the list
from pathlib import Path
import json

label_space = json.loads(Path("action_det/data/splits_pct_20_T8_S4/label_space.json").read_text())["label_space"]

for i, head in enumerate(CATS):
    counts = {}
    for idx in lbls[:, i].tolist():
        cls_name = label_space[head][idx]
        counts[cls_name] = counts.get(cls_name, 0) + 1
    print(f"\n{head}")
    for name, c in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name:30s} {c}")
