# action_det/scripts/_sampler.py
import json, numpy as np, torch
from pathlib import Path
from torch.utils.data import WeightedRandomSampler

def build_weighted_sampler(items, label_space_json, repeat_factor: float = 1.5,
                           exclude_none: bool = True, cap: float = 10.0):
    """
    Returns a torch WeightedRandomSampler that samples with replacement for exactly len(items) samples/epoch.
    - repeat_factor > 1.0 boosts rare classes more
    - exclude_none optionally ignores 'none' when computing rarity
    - cap prevents a single index from dominating
    """
    label_space = json.loads(Path(label_space_json).read_text())["label_space"]

    # Laplace smoothing
    counts = {h: {c: 1 for c in label_space[h]} for h in label_space}
    for it in items:
        labs = it.get("labels", {})
        for h, c in labs.items():
            if exclude_none and c == "none":
                continue
            if c in counts[h]:
                counts[h][c] += 1

    rarity = {h: {c: 1.0 / counts[h][c] for c in counts[h]} for h in counts}

    w = []
    for it in items:
        labs = it.get("labels", {})
        r = 0.0
        for h, c in labs.items():
            if exclude_none and c == "none":
                continue
            r += rarity[h].get(c, 0.0)
        w.append(r)

    w = np.asarray(w, dtype=np.float64)
    # normalize around 1, apply boost, clip
    w = w / (w.mean() + 1e-12)
    w = np.power(w, repeat_factor)
    w = np.clip(w, 1e-3, cap)

    weights = torch.tensor(w, dtype=torch.double)
    num_samples = len(items)
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    return sampler
