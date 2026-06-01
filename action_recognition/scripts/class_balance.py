import json
from pathlib import Path

CATS = ["atomic", "simple-context", "complex-context", "communicative", "transportive"]

def load_rare_classes(report_path, freq_thresh=0.02):
    """
    Return dict[head] -> set of rare class names (excluding 'none').

    freq_thresh: classes with fraction <= freq_thresh are considered rare.
    """
    report = json.loads(Path(report_path).read_text())
    rare = {}
    for head in CATS:
        counts = report[head]
        total = sum(counts.values())
        rare_head = set()
        for cls, c in counts.items():
            if cls == "none":
                continue
            frac = c / total
            if frac <= freq_thresh:
                rare_head.add(cls)
        rare[head] = rare_head
    return rare
