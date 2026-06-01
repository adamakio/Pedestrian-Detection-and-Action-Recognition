import pandas as pd
from pathlib import Path

def last_metrics_csv(path):
    df = pd.read_csv(path)
    # Ultralytics metrics.csv usually has rows per epoch; take the last row
    return df.iloc[-1]

root = Path("./runs/detect")

# YOLO11n full (1 epoch)
n_full = pd.read_csv(root / "titan_person_11n" / "test_results_yolo_11n" / "metrics.csv").iloc[0]

# YOLO11n 10% (final merged run, no TTA)
n_01 = pd.read_csv(root / "titan_person_11n_0.1_merged" / "test_results_wo_tta" / "metrics.csv").iloc[0]
n_01_w_tta = pd.read_csv(root / "titan_person_11n_0.1_merged" / "test_results_w_tta" / "metrics.csv").iloc[0]

# YOLO11L (off-the-shelf)
l_full = pd.read_csv(root / "titan_person_11l" / "test_results_yolo_11l" / "metrics.csv").iloc[0]

for name, row in [("yolo11n_full", n_full), ("yolo11n_0.1_wo_tta", n_01), ("yolo_11n_0.1_w_tta", n_01_w_tta), ("yolo11l", l_full)]:
    print(name)
    for header in ["Images","Instances","Box-P","Box-R","Box-F1","mAP50","mAP50-95"]:
        print(
            header, row[header]
        )
