from ultralytics import YOLO
from pathlib import Path

RUN    = Path("runs/detect/titan_person_11n_0.1_merged")
MODEL  = RUN / "weights" / "best.pt"
DATA   = "yolo_0.1/titan_person_0.1.yaml"  # has train/val/test
TEST_FOLDER = "test_results_yolo_11l"
RUN = Path("runs/detect/titan_person_11l")
if not RUN.exists():
    RUN.mkdir(parents=True, exist_ok=True)
MODEL = Path("yolo11l.pt")  # pre-trained model

model = YOLO(str(MODEL))
metrics = model.val(
    data=DATA,
    split="test",
    imgsz=992,        # match (or at least be close to) your train size; multiple of 32
    batch=16,
    device="mps",
    workers=0,
    rect=True,
    save_json=True,   # COCO-style json for your appendix
    plots=True,       # PR curve, F1/conf curve, etc.
    project=RUN,
    name=TEST_FOLDER,
)

with open(RUN / TEST_FOLDER / "metrics.csv", "w") as f:
    f.write(metrics.to_csv())

# model.predict(
#     source="yolo_0.1/images/test",
#     imgsz=992, device="mps", workers=0,
#     conf=0.25, iou=0.7, save=True, save_conf=True, project=RUN, name="test_vis"
# )
