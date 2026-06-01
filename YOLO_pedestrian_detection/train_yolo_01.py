from ultralytics import YOLO
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_CFG = ROOT / "yolo_0.1" / "titan_person_0.1.yaml"

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def main():
    device = get_device()
    print(f"[INFO] Using device: {device}")

    model = YOLO("runs/detect/titan_person_11n_0.1/weights/last.pt")

    model.train(
        data=str(DATA_CFG),
        device=device,
        epochs=20,
        patience=5,
        imgsz=640,
        batch=32,
        workers=0,        # safer on macOS
        optimizer="SGD",
        lrf=0.1,
        cos_lr=True,
        plots=False,
        save_period=1,
        exist_ok=False,
        name="titan_person_11n_0.1v2",
    )

if __name__ == "__main__":
    main()
