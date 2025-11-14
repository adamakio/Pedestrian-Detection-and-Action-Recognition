from ultralytics import YOLO

data_cfg = "yolo/titan_person.yaml"

def train(model_ckpt, name):
    model = YOLO(model_ckpt)
    model.train(
        data=data_cfg,
        epochs=30,          # ↓ from default 100
        imgsz=992,          # ↑ from default 640
        batch=16,           # (optional) same as default; you can omit if you like
        workers=4,          # ↓ from default 8 (better for your Mac)
        device="mps",       # use Apple GPU
        cos_lr=True,        # enable cosine LR schedule (default: False)
        optimizer="SGD",    # override 'auto'
        lrf=0.1,            # > default 0.01 (final LR factor)
        patience=8,         # ↓ from default 100 (faster early stopping)
        mixup=0.15,         # > default 0.0
        save_period=1,      # save every epoch (default: -1 = disabled)
        plots=False,        # disable plots
        name=name
    )

train("yolo11n.pt", name="titan_person_11n_10_percent")
