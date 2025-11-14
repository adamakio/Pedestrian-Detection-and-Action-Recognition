from ultralytics import YOLO

# Path to your previous run's last checkpoint
last_ckpt = "runs/detect/titan_person_11n_0.1/weights/last.pt"

model = YOLO(last_ckpt)

model.train(
    resume=True,  # tells Ultralytics to load and continue training state
    epochs=40
)
