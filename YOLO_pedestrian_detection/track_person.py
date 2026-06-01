from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import re

# --- config ---
# MODEL   = "runs/detect/titan_person_11n_0.1_merged/weights/best.pt"
MODEL = "yolo11l.pt"
TRACKER = "bytetrack.yaml"        # or "botsort.yaml"
DEVICE  = "mps"
IMGSZ   = 992
CONF    = 0.25
IOU     = 0.7
FPS     = 15                      # pick a sensible FPS for image sequences

# Read 10% of test clips
with open("dataset/test_set.txt", "r") as f:
    test_clips = [line.strip() for line in f]
test_clips = test_clips[:10]

PROJECT = f"runs/track_{TRACKER.split('.')[0]}_yolo_11l"

# Natural sort helper so frames go 000001, 000002, ...
_num = re.compile(r'(\d+)')
def _ns_key(s: str):
    return tuple(int(x) if x.isdigit() else x for x in _num.split(s))

model = YOLO(MODEL)

root = Path("dataset/images_anonymized")
for clip_dir in sorted(root.glob("clip_*")):
    if clip_dir.name not in test_clips:
        continue

    src_dir = clip_dir / "images"           # your test frames live here
    if not src_dir.exists():
        print(f"[SKIP] {clip_dir.name}: no {src_dir}")
        continue

    # Collect image paths in order
    imgs = sorted([p for p in src_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
                  key=lambda p: _ns_key(p.name))
    if not imgs:
        print(f"[SKIP] {clip_dir.name}: no images found")
        continue

    out_dir = Path(PROJECT) / clip_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f"{clip_dir.name}.mp4"
    mot_path = out_dir / "tracks_mot16.txt"

    # Prime size from first frame
    first = cv2.imread(str(imgs[0]))
    if first is None:
        print(f"[SKIP] {clip_dir.name}: can't read first frame")
        continue
    H, W = first.shape[:2]
    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (W, H),
        True
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {mp4_path}")

    # Track over the ordered list of images; don't let Ultralytics save frames
    # stream=True yields Results per frame with persistent tracker state
    results = model.track(
        source=[str(p) for p in imgs],
        tracker=TRACKER,
        device=DEVICE,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        classes=[0],        # person only
        stream=True,
        save=False,         # IMPORTANT: don't write annotated JPGs
        verbose=False,
    )

    # Write MOT file and annotated video
    with mot_path.open("w") as f:
        for frame_idx, r in enumerate(results, start=1):
            frame = r.plot()  # draw detections/tracks on the image
            # Ensure BGR uint8 for VideoWriter
            if isinstance(frame, np.ndarray) and frame.shape[1] == W and frame.shape[0] == H:
                writer.write(frame)
            else:
                # fallback: read original to keep consistent size
                orig = cv2.imread(str(imgs[frame_idx-1]))
                if orig is not None:
                    writer.write(orig)

            if r.boxes is None or r.boxes.xywh is None:
                continue
            ids = r.boxes.id
            if ids is None:
                continue

            ids  = ids.cpu().numpy().astype(int)
            xywh = r.boxes.xywh.cpu().numpy()   # cx,cy,w,h in pixels
            cls  = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None

            for i in range(len(ids)):
                if cls is not None and int(cls[i]) != 0:
                    continue
                cx, cy, w, h = xywh[i]
                tlx = cx - w / 2.0
                tly = cy - h / 2.0
                tid = ids[i]
                # MOT: frame,id,x,y,w,h,1,-1,-1,-1  (x,y are top-left pixels)
                f.write(f"{frame_idx},{tid},{tlx:.2f},{tly:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

    writer.release()
    print(f"[OK] {clip_dir.name}: wrote {mp4_path.name} and {mot_path.name} -> {out_dir}")
