from pathlib import Path
import os
import random

ROOT = Path(__file__).resolve().parents[1]  # titan_data/
YOLO_ROOT = ROOT / "yolo"

VAL_IMG_DIR = YOLO_ROOT / "images" / "val"
VAL_LBL_DIR = YOLO_ROOT / "labels" / "val"

SMALL_IMG_DIR = YOLO_ROOT / "images" / "val_small"
SMALL_LBL_DIR = YOLO_ROOT / "labels" / "val_small"

NUM_SAMPLES = 10  # adjust as you like


def make_symlink(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)


def main():
    imgs = sorted(VAL_IMG_DIR.glob("*.png"))
    if len(imgs) == 0:
        print(f"[ERROR] No images found in {VAL_IMG_DIR}")
        return

    n = min(NUM_SAMPLES, len(imgs))
    samples = random.sample(imgs, n)
    print(f"[INFO] Creating val_small with {n} samples")

    for img in samples:
        stem = img.stem
        lbl = VAL_LBL_DIR / f"{stem}.txt"
        if not lbl.exists():
            # skip images without labels
            continue

        img_dst = SMALL_IMG_DIR / img.name
        lbl_dst = SMALL_LBL_DIR / lbl.name

        make_symlink(img, img_dst)
        make_symlink(lbl, lbl_dst)

    print(f"[INFO] Done. Images in: {SMALL_IMG_DIR}")
    print(f"[INFO]       Labels in: {SMALL_LBL_DIR}")


if __name__ == "__main__":
    main()
