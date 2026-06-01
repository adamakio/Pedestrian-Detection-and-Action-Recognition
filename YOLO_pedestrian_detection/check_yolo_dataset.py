from pathlib import Path
import cv2
import random

ROOT = Path(__file__).resolve().parents[1]      # titan_data/
YOLO_ROOT = ROOT / "yolo"

SPLITS = ["train", "val", "test"]


def main():
    print("========== YOLO Dataset Check ==========")
    print(f"[INFO] ROOT:      {ROOT}")
    print(f"[INFO] YOLO_ROOT: {YOLO_ROOT}")
    print("========================================")

    for split in SPLITS:
        images_dir = YOLO_ROOT / "images" / split
        labels_dir = YOLO_ROOT / "labels" / split

        if not images_dir.exists() or not labels_dir.exists():
            print(f"[WARN] Missing dirs for split {split}: {images_dir}, {labels_dir}")
            continue

        image_files = sorted([p for p in images_dir.glob("*.png")])
        label_files = sorted([p for p in labels_dir.glob("*.txt")])

        print(f"\n[INFO] Split: {split}")
        print(f"[INFO] Images: {len(image_files)}")
        print(f"[INFO] Labels: {len(label_files)}")

        # Map basenames for consistency check
        img_stems = {p.stem for p in image_files}
        lbl_stems = {p.stem for p in label_files}

        only_labels = lbl_stems - img_stems
        only_images = img_stems - lbl_stems

        if only_labels:
            print(f"[WARN] {len(only_labels)} labels without images (first 5): {list(only_labels)[:5]}")
        if only_images:
            print(f"[WARN] {len(only_images)} images without labels (first 5): {list(only_images)[:5]}")

        if not only_labels and not only_images:
            print("[INFO] 1:1 match between images and labels ✅")

        # Try reading a few random image + label pairs
        sample_count = min(5, len(image_files))
        if sample_count == 0:
            print("[WARN] No images to sample in this split.")
            continue

        samples = random.sample(image_files, sample_count)

        for img_path in samples:
            stem = img_path.stem
            lbl_path = labels_dir / f"{stem}.txt"

            if not lbl_path.exists():
                print(f"[WARN] Sample missing label: {lbl_path}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Failed to read image: {img_path}")
                continue

            with lbl_path.open("r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            print(f"[DEBUG] {split} sample '{stem}': "
                  f"img_shape={img.shape}, "
                  f"{len(lines)} boxes, "
                  f"first_line='{lines[0] if lines else 'EMPTY'}'")

    print("\n[INFO] YOLO dataset check complete.")
    print("[INFO] If you see 1:1 matches and readable samples above, step 2 is done ✅")


if __name__ == "__main__":
    main()
