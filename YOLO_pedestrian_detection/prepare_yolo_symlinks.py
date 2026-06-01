from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]      # titan_data/
DATASET = ROOT / "dataset"
IMAGES_SRC = DATASET / "images_anonymized"
YOLO_ROOT = ROOT / "yolo"

SPLITS = ["train", "val", "test"]


def make_symlink(src: Path, dst: Path):
    if dst.exists():
        # If it's already the correct symlink or a file, leave it
        return
    # Ensure parent dir exists
    dst.parent.mkdir(parents=True, exist_ok=True)
    # On macOS, this creates a symlink without copying data
    os.symlink(src, dst)


def main():
    for split in SPLITS:
        labels_dir = YOLO_ROOT / "labels" / split
        images_out_dir = YOLO_ROOT / "images" / split

        if not labels_dir.exists():
            print(f"[WARN] Labels dir missing for {split}: {labels_dir}")
            continue

        images_out_dir.mkdir(parents=True, exist_ok=True)

        label_files = sorted(labels_dir.glob("*.txt"))
        print(f"[INFO] {split}: found {len(label_files)} label files")

        missing_images = 0
        created_links = 0

        for lf in label_files:
            # label filename: clip_306_000006.txt
            stem = lf.stem  # "clip_306_000006"
            parts = stem.split("_")
            if len(parts) < 3:
                print(f"[WARN] Unexpected label name: {lf.name}")
                continue

            clip = "_".join(parts[0:2])   # "clip_306"
            frame = parts[2]              # "000006"

            src_img = IMAGES_SRC / clip / "images" / f"{frame}.png"
            dst_img = images_out_dir / f"{stem}.png"

            if not src_img.exists():
                print(f"[WARN] Missing source image for {lf.name}: {src_img}")
                missing_images += 1
                continue

            make_symlink(src_img, dst_img)
            created_links += 1

        print(
            f"[INFO] {split}: created {created_links} symlinks, "
            f"{missing_images} missing images"
        )


if __name__ == "__main__":
    main()
