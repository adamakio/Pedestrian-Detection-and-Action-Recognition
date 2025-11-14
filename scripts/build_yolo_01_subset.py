import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # titan_data/
DATASET = ROOT / "dataset"
YOLO = ROOT / "yolo"
OUT_ROOT = ROOT / "yolo_0.1"

# How many clips per split
N_TRAIN = 40
N_VAL = 20
N_TEST = 10


def read_clips(path: Path, n: int):
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    clips = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    if len(clips) < n:
        raise ValueError(f"Requested {n} clips from {path.name}, but only {len(clips)} available.")
    return clips[:n]  # deterministic: first N


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def process_split(split: str, clips):
    """
    For each selected clip in this split:
      - find label files yolo/labels/<split>/clip_xxx_*.txt
      - symlink labels into yolo_0.1/labels/<split>/
      - symlink corresponding images into yolo_0.1/images/<split>/
        (preferring yolo/images/<split>, falling back to dataset/images_anonymized)
    """
    print(f"[INFO] Processing split='{split}' with {len(clips)} clips")

    src_labels_dir = YOLO / "labels" / split
    src_images_dir = YOLO / "images" / split

    if not src_labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {src_labels_dir}")

    out_labels_dir = OUT_ROOT / "labels" / split
    out_images_dir = OUT_ROOT / "images" / split
    ensure_clean_dir(out_labels_dir)
    ensure_clean_dir(out_images_dir)

    total_labels = 0
    total_images = 0

    for clip in clips:
        pattern = f"{clip}_*.txt"
        label_files = list(src_labels_dir.glob(pattern))

        if not label_files:
            print(f"[WARN] No labels found for {clip} in {split}, pattern={pattern}")
            continue

        print(f"[INFO]  - {clip}: {len(label_files)} frames")

        for lbl in label_files:
            stem = lbl.stem  # clip_xxx_yyyy
            img_name = stem + ".png"

            # Symlink label
            dst_lbl = out_labels_dir / lbl.name
            make_symlink(lbl, dst_lbl)
            total_labels += 1

            # Find image: prefer yolo/images/<split>, else raw dataset
            src_img = src_images_dir / img_name
            if not src_img.exists():
                # fallback to dataset/images_anonymized/clip_xxx/images/yyyy.png
                try:
                    clip_id, frame_id = stem.rsplit("_", 1)
                except ValueError:
                    clip_id = clip
                    frame_id = stem.split("_")[-1]
                src_img = (
                    DATASET
                    / "images_anonymized"
                    / clip
                    / "images"
                    / f"{frame_id}.png"
                )

            if not src_img.exists():
                print(f"[WARN]   Missing image for {lbl.name}: expected {src_img}")
                continue

            dst_img = out_images_dir / img_name
            make_symlink(src_img, dst_img)
            total_images += 1

    print(f"[INFO] Done split='{split}': {total_labels} labels, {total_images} images")


def write_yaml():
    yaml_path = OUT_ROOT / "titan_person_0.1.yaml"
    content = f"""# 10% TITAN subset (40/20/10 clips)
path: {OUT_ROOT}

train: images/train
val: images/val
test: images/test

names:
  0: person
"""
    yaml_path.write_text(content)
    print(f"[INFO] Wrote subset YAML: {yaml_path}")


def main():
    print("========== Building 10% YOLO subset: yolo_0.1 ==========")
    print(f"[INFO] ROOT:     {ROOT}")
    print(f"[INFO] OUT_ROOT: {OUT_ROOT}")

    # Ensure root exists
    if OUT_ROOT.exists():
        print(f"[INFO] Removing existing {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    # Read clips
    train_clips = read_clips(DATASET / "train_set.txt", N_TRAIN)
    val_clips = read_clips(DATASET / "val_set.txt", N_VAL)
    test_clips = read_clips(DATASET / "test_set.txt", N_TEST)

    print(f"[INFO] Using clips -> train: {len(train_clips)}, val: {len(val_clips)}, test: {len(test_clips)}")

    # Build splits
    process_split("train", train_clips)
    process_split("val", val_clips)
    process_split("test", test_clips)

    # YAML
    write_yaml()

    print("[INFO] Done. Use data='yolo_0.1/titan_person_0.1.yaml' for training.")


if __name__ == "__main__":
    main()
