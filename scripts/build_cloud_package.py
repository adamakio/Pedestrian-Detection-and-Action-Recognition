import shutil
import tarfile
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]  # titan_data/
DATASET = ROOT / "dataset"
YOLO = ROOT / "yolo"

TRAIN_SPLIT = DATASET / "train_set.txt"
VAL_SPLIT = DATASET / "val_set.txt"

DEST_ROOT = ROOT.parent / "titan_yolo_min"  # sibling to titan_data
TAR_PATH = ROOT.parent / "titan_yolo_min.tar.gz"


def read_clips(split_path: Path):
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    return {
        line.strip()
        for line in split_path.read_text().splitlines()
        if line.strip()
    }

def copy_scripts():
    src = ROOT / "scripts"
    dst = DEST_ROOT / "scripts"
    if not src.exists():
        print(f"[WARN] scripts/ not found at {src}, skipping.")
        return
    print("[INFO] Copying scripts/ ...")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_yolo_labels():
    print("[INFO] Copying yolo/labels/{train,val} ...")
    for split in ["train", "val"]:
        src = YOLO / "labels" / split
        dst = DEST_ROOT / "yolo" / "labels" / split
        if not src.exists():
            print(f"[WARN] Missing labels dir: {src}")
            continue
        shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_yolo_images_resolved():
    """
    Copy yolo/images/{train,val} but resolve symlinks into real files.
    This way the archive is self-contained on the cloud.
    """
    print("[INFO] Copying yolo/images/{train,val} with resolved symlinks...")

    for split in ["train", "val"]:
        src_dir = YOLO / "images" / split
        dst_dir = DEST_ROOT / "yolo" / "images" / split

        if not src_dir.exists():
            print(f"[WARN] Missing images dir: {src_dir}")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img in src_dir.glob("*.png"):
            dst = dst_dir / img.name
            if img.is_symlink():
                real = img.resolve()
                if not real.exists():
                    print(f"[WARN] Broken symlink skipped: {img} -> {real}")
                    continue
                shutil.copy2(real, dst)
            else:
                shutil.copy2(img, dst)
            count += 1

        print(f"[INFO]  - {split}: copied {count} images")


def copy_yolo_yaml():
    src = YOLO / "titan_person.yaml"
    if not src.exists():
        print(f"[WARN] titan_person.yaml not found at {src}, skipping.")
        return
    dst = DEST_ROOT / "yolo" / "titan_person.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print("[INFO] Copied yolo/titan_person.yaml")


def create_tar():
    if TAR_PATH.exists():
        print(f"[INFO] Removing existing archive: {TAR_PATH}")
        TAR_PATH.unlink()

    print(f"[INFO] Creating archive: {TAR_PATH}")
    with tarfile.open(TAR_PATH, "w:gz") as tar:
        # Store everything under top-level titan_yolo_min/
        tar.add(DEST_ROOT, arcname="titan_yolo_min")
    print("[INFO] Archive created.")


def main():
    print("========== Building cloud package ==========")
    print(f"[INFO] ROOT:      {ROOT}")
    print(f"[INFO] DEST_ROOT: {DEST_ROOT}")
    print(f"[INFO] TAR_PATH:  {TAR_PATH}")
    print("============================================")

    # Start clean
    if DEST_ROOT.exists():
        print(f"[INFO] Removing existing {DEST_ROOT}")
        shutil.rmtree(DEST_ROOT)

    # 1) collect clips from train + val
    train_clips = read_clips(TRAIN_SPLIT)
    val_clips = read_clips(VAL_SPLIT)
    clips = train_clips | val_clips
    print(f"[INFO] Train clips: {len(train_clips)}, Val clips: {len(val_clips)}, Union: {len(clips)}")

    # 2) copy scripts
    copy_scripts()

    # 3) copy YOLO labels (train+val)
    copy_yolo_labels()

    # 4) copy YOLO images (train+val), resolving symlinks
    copy_yolo_images_resolved()

    # 5) copy yaml
    copy_yolo_yaml()

    # 6) pack into tar.gz
    create_tar()

    print("[INFO] Done. Upload titan_yolo_min.tar.gz to your cloud instance.")


if __name__ == "__main__":
    create_tar()