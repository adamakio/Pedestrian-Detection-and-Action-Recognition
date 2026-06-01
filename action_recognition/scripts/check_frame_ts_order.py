#!/usr/bin/env python3
import json
import csv
from pathlib import Path
from collections import defaultdict

# Run this script from the titan_data root
ROOT = Path('.').resolve()
print(f"[INFO] Running from root: {ROOT}")

JSONL_DIR = ROOT / "action_det" / "data" / "splits_pct_20_T8_S4"
JSONL_FILES = [
    "train.jsonl",
    "train_dev.jsonl",
    "val.jsonl",
    "val_dev.jsonl",
]

IMU_ROOT = ROOT / "dataset" / "imu_data"

def load_imu_for_clip(clip_name: str):
    """
    Load synced_sensors.csv for a clip and return:
        image_ts_map: dict mapping 'clip_x/images/000006.png' -> timestamp (float)
    """
    csv_path = IMU_ROOT / clip_name / "synced_sensors.csv"
    if not csv_path.exists():
        print(f"[WARN] No synced_sensors.csv for {clip_name} at {csv_path}")
        return {}

    image_ts_map = {}
    with csv_path.open("r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # [image_ts, image_path, accel_ts, accel, gyro_ts, ang_vel]
            try:
                image_ts = float(row[0])
                image_path = row[1]  # e.g. 'clip_317/images/000006.png'
            except (ValueError, IndexError):
                continue
            image_ts_map[image_path] = image_ts
    return image_ts_map


def frame_path_to_rel(frame_path: str) -> str:
    """
    Convert the absolute image path in JSONL to the relative path
    used in IMU CSVs, by taking the substring starting at 'clip_'.

    Example:
      '/Users/.../dataset/images_anonymized/clip_317/images/000006.png'
      -> 'clip_317/images/000006.png'
    """
    idx = frame_path.rfind("clip_")
    if idx == -1:
        # Fallback: just return basename (unlikely to match, but at least visible)
        return Path(frame_path).name
    return frame_path[idx:]


def check_jsonl_file(jsonl_path: Path):
    """
    Check that the frames in each JSONL entry are ordered by image_ts
    according to the corresponding synced_sensors.csv.
    """
    print(f"\n[INFO] Checking {jsonl_path}")
    n_entries = 0
    n_ok = 0
    n_out_of_order = 0
    n_missing_ts = 0

    # Cache IMU maps per clip to avoid re-reading
    imu_cache = {}

    with jsonl_path.open("r") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"[ERROR] Invalid JSON at {jsonl_path}, line {line_idx}")
                continue

            n_entries += 1
            clip = entry.get("clip")
            frames = entry.get("frames", [])
            track_id = entry.get("track_id", "NA")

            if not clip or not frames:
                print(f"[WARN] Missing clip/frames at {jsonl_path}, line {line_idx}")
                continue

            # Load IMU mapping for this clip if not cached
            if clip not in imu_cache:
                imu_cache[clip] = load_imu_for_clip(clip)
            image_ts_map = imu_cache[clip]

            ts_list = []
            missing_for_this_entry = False

            for fp in frames:
                rel = frame_path_to_rel(fp)
                if rel not in image_ts_map:
                    print(f"[WARN] Missing timestamp for frame {rel} (clip={clip})")
                    n_missing_ts += 1
                    missing_for_this_entry = True
                    ts_list.append(None)
                else:
                    ts_list.append(image_ts_map[rel])

            # If any timestamps are missing, we can still attempt order on the existing ones,
            # but let's primarily flag missing.
            if not ts_list or all(t is None for t in ts_list):
                # nothing to check
                continue

            # Check non-decreasing order for the timestamps we have
            # We'll skip comparisons that involve None.
            out_of_order = False
            last_ts = None
            for idx, t in enumerate(ts_list):
                if t is None:
                    continue
                if last_ts is not None and t < last_ts:
                    out_of_order = True
                    print(
                        f"[OUT-OF-ORDER] {jsonl_path.name} line {line_idx} "
                        f"(clip={clip}, track_id={track_id}) "
                        f"frame_index={idx}, prev_ts={last_ts:.6f}, cur_ts={t:.6f}"
                    )
                    break
                last_ts = t

            if out_of_order:
                n_out_of_order += 1
            else:
                n_ok += 1

    print(f"[SUMMARY] {jsonl_path.name}:")
    print(f"  Total entries       : {n_entries}")
    print(f"  OK order            : {n_ok}")
    print(f"  Out-of-order entries: {n_out_of_order}")
    print(f"  Missing timestamps  : {n_missing_ts}")


def main():
    for name in JSONL_FILES:
        path = JSONL_DIR / name
        if not path.exists():
            print(f"[WARN] {path} does not exist, skipping.")
            continue
        check_jsonl_file(path)


if __name__ == "__main__":
    main()
