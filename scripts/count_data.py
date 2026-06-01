import csv
from pathlib import Path

root = Path("dataset")
ann_dir = root / "titan_0_4"

# Load split membership
def load_ids(path):
    with open(path) as f:
        return {line.strip().split(".")[0] for line in f if line.strip()}

train_ids = load_ids(root / "train_set.txt")
val_ids   = load_ids(root / "val_set.txt")
test_ids  = load_ids(root / "test_set.txt")

def accumulate_stats(clip_ids):
    n_frames = 0
    n_person = 0
    for clip_id in clip_ids:
        csv_path = ann_dir / f"{clip_id}.csv"
        seen_frames = set()
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_id = int(row["frames"].split('.')[0])
                seen_frames.add(frame_id)
                if row["label"] == "person":
                    n_person += 1
        n_frames += len(seen_frames)
    return n_frames, n_person

train_frames, train_person = accumulate_stats(train_ids)
val_frames,   val_person   = accumulate_stats(val_ids)
test_frames,  test_person  = accumulate_stats(test_ids)

total_frames = train_frames + val_frames + test_frames
total_person = train_person + val_person + test_person

print("Frames   (train/val/test/total):",
      train_frames, val_frames, test_frames, total_frames)
print("Person boxes (train/val/test/total):",
      train_person, val_person, test_person, total_person)
