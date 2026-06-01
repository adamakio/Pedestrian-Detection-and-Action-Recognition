# action_det/scripts/_dataset.py
import json, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as TV

# fixed order of heads
CATS = ["atomic", "simple-context", "complex-context", "communicative", "transportive"]


def _cxcywh_to_xywh(box):
    # box: [cx, cy, w, h] -> [x, y, w, h] (top-left)
    cx, cy, w, h = map(float, box)
    return [cx - w / 2.0, cy - h / 2.0, w, h]


def _ensure_xywh_list(item):
    """
    Returns a list of per-frame top-left xywh boxes aligned with item['frames'].
    Supports several possible field names produced by different indexers.
    Priority:
      1) 'bboxes_xywh'       : list of [x,y,w,h] per frame
      2) 'bboxes_cxcywh'     : list of [cx,cy,w,h] per frame (convert)
      3) 'bbox_xywh'/'bbox'  : single [x,y,w,h] for all frames
      4) 'bbox_cxcywh'       : single [cx,cy,w,h] for all frames (convert)
    """
    if "bboxes_xywh" in item:
        return [list(map(float, b)) for b in item["bboxes_xywh"]]

    if "bboxes_cxcywh" in item:
        return [_cxcywh_to_xywh(b) for b in item["bboxes_cxcywh"]]

    if "bbox_xywh" in item:
        b = list(map(float, item["bbox_xywh"]))
        return [b] * len(item["frames"])

    if "bbox" in item:  # assume top-left xywh if present under 'bbox'
        b = list(map(float, item["bbox"]))
        return [b] * len(item["frames"])

    if "bbox_cxcywh" in item:
        b = _cxcywh_to_xywh(item["bbox_cxcywh"])
        return [b] * len(item["frames"])

    # Last resort: try 'boxes' or 'bboxes' if they exist
    for k in ("boxes", "bboxes"):
        if k in item:
            seq = item[k]
            if len(seq) and len(seq[0]) == 4:
                return [list(map(float, b)) for b in seq]

    raise KeyError(
        "No bbox field found. Expected one of "
        "['bboxes_xywh','bboxes_cxcywh','bbox_xywh','bbox','bbox_cxcywh','boxes','bboxes']"
    )


def item_has_rare_label(item, rare_classes):
    labels = item.get("labels", {})
    for head, rares in rare_classes.items():
        lbl = labels.get(head, "none")
        if lbl in rares:
            return True
    return False


def rebalance_items(items, rare_classes, majority_ratio=3.0, seed=0):
    rng = random.Random(seed)

    rare_items = []
    majority_items = []

    for it in items:
        if item_has_rare_label(it, rare_classes):
            rare_items.append(it)
        else:
            majority_items.append(it)

    print(f"[balance] rare={len(rare_items)}, majority={len(majority_items)}")

    if not rare_items:
        # fallback, nothing to do
        return items

    max_majority = int(majority_ratio * len(rare_items))
    if len(majority_items) > max_majority:
        majority_items = rng.sample(majority_items, max_majority)

    balanced = rare_items + majority_items
    rng.shuffle(balanced)
    print(f"[balance] -> total={len(balanced)} (maj:rare ~ {len(majority_items)}:{len(rare_items)})")
    return balanced


class TubeDataset(Dataset):
    def __init__(self, jsonl_path, label_space_json, img_size=112,
                 train=False, rare_classes=None):
        self.items = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines()]
        self.train = train

        meta = json.loads(Path(label_space_json).read_text())
        self.label_space = meta["label_space"]
        self.T = meta["T"]
        self.stride = meta["stride"]

        # --- Rebalance + duplicate rare tubes with a forced flip clone ---
        if train and rare_classes is not None:
            balanced = rebalance_items(self.items, rare_classes, majority_ratio=3.0)
            new_items = []
            dup_count = 0
            for it in balanced:
                is_rare = item_has_rare_label(it, rare_classes)

                # base copy
                base = dict(it)
                base["is_rare"] = is_rare
                base["force_flip"] = False
                new_items.append(base)

                # mirrored copy for every rare tube
                if is_rare:
                    aug = dict(it)
                    aug["is_rare"] = True
                    aug["force_flip"] = True
                    new_items.append(aug)
                    dup_count += 1

            self.items = new_items
            print(f"[augment] duplicated {dup_count} rare items with force_flip=True")
        else:
            for it in self.items:
                it["is_rare"] = False
                it["force_flip"] = False

        # --- Transforms ---
        self.resize = TV.Resize((img_size, img_size))
        self.to_tensor = TV.ToTensor()
        self.normalize = TV.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989],
        )

        # stronger aug used only for rare tubes
        self.cj = TV.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        )
        self.affine = TV.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
        )

        self.l2i = {
            cat: {l: i for i, l in enumerate(self.label_space[cat])}
            for cat in CATS
        }

    def __len__(self):
        return len(self.items)

    def _augment_tube(self, crops, is_rare=False, force_flip=False):
        """
        crops: list of PIL images (already person-cropped)
        Returns a tensor of shape (T, C, H, W).
        All random aug is sampled once and applied consistently across time.
        """
        # resize + to_tensor per frame, then stack into (T,C,H,W)
        tensors = [self.to_tensor(self.resize(im)) for im in crops]
        batch = torch.stack(tensors, dim=0)  # (T, C, H, W)

        if not self.train:
            # no random aug for val/test, just normalize
            return self.normalize(batch)

        # stronger aug for rare tubes (shared parameters across all frames)
        if is_rare:
            batch = self.cj(batch)
            batch = self.affine(batch)

        # horizontal flip at tube level
        do_flip = force_flip or (not force_flip and random.random() < 0.5)
        if do_flip:
            # flip along width dimension
            batch = torch.flip(batch, dims=[3])

        batch = self.normalize(batch)
        return batch

    def __getitem__(self, idx):
        it = self.items[idx]

        frame_paths = it["frames"]
        boxes_xywh = _ensure_xywh_list(it)
        assert len(frame_paths) == len(boxes_xywh), \
            f"frames ({len(frame_paths)}) and bboxes ({len(boxes_xywh)}) length mismatch"

        is_rare = it.get("is_rare", False)
        force_flip = it.get("force_flip", False)

        # load & crop all frames first (PIL images)
        crops = []
        for p, (x, y, w, h) in zip(frame_paths, boxes_xywh):
            im = Image.open(p).convert("RGB")
            W, H = im.size

            # clamp crop to image bounds
            tlx = max(0, int(np.floor(x)))
            tly = max(0, int(np.floor(y)))
            brx = min(W, int(np.ceil(x + w)))
            bry = min(H, int(np.ceil(y + h)))

            # handle degenerate boxes by falling back to full image
            if brx <= tlx or bry <= tly:
                crop = im
            else:
                crop = im.crop((tlx, tly, brx, bry))
            crops.append(crop)

        # apply tube-level augmentation
        batch = self._augment_tube(crops, is_rare=is_rare, force_flip=force_flip)  # (T, C, H, W)

        # (T, C, H, W) -> (C, T, H, W)
        tube = batch.permute(1, 0, 2, 3).contiguous()

        # targets per head (class indices)
        labels = it.get("labels", {})
        targets = []
        for cat in CATS:
            lbl = labels.get(cat, "none")
            targets.append(self.l2i[cat].get(lbl, self.l2i[cat]["none"]))
        targets = torch.tensor(targets, dtype=torch.long)

        return tube, targets
