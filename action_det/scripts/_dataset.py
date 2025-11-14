# action_det/scripts/_dataset.py
import json, torch
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as TV

# fixed order of heads
CATS = ["atomic", "simple-context", "complex-context", "communicative", "transportive"]

def _cxcywh_to_xywh(box):
    # box: [cx, cy, w, h] -> [x, y, w, h] (top-left)
    cx, cy, w, h = map(float, box)
    return [cx - w/2.0, cy - h/2.0, w, h]

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
        return [ _cxcywh_to_xywh(b) for b in item["bboxes_cxcywh"] ]

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

class TubeDataset(Dataset):
    def __init__(self, jsonl_path, label_space_json, img_size=224, aug=True):
        self.items = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines()]
        meta = json.loads(Path(label_space_json).read_text())
        self.label_space = meta["label_space"]
        self.T = meta["T"]
        self.stride = meta["stride"]
        self.train_mode = "train" in str(jsonl_path).lower()

        base = [TV.Resize((img_size, img_size))]
        if aug and self.train_mode:
            base += [
                TV.ColorJitter(0.1, 0.1, 0.1, 0.05),
                TV.RandomHorizontalFlip(p=0.5),
            ]
        base += [TV.ToTensor()]
        self.to_tensor = TV.Compose(base)

        self.l2i = {cat: {l:i for i,l in enumerate(self.label_space[cat])}
                    for cat in CATS}

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]

        frame_paths = it["frames"]
        boxes_xywh = _ensure_xywh_list(it)
        assert len(frame_paths) == len(boxes_xywh), \
            f"frames ({len(frame_paths)}) and bboxes ({len(boxes_xywh)}) length mismatch"

        imgs = []
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

            imgs.append(self.to_tensor(crop))  # C,H,W

        # (T,C,H,W) -> (C,T,H,W)
        tube = torch.stack(imgs, dim=0).permute(1,0,2,3).contiguous()

        # targets per head (class indices)
        labels = it.get("labels", {})
        targets = []
        for cat in CATS:
            lbl = labels.get(cat, "none")
            targets.append(self.l2i[cat].get(lbl, self.l2i[cat]["none"]))
        targets = torch.tensor(targets, dtype=torch.long)
        
        return tube, targets
