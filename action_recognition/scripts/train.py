#!/usr/bin/env python3
import argparse, json, time, csv
from pathlib import Path
import math
import random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from torch import autocast
from tqdm import tqdm
import torch.nn.functional as F
from _dataset import TubeDataset, CATS  # CATS = ["atomic", "simple-context", ...]
from _sampler import build_weighted_sampler


# Heads we consider "hard" and will apply focal loss to
FOCAL_HEADS = {"complex-context", "communicative", "transportive"}

# Default focal gamma. 0.0 means "disabled"
FOCAL_GAMMA = 1.5  # you can tune: 1.0, 1.5, 2.0 ...


def focal_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss on top of standard (weighted) cross-entropy.
    Returns a scalar mean over the batch.
    """
    # log-softmax and probabilities
    logp = F.log_softmax(logits.to(torch.float32), dim=1)        # (B, C)
    p = logp.exp()                                               # (B, C)

    # CE per sample (reduction='none')
    ce = F.nll_loss(
        logp,
        targets,
        weight=class_weights,
        reduction="none",
    )  # (B,)

    # p_t = probability of the true class for each sample
    idx = torch.arange(targets.numel(), device=targets.device)
    pt = p[idx, targets]                                         # (B,)

    # focal weight
    focal_weight = (1.0 - pt).pow(gamma)                         # (B,)

    # apply and average
    loss = (focal_weight * ce).mean()
    return loss


# -------------------------
# Model / loss / metrics
# -------------------------
def build_model(label_space):
    backbone = r3d_18(weights="KINETICS400_V1")  # torchvision >= 0.15
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    hidden_dim = 256
    heads = nn.ModuleDict({
        cat: nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, len(labels)),
        )
        for cat, labels in label_space.items()
    })

    class MultiHead(nn.Module):
        def __init__(self, bb, hd):
            super().__init__()
            self.bb = bb
            self.hd = hd
        def forward(self, x):
            f = self.bb(x)  # (N, feat)
            return {k: self.hd[k](f) for k in self.hd.keys()}

    return MultiHead(backbone, heads)

def _inv_freq_weights(counts, power=1.0, eps=1.0):
    """
    counts: list[int] per-class counts in label order
    returns a torch.tensor of normalized inverse-frequency weights
    """
    import numpy as np
    c = np.asarray(counts, dtype=float)
    c = np.maximum(c, eps)
    w = (c.sum() / c) ** power          # inverse-frequency^power
    w = w / w.sum() * len(c)            # normalize around 1.0
    return torch.tensor(w, dtype=torch.float32)

def ce_smooth_manual(logits, targets, num_classes: int, eps: float, class_weights=None):
    """
    logits: (N, C)  on device (MPS)
    targets: (N,)   long on device
    Computes cross-entropy with optional label smoothing WITHOUT using CE kernel.
    """
    # ensure fp32 math on MPS
    logp = F.log_softmax(logits.to(torch.float32), dim=1)

    if eps > 0.0:
        with torch.no_grad():
            true_dist = torch.zeros_like(logp).scatter_(1, targets.unsqueeze(1), 1)
            # smooth the 1-hot
            true_dist = true_dist * (1.0 - eps) + eps / float(num_classes)
        # KLDivLoss-style CE: -sum p * logq
        loss = torch.sum(-true_dist * logp, dim=1)
        if class_weights is not None:
            loss = loss * class_weights.to(logp.device)[targets]
        loss = loss.mean()
    else:
        loss = F.nll_loss(logp, targets, weight=class_weights)
    return loss

def build_ce_by_head(train_label_report, label_space, label_smooth, power=0.5):
    ce_cfg = {}
    for head in CATS:
        classes = label_space[head]               # class order used in dataset/model
        cls_counts = train_label_report[head]     # dict[class_name -> count]
        counts = [cls_counts.get(lbl, 0) for lbl in classes]
        w = _inv_freq_weights(counts, power=power)
        ce_cfg[head] = (len(classes), label_smooth, w)
    return ce_cfg

def multitask_loss(outputs, targets, ce_cfg_by_head, log=False):
    """
    outputs: dict of head -> logits (N, C)
    targets: (N, num_heads)
    ce_cfg_by_head: dict[head] -> (num_classes, label_smooth)
    """
    
    loss = 0.0
    losses = {}
    for i, head in enumerate(CATS):
        num_classes, eps, w_cls = ce_cfg_by_head[head]
        logits = outputs[head]
        t = targets[:, i]
        if head in FOCAL_HEADS and FOCAL_GAMMA > 0.0:
            # For focal heads, we ignore label smoothing for now (eps=0)
            li = focal_ce_loss(
                logits,
                t,
                class_weights=w_cls.to(logits.device) if w_cls is not None else None,
                gamma=FOCAL_GAMMA,
            )
        else:
            li = ce_smooth_manual(logits, t, num_classes, eps, w_cls.to(logits.device))

        losses[head] = li.detach().item()
        loss = loss + li
    loss = loss / len(CATS)
    return loss, losses


def accuracy(outputs, targets):
    acc = {}
    with torch.no_grad():
        for i, cat in enumerate(CATS):
            pred = outputs[cat].argmax(dim=1)
            acc[cat] = (pred == targets[:, i]).float().mean().item()
    return acc


# -------------------------
# Utils
# -------------------------
def get_device(arg_dev: str) -> torch.device:
    if arg_dev == "cuda" and torch.cuda.is_available():
        print("[INFO] Using CUDA device")   
        return torch.device("cuda")
    if arg_dev == "mps" and torch.backends.mps.is_available():
        print("[INFO] Using MPS device")
        return torch.device("mps")
    print("[INFO] Using CPU device")
    return torch.device("cpu")


def init_csv(log_csv: Path):
    header = (
        ["epoch", "train_loss", "val_loss", "avg_acc"] 
        + [f"{c}_acc" for c in CATS] 
        + ["macro_mAP"]
        + [f"{c}_mAP" for c in CATS] 
        + ["best_val_loss", "lr", "wall_time_s"]
    )
    new_file = not log_csv.exists()
    if new_file:
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        with log_csv.open("w", newline="") as f:
            csv.writer(f).writerow(header)
    return header


def append_csv(log_csv: Path, row: dict, header: list[str]):
    with log_csv.open("a", newline="") as f:
        csv.writer(f).writerow([row.get(k, "") for k in header])


def save_ckpt(path: Path, epoch: int, model: nn.Module, opt: torch.optim.Optimizer, best_val_loss: float, label_space: dict):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "best_val_loss": best_val_loss,
        "label_space": label_space
    }, str(path))


def try_resume(run_dir: Path, model: nn.Module, opt: torch.optim.Optimizer, device: torch.device):
    ckpt_path = run_dir / "last.pt"
    if not ckpt_path.exists():
        return 1, float("inf")  # start_epoch, best_val_loss
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    start_epoch = int(ckpt.get("epoch", 1)) + 1
    print(f"[RESUME] Loaded {ckpt_path} (epoch {start_epoch-1}, best_val_loss={best_val_loss:.4f})")
    return start_epoch, best_val_loss

# -------------------------
# Calculate mAP
# -------------------------
def _softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

def _pr_points(y_true_bin, y_score):
    order = np.argsort(-y_score)
    y = y_true_bin[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    denom = np.maximum(tp + fp, 1)
    precision = tp / denom
    recall = tp / max(int((y == 1).sum()), 1)
    # endpoints for nicer curves
    precision = np.concatenate(([1.0], precision, [0.0]))
    recall    = np.concatenate(([0.0], recall,    [1.0]))
    return precision, recall

def _average_precision(y_true_bin, y_score):
    P, R = _pr_points(y_true_bin, y_score)
    # precision envelope
    for i in range(P.size - 1, 0, -1):
        P[i - 1] = max(P[i - 1], P[i])
    idx = np.where(R[1:] != R[:-1])[0]
    return float(np.sum((R[idx + 1] - R[idx]) * P[idx + 1]))


# Class balancing
def load_rare_classes(report_path, freq_thresh=0.15):
    """
    Return dict[head] -> set of rare class names (excluding 'none').

    freq_thresh: classes with fraction <= freq_thresh are considered rare.
    """
    report = json.loads(Path(report_path).read_text())
    rare = {}
    for head in CATS:
        counts = report[head]
        total = sum(counts.values())
        rare_head = set()
        for cls, c in counts.items():
            if cls == "none":
                continue
            frac = c / total
            if frac <= freq_thresh:
                rare_head.add(cls)
        rare[head] = rare_head
    return rare

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--run-name", type=str, default=None, help="optional override for run dir name")
    ap.add_argument("--no-resume", action="store_true", help="force fresh training, ignore last.pt")
    ap.add_argument("--clip-grad", type=float, default=4.0)
    ap.add_argument("--sched", type=str, default="cosine", choices=["cosine", "none"])
    ap.add_argument("--warmup-epochs", type=int, default=4)
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--rare-boost", type=float, default=2.5, help=">1.0 oversamples rare labels")

    args = ap.parse_args()

    split_dir = Path(f"action_det/data/splits_pct_{int(args.pct*100)}_T{args.T}_S{args.stride}")
    label_space_json = split_dir / "label_space.json"
    train_label_report_json = split_dir / "train_label_report.json"
    assert label_space_json.exists(), "Run 01_build_index.py first (it writes label_space.json)"

    meta = json.loads(label_space_json.read_text())
    label_space = meta["label_space"]

    train_label_space = json.loads(train_label_report_json.read_text())


    # Get rare classes from training set report
    freq_thresh = 0.15
    rare_classes = load_rare_classes(train_label_report_json, freq_thresh=freq_thresh)
    print(f"[INFO] Rare classes per head (freq <= {freq_thresh*100:.0f}%):")
    for head in CATS:
        print(f"  {head}: {rare_classes[head]}")


    # --- Sanity check: number of classes from label_space vs train_label_space ---
    for head in CATS:
        print(
            f"[DEBUG] head={head}: "
            f"num_classes(label_space)={len(label_space[head])}, "
            f"num_classes(train_label_space)={len(train_label_space[head])}"
        )

    # build criterions once; keep on device later if you want (PyTorch handles CPU weights fine)
    ce_by_head = build_ce_by_head(train_label_space, label_space, args.label_smooth)

    train_jsonl = split_dir / "train.jsonl"
    val_jsonl   = split_dir / "val.jsonl"
    assert train_jsonl.exists() and val_jsonl.exists(), "Train/val JSONL files not found."

    use_workers = max(0, args.workers)
    prefetch = 2 if use_workers > 0 else None

    ds_tr = TubeDataset(train_jsonl, label_space_json, img_size=args.img_size, train=True, rare_classes=rare_classes)
    ds_va = TubeDataset(val_jsonl,   label_space_json, img_size=args.img_size)

    train_sampler=None
    if args.rare_boost > 1.0:
        train_sampler = build_weighted_sampler(
            ds_tr.items, 
            label_space_json, 
            repeat_factor=args.rare_boost,
            cap=20.0
        )
        print(f"[INFO] Using WeightedRandomSampler with repeat_factor={args.rare_boost}")

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=use_workers,
        pin_memory=False,
        prefetch_factor=(prefetch if use_workers > 0 else None),
        persistent_workers=False,
        timeout=0,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch,
        shuffle=False,
        num_workers=use_workers,
        pin_memory=False,
        prefetch_factor=(prefetch if use_workers > 0 else None),
        persistent_workers=False,
        timeout=0,
    )

    device = get_device(args.device)
    model = build_model(label_space).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)

    sched = None
    if args.sched == "cosine":
        total_steps = args.epochs * math.ceil(len(dl_tr))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)


    freeze_epochs = args.warmup_epochs
    if freeze_epochs > 0:
        for p in model.bb.parameters():
            p.requires_grad = False
    
    

    # run dir, logs, resume
    default_name = f"r3d18_pct{int(args.pct*100)}_T{args.T}_S{args.stride}_img{args.img_size}_b{args.batch}"
    if args.rare_boost > 1.0:
        default_name += f"_rare{args.rare_boost}_no_s"
    run_dir = Path("action_det/runs") / (args.run_name or default_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/"meta.json").write_text(json.dumps({
        "pct": args.pct, "T": args.T, "stride": args.stride,
        "img_size": args.img_size, "batch": args.batch, "epochs": args.epochs,
        "label_space": label_space
    }, indent=2))

    log_csv = run_dir / "results.csv"
    header = init_csv(log_csv)

    if args.no_resume:
        start_epoch, best_val_loss = 1, float("inf")
        print("[INFO] Starting fresh (no-resume).")
    else:
        start_epoch, best_val_loss = try_resume(run_dir, model, opt, device)

    epochs_no_improve = 0 if start_epoch == 1 else 0  # reset counter on resume

    # --------------- Training loop ---------------
    num_epochs = args.epochs

        
    print(f"[INFO] Starting training for {num_epochs} epochs from epoch {start_epoch}...")
    for epoch in range(start_epoch, num_epochs + 1):
        wall_t0 = time.time()
        model.train()
        if epoch == freeze_epochs + 1:
            for p in model.bb.parameters():
                p.requires_grad = True  # unfreeze backbone
        tr_loss_sum, ntr = 0.0, 0

        pbar = tqdm(dl_tr, desc=f"Epoch {epoch}/{num_epochs} [train]", ncols=100)
        for bidx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)  # x: (N,C,T,H,W)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=str(device), dtype=torch.float16):
                out = model(x)
                loss, losses = multitask_loss(out, y, ce_by_head)
                
            loss.backward()
            if args.clip_grad > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            if sched:
                sched.step()

            tr_loss_sum += loss.item() * x.size(0)
            ntr += x.size(0)
            pbar.set_postfix(loss=f"{tr_loss_sum/max(1,ntr):.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

            # if bidx == 120: # limit number of batches per epoch for faster debugging
            #     break

        train_loss = tr_loss_sum / max(1, ntr)

        # -------- Validation --------
        
        model.eval()
        with torch.no_grad():
            va_loss_sum, nva = 0.0, 0
            acc_sum = {c: 0.0 for c in CATS}
            cnt = 0

            # collect logits/targets for mAP
            all_logits = {c: [] for c in CATS}
            all_targets = {c: [] for c in CATS}

            pbar_v = tqdm(dl_va, desc=f"Epoch {epoch}/{args.epochs} [val]  ", ncols=100)
            for x, y in pbar_v:
                x, y = x.to(device), y.to(device)
                with autocast(device_type=str(device), dtype=torch.float16):
                    out = model(x)
                    loss, _ = multitask_loss(out, y, ce_by_head) 
                va_loss_sum += loss.item() * x.size(0)
                nva += x.size(0)

                # --- Sanity debug: show a few true vs predicted labels on first val batch ---
                if epoch == start_epoch and cnt == 0:
                    for i_head, head in enumerate(CATS):
                        preds = out[head].argmax(dim=1)
                        print(f"[DEBUG] val head={head} true[:8] = {y[:8, i_head].tolist()}")
                        print(f"[DEBUG] val head={head} pred[:8] = {preds[:8].tolist()}")

                a = accuracy(out, y)
                for k in CATS:
                    acc_sum[k] += a[k]
                cnt += 1

                # stach logits/targets for mAP
                for k in CATS:
                    all_logits[k].append(out[k].detach().cpu().numpy())
                    all_targets[k].append(y[:, CATS.index(k)].detach().cpu().numpy())
                pbar_v.set_postfix(loss=f"{va_loss_sum/max(1,nva):.4f}")

            val_loss = va_loss_sum / max(1, nva)
            acc_avg = {k: acc_sum[k] / max(1, cnt) for k in CATS}
            score = sum(acc_avg.values()) / len(CATS)  # avg head accuracy

            # ---- per-head mAP (exclude 'none' if present) ----
            head_maps = {}
            for k in CATS:
                targets = np.concatenate(all_targets[k], axis=0).astype(int)
                classes = label_space[k]
                idx_none = classes.index("none") if "none" in classes else -1
                cls_indices = [i for i in range(len(classes)) if i != idx_none]

                # Normal path: use model logits
                logits = np.concatenate(all_logits[k], axis=0)
                probs = _softmax_np(logits)

                ap_vals = []
                for ci in cls_indices:
                    y_bin = (targets == ci).astype(int)
                    cls_scores = probs[:, ci]
                    if y_bin.sum() == 0:   # no positives for this class in val
                        continue
                    ap_vals.append(_average_precision(y_bin, cls_scores))

                head_maps[k] = float(np.mean(ap_vals)) if ap_vals else 0.0
            macro_map = float(np.mean(list(head_maps.values()))) if head_maps else 0.0

        # console summary
        acc_str = " ".join([f"{k[:4]}_acc {acc_avg[k]:.3f}" for k in CATS])
        map_str = " ".join([f"{k[:4]}_mAP {head_maps[k]:.3f}" for k in CATS])
        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | best_val {best_val_loss:.4f} | "
              f"{acc_str} | avg_acc {score:.3f} | {map_str} | macro_mAP {macro_map:.3f}")

        # save last + maybe best; write CSV row
        save_ckpt(run_dir/f"epoch{epoch}.pt", epoch, model, opt, best_val_loss, label_space)
        save_ckpt(run_dir/"last.pt", epoch, model, opt, best_val_loss, label_space)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(run_dir/"best.pt", epoch, model, opt, best_val_loss, label_space)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        append_csv(log_csv, {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "best_val_loss": f"{best_val_loss:.6f}",
            "avg_acc": f"{score:.6f}",
            **{f"{k}_acc": f"{acc_avg[k]:.6f}" for k in CATS},
            "macro_mAP": f"{macro_map:.6f}",
            **{f"{k}_mAP": f"{head_maps[k]:.6f}" for k in CATS},
            "lr": f"{opt.param_groups[0]['lr']:.6e}",
            "wall_time_s": f"{time.time() - wall_t0:.2f}",
        }, header)

        if epochs_no_improve >= args.patience:
            print(f"[EarlyStop] no improvement for {args.patience} epochs (best={best_val_loss:.4f})")
            break

    print(f"[OK] Run complete -> {run_dir}")


if __name__ == "__main__":
    main()
