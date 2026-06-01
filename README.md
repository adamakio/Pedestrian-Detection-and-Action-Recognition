# Pedestrian Detection and Action Recognition

**AER1515 — Perception for Robotics | University of Toronto (UTIAS) | Fall 2025**

A two-stage perception pipeline for autonomous driving: pedestrian detection with fine-tuned YOLO11 followed by multi-label action recognition using a 3D ResNet (R3D18) video backbone — trained and evaluated on the [TITAN dataset](https://usa.honda-ri.com/titan).

---

## Pipeline Overview

```
TITAN video clips
      │
      ▼
┌─────────────────────┐
│  YOLO11n Detection  │  Fine-tuned on TITAN pedestrian annotations
│  (scripts/)         │  with a 10% subset training strategy
└─────────────────────┘
      │  pedestrian bounding boxes + tracks
      ▼
┌─────────────────────┐
│  R3D18 Classifier   │  T=8 frame clips, 5 simultaneous action heads
│  (action_det/)      │  focal loss + rare-class weighted sampling
└─────────────────────┘
      │
      ▼
 Multi-label action predictions per pedestrian
```

---

## Action Categories (TITAN taxonomy)

The action recognition model predicts 5 independent behavioral heads simultaneously:

| Head | Labels |
|------|--------|
| **Atomic** | standing, walking, running, sitting, jumping, ... |
| **Simple context** | crossing legally, jaywalking, waiting to cross, biking, ... |
| **Complex context** | getting in/out of vehicle, loading/unloading, ... |
| **Communicative** | talking on phone, looking at phone, talking in group |
| **Transportive** | carrying, pushing, pulling |

---

## Repository Structure

```
scripts/                    # YOLO pedestrian detection pipeline
├── export_titan_to_yolo.py # Convert TITAN annotations → YOLO format
├── train_yolo.py           # Fine-tune YOLO11n on TITAN
├── eval_yolo.py            # Evaluate detection (mAP, precision, recall)
├── track_person.py         # ByteTrack multi-object tracking
├── generate_frame.py       # Visualize detections on frames
└── ...

action_det/                 # Action recognition pipeline
├── scripts/
│   ├── 01_build_index.py   # Build clip index from TITAN
│   ├── 02_build_label_space.py  # Map TITAN labels to head categories
│   ├── _dataset.py         # TubeDataset — loads T-frame pedestrian clips
│   ├── heads.py            # Action head definitions
│   ├── train.py            # R3D18 training with focal loss + weighted sampling
│   ├── 06_eval.py          # Per-head accuracy, F1, confusion matrices
│   └── ...
├── data/                   # Clip split JSONLs (not included — TITAN license)
└── runs/                   # Checkpoints (tracked via Git LFS)

figs/                       # Detector comparison figures
```

---

## Key Design Choices

- **YOLO11n** (Ultralytics): lightweight detector fine-tuned for pedestrian-only detection on TITAN; trained with a 10% data subset and merged multi-run weights for robustness.
- **R3D18**: 3D ResNet-18 pretrained on Kinetics; fine-tuned with T=8 frames at stride S=4, image size 224×224, batch size 24.
- **Rare-class boosting**: `_sampler.py` builds a weighted sampler that upsamples underrepresented action labels, combined with focal loss (γ=1.5) on the three hardest heads (complex-context, communicative, transportive).
- **Multi-head architecture**: a shared R3D18 backbone with 5 independent linear classification heads, trained jointly.

---

## Presentation

[![AER1515 Final Project Presentation](https://img.youtube.com/vi/CaRRNdxRcNQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=CaRRNdxRcNQ)

---

## Docs

| File | Description |
|------|-------------|
| [`AER1515_FinalReport_ZouhairAdamHamaimou.pdf`](docs/AER1515_FinalReport_ZouhairAdamHamaimou.pdf) | Final report |
| [`AER1515_Initial_Results_ZouhairHamaimou_1004891986.pdf`](docs/AER1515_Initial_Results_ZouhairHamaimou_1004891986.pdf) | Midterm results |
| [`AER1515_Proposal_ZouhairHamaimou_WayneLu.pdf`](docs/AER1515_Proposal_ZouhairHamaimou_WayneLu.pdf) | Project proposal |

---

## Tools

Python · PyTorch · Ultralytics (YOLO11) · torchvision (R3D18) · OpenCV · NumPy · Matplotlib
