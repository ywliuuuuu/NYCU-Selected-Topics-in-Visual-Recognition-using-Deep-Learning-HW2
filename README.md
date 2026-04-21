# NYCU Visual Recognition using Deep Learning 2026 — Homework 2

* **Student ID**: 313553049
* **Name**: 劉怡妏

## Introduction

This repository contains the implementation for HW2: **Digit Detection** on the Street View House Numbers (SVHN) dataset using **DAB-DETR** (Dynamic Anchor Boxes DETR, Liu et al., ICLR 2022) with a ResNet-50 backbone.

The final submission uses a **two-model weighted box fusion (WBF) ensemble**:

* **Model A (v1)**: Trained with 256×256 stretch resize, ColorJitter augmentation, standard DETR loss weights
* **Model B (v2)**: Trained with 320×320 letterbox resize (aspect-ratio preserving), heavier GIoU loss weighting (`GIOU=4.0`, `MATCH_GIOU=5.0`), lower inference threshold

Key techniques:
* DAB-DETR decoder with dynamic anchor box queries and iterative per-layer anchor refinement
* Auxiliary losses from all intermediate decoder layers (weight = 0.5) for faster convergence
* Differential learning rate: backbone at `1e-6`, transformer/heads at `1e-5`
* Automatic Mixed Precision (AMP) training with `GradScaler`
* Weighted Box Fusion (WBF) ensemble of two independently trained models

## File Structure

```
.
├── train.py           # Training loop, loss, inference entry point
├── models.py          # DAB-DETR model architecture
├── datasets.py        # Dataset, letterbox transform, collate function
├── wbf_ensemble.py    # WBF ensemble of two pred.json files
├── nycu-hw2-data/
│   ├── train/         # Training images
│   ├── valid/         # Validation images
│   ├── test/          # Test images (unlabelled)
│   ├── train.json     # COCO-format training annotations
│   └── valid.json     # COCO-format validation annotations
```

## Environment Setup

### Requirements

* Python 3.9+
* CUDA-compatible GPU (RTX 3060 Ti 8 GB or better recommended)

### Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pycocotools scipy ensemble-boxes pillow numpy
```

## Usage

### 1. Training

**Train Model A (v1) — 256×256 stretch resize:**

```bash
python train.py --version v1 --ckpt best_model_v1.pth
```

**Train Model B (v2) — 320×320 letterbox resize:**

```bash
python train.py --version v2 --ckpt best_model_v2.pth
```

**Resume training from a checkpoint:**

```bash
python train.py --version v1 --resume --ckpt best_model_v1.pth
python train.py --version v2 --resume --ckpt best_model_v2.pth
```

Checkpoints are saved automatically whenever val mAP improves. Each checkpoint includes the full training state (model weights, optimizer, scheduler, AMP scaler, and current best mAP) to allow seamless resumption.

### 2. Inference

Run inference with each model separately to produce two prediction files:

```bash
# Model A inference
python train.py --infer --version v1 --ckpt best_model_v1.pth
mv pred.json pred_log1.json

# Model B inference
python train.py --infer --version v2 --ckpt best_model_v2.pth
mv pred.json pred_log2.json
```

Each `pred.json` is a COCO-format list where every entry contains `image_id`, `category_id`, `bbox` (in `[x, y, w, h]` pixel format), and `score`.

### 3. WBF Ensemble

Merge the two prediction files using Weighted Box Fusion:

```bash
python wbf_ensemble.py
```

This reads `pred_log1.json` and `pred_log2.json`, applies WBF with weights `[2, 1]` (Model A weighted higher), `iou_thr=0.55`, and `skip_box_thr=0.05`, then writes the final `pred.json`.

**To submit**, compress the output:

```bash
zip submission.zip pred.json
```

Upload `submission.zip` to CodaBench. The file inside the zip must be named `pred.json`.

## Hyperparameters

| Parameter | v1 | v2 |
|---|---|---|
| Image size | 256×256 (stretch) | 320×320 (letterbox) |
| Encoder / decoder layers | 4 / 4 | 4 / 4 |
| `d_model` | 256 | 256 |
| `dim_ff` | 1024 | 1024 |
| `LR_BACKBONE` | 1e-6 | 1e-6 |
| `LR_MAIN` | 1e-5 | 1e-5 |
| `LR_DROP` (StepLR epoch) | 40 | 40 |
| `BBOX_WEIGHT` | 5.0 | 3.0 |
| `GIOU_WEIGHT` | 2.0 | 4.0 |
| `NO_OBJ_COEF` | 0.2 | 0.3 |
| `INFER_THRESHOLD` | 0.3 | 0.05 |
| Batch size | 8 | 8 |
| Epochs | 50 | 50 |

## Performance Snapshot

*(Insert leaderboard screenshot here)*
