"""
train.py - Training and inference entry point for DAB-DETR digit detection.

Usage:
    Train V1 : python train.py --version v1
    Train V2 : python train.py --version v2
    Resume   : python train.py --resume --ckpt best_model.pth --version v1
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

from datasets import DigitDataset, TestDataset, collate_fn
from models import DABDETR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ─────────────────────────── Global Config ────────────────────────────
DATA_DIR = "nycu-hw2-data"
NUM_CLS = 10

# Model architecture configs (Shared across versions)
NUM_QUERIES = 100
D_MODEL = 256
NHEAD = 8
ENC_LAYERS = 4
DEC_LAYERS = 4
DIM_FF = 1024

# Training hyperparameters (Shared)
EPOCHS = 50
BATCH_SIZE = 8
LR_BACKBONE = 1e-6
LR_MAIN = 1e-5
WEIGHT_DECAY = 1e-4
LR_DROP = 40
GRAD_CLIP = 0.1
AUX_WEIGHT = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# ─── Version-Specific Globals (Overwritten by set_version_config) ───
VERSION = "v1"
IMG_SIZE = 256
CLS_WEIGHT = 1.0
BBOX_WEIGHT = 5.0
GIOU_WEIGHT = 2.0
NO_OBJ_COEF = 0.2
MATCH_CLS = 1.0
MATCH_L1 = 5.0
MATCH_GIOU = 2.0
INFER_THRESHOLD = 0.3


def set_version_config(version):
    global VERSION, IMG_SIZE, CLS_WEIGHT, BBOX_WEIGHT, GIOU_WEIGHT, NO_OBJ_COEF
    global MATCH_CLS, MATCH_L1, MATCH_GIOU, INFER_THRESHOLD

    VERSION = version
    if version == 'v1':
        IMG_SIZE = 256
        CLS_WEIGHT = 1.0
        BBOX_WEIGHT = 5.0
        GIOU_WEIGHT = 2.0
        NO_OBJ_COEF = 0.2
        MATCH_CLS = 1.0
        MATCH_L1 = 5.0
        MATCH_GIOU = 2.0
        INFER_THRESHOLD = 0.3
    elif version == 'v2':
        IMG_SIZE = 320
        CLS_WEIGHT = 1.0
        BBOX_WEIGHT = 3.0
        GIOU_WEIGHT = 4.0
        NO_OBJ_COEF = 0.2
        MATCH_CLS = 2.0
        MATCH_L1 = 2.0
        MATCH_GIOU = 5.0
        INFER_THRESHOLD = 0.3


# ──────────────────────── Box Utilities ────────────────────────
def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_iou(boxes1, boxes2):
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(0)
             * (boxes1[:, 3] - boxes1[:, 1]).clamp(0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(0)
             * (boxes2[:, 3] - boxes2[:, 1]).clamp(0))
    ix1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    iy1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    ix2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    iy2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(1e-6)
    ex1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    ey1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    ex2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    ey2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    enc = (ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0)
    return iou - (enc - union) / enc.clamp(1e-6)


# ──────────────────── Hungarian Matcher ────────────────────────
@torch.no_grad()
def hungarian_match(pred_logits, pred_boxes, targets):
    B, Q, _ = pred_logits.shape
    indices = []
    for b in range(B):
        gt_boxes = targets[b]["boxes"].to(pred_boxes.device)
        gt_labels = targets[b]["labels"].to(pred_logits.device)
        if len(gt_boxes) == 0:
            indices.append(
                (torch.tensor(
                    [], dtype=torch.long), torch.tensor(
                    [], dtype=torch.long)))
            continue

        probs = pred_logits[b].softmax(-1)
        cost_cls = -probs[:, gt_labels]
        cost_l1 = torch.cdist(pred_boxes[b], gt_boxes, p=1)
        cost_giou = - \
            generalized_iou(
                box_cxcywh_to_xyxy(
                    pred_boxes[b]),
                box_cxcywh_to_xyxy(gt_boxes))

        cost = (
            MATCH_CLS *
            cost_cls +
            MATCH_L1 *
            cost_l1 +
            MATCH_GIOU *
            cost_giou).cpu().numpy()
        r, c = linear_sum_assignment(cost)
        indices.append(
            (torch.tensor(
                r, dtype=torch.long), torch.tensor(
                c, dtype=torch.long)))
    return indices


# ──────────────────────── Loss Functions ──────────────────────
def loss_for_output(pred_logits, pred_boxes, targets, indices):
    B, Q, _ = pred_logits.shape

    tgt_cls = torch.full((B, Q), NUM_CLS, dtype=torch.long, device=DEVICE)
    for b, (pi, gi) in enumerate(indices):
        if len(pi):
            tgt_cls[b, pi] = targets[b]["labels"].to(DEVICE)[gi]

    cls_w = torch.ones(NUM_CLS + 1, device=DEVICE)
    cls_w[-1] = NO_OBJ_COEF
    loss_cls = F.cross_entropy(pred_logits.reshape(
        B * Q, -1), tgt_cls.reshape(-1), weight=cls_w)

    loss_l1 = torch.tensor(0.0, device=DEVICE)
    loss_giou = torch.tensor(0.0, device=DEVICE)
    num_boxes = 0
    for b, (pi, gi) in enumerate(indices):
        if len(pi) == 0:
            continue
        pb = pred_boxes[b][pi]
        gb = targets[b]["boxes"].to(DEVICE)[gi]
        loss_l1 += F.l1_loss(pb, gb, reduction="sum")
        loss_giou += (1 - generalized_iou(box_cxcywh_to_xyxy(pb),
                      box_cxcywh_to_xyxy(gb)).diag()).sum()
        num_boxes += len(pi)

    num_boxes = max(num_boxes, 1)
    loss_l1 = loss_l1 / num_boxes
    loss_giou = loss_giou / num_boxes
    total = (
        CLS_WEIGHT * loss_cls
        + BBOX_WEIGHT * loss_l1
        + GIOU_WEIGHT * loss_giou
    )
    return total, loss_cls.item(), loss_l1.item(), loss_giou.item()


def compute_loss(outputs, targets):
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    indices = hungarian_match(pred_logits, pred_boxes, targets)
    total, lc, ll, lg = loss_for_output(
        pred_logits, pred_boxes, targets, indices)

    for aux in outputs.get("aux_outputs", []):
        aux_idx = hungarian_match(
            aux["pred_logits"],
            aux["pred_boxes"],
            targets)
        aux_loss, _, _, _ = loss_for_output(
            aux["pred_logits"], aux["pred_boxes"], targets, aux_idx)
        total = total + AUX_WEIGHT * aux_loss

    return total, lc, ll, lg


# ──────────────────────── Train Loop ───────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    total_loss = cls_s = l1_s = giou_s = 0.0
    t0 = time.time()

    for i, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(DEVICE, non_blocking=True)

        with torch.amp.autocast('cuda'):
            outputs = model(imgs)
            loss, lc, ll, lg = compute_loss(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        cls_s += lc
        l1_s += ll
        giou_s += lg

        if (i + 1) % 100 == 0:
            n = i + 1
            print(
                f"  [Ep{epoch} {n}/{len(loader)}] loss={total_loss/n:.4f} "
                f"cls={cls_s/n:.3f} l1={l1_s/n:.3f} giou={giou_s/n:.3f} "
                f"({time.time()-t0:.0f}s)"
            )

    n = len(loader)
    print(
        f"Epoch {epoch}: loss={total_loss/n:.4f} "
        f"cls={cls_s/n:.3f} l1={l1_s/n:.3f} giou={giou_s/n:.3f}"
    )


# ──────────────────── Validation (mAP) ─────────────────────────
def evaluate(model, loader, ann_file):
    model.eval()
    results = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(imgs)
            pred_logits = out["pred_logits"].float()
            pred_boxes = out["pred_boxes"].float()

            if VERSION == 'v1':
                for b, tgt in enumerate(targets):
                    H, W = tgt["orig_size"]
                    probs = pred_logits[b].softmax(-1)[:, :-1]
                    scores, labels = probs.max(-1)
                    keep = scores > INFER_THRESHOLD
                    bx = pred_boxes[b][keep].cpu()
                    sc = scores[keep].cpu()
                    lb = labels[keep].cpu()
                    cx, cy, w, h = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
                    x_coords = (cx - w / 2) * W
                    y_coords = (cy - h / 2) * H
                    widths = w * W
                    heights = h * H
                    for s, l, xv, yv, wv, hv in zip(
                        sc, lb, x_coords, y_coords,
                        widths, heights
                    ):
                        results.append({
                            "image_id": int(tgt["image_id"]),
                            "category_id": int(l) + 1,
                            "bbox": [float(xv),
                                     float(yv),
                                     float(wv),
                                     float(hv)],
                            "score": float(s),
                        })

            elif VERSION == 'v2':
                prob = pred_logits.softmax(-1)
                scores, labels = prob[..., :-1].max(-1)

                for i, target in enumerate(targets):
                    scale, pad_x, pad_y = target["letterbox_params"]
                    if isinstance(scale, torch.Tensor):
                        scale = scale.item()
                    if isinstance(pad_x, torch.Tensor):
                        pad_x = pad_x.item()
                    if isinstance(pad_y, torch.Tensor):
                        pad_y = pad_y.item()

                    b_boxes = pred_boxes[i]
                    cx, cy, w, h = b_boxes.unbind(-1)

                    x1 = (cx - 0.5 * w) * IMG_SIZE
                    y1 = (cy - 0.5 * h) * IMG_SIZE
                    x2 = (cx + 0.5 * w) * IMG_SIZE
                    y2 = (cy + 0.5 * h) * IMG_SIZE

                    x1 = (x1 - pad_x) / scale
                    y1 = (y1 - pad_y) / scale
                    x2 = (x2 - pad_x) / scale
                    y2 = (y2 - pad_y) / scale

                    topk = min(100, scores[i].shape[0])
                    sc, idx = scores[i].topk(topk)
                    lbl = labels[i][idx]
                    bx1, by1, bx2, by2 = x1[idx], y1[idx], x2[idx], y2[idx]

                    img_id = target["image_id"]
                    img_id = int(
                        img_id.item()) if isinstance(
                        img_id, torch.Tensor) else int(img_id)

                    for s, l, xx1, yy1, xx2, yy2 in zip(
                            sc, lbl, bx1, by1, bx2, by2):
                        if s.item() > INFER_THRESHOLD:  # V2 是 0.05
                            results.append({
                                "image_id": img_id,
                                "category_id": int(l.item()) + 1,
                                "bbox": [float(xx1),
                                         float(yy1),
                                         float(xx2 - xx1),
                                         float(yy2 - yy1)],
                                "score": float(s)
                            })

    if not results:
        print("No predictions above threshold.")
        return 0.0

    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats[0]


# ──────────────────────── Inference ────────────────────────────
def infer(model, test_dir, out_path="pred.json"):
    model.eval()
    loader = DataLoader(TestDataset(test_dir, version=VERSION), batch_size=8,
                        num_workers=4, pin_memory=True,
                        persistent_workers=True)
    results = []

    with torch.no_grad():
        for batch in loader:
            if VERSION == 'v1':
                imgs, img_ids, Hs, Ws = batch
            else:
                imgs, img_ids, Hs, Ws, scales, pad_xs, pad_ys = batch

            imgs = imgs.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(imgs)
            pred_logits = out["pred_logits"].float()
            pred_boxes = out["pred_boxes"].float()

            if VERSION == 'v1':
                for b in range(len(img_ids)):
                    H, W = Hs[b].item(), Ws[b].item()
                    probs = pred_logits[b].softmax(-1)[:, :-1]
                    scores, labels = probs.max(-1)
                    keep = scores > INFER_THRESHOLD
                    bx = pred_boxes[b][keep].cpu()
                    sc = scores[keep].cpu()
                    lb = labels[keep].cpu()
                    cx, cy, w, h = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
                    for s, l, xv, yv, wv, hv in zip(
                            sc, lb, (cx - w / 2) * W,
                            (cy - h / 2) * H,
                            w * W, h * H
                    ):
                        results.append({
                            "image_id": int(img_ids[b]),
                            "category_id": int(l) + 1,
                            "bbox": [float(xv),
                                     float(yv),
                                     float(wv),
                                     float(hv)],
                            "score": float(s),
                        })

            elif VERSION == 'v2':
                probs = pred_logits.softmax(-1)
                scores, labels = probs[..., :-1].max(-1)

                for b in range(len(img_ids)):
                    scale = scales[b].item()
                    pad_x = pad_xs[b].item()
                    pad_y = pad_ys[b].item()

                    b_scores = scores[b]
                    b_labels = labels[b]
                    b_boxes = pred_boxes[b]

                    cx, cy, w, h = b_boxes.unbind(-1)
                    x1 = (cx - 0.5 * w) * IMG_SIZE
                    y1 = (cy - 0.5 * h) * IMG_SIZE
                    x2 = (cx + 0.5 * w) * IMG_SIZE
                    y2 = (cy + 0.5 * h) * IMG_SIZE

                    x1 = (x1 - pad_x) / scale
                    y1 = (y1 - pad_y) / scale
                    x2 = (x2 - pad_x) / scale
                    y2 = (y2 - pad_y) / scale

                    topk = min(100, b_scores.shape[0])
                    sc, idx = b_scores.topk(topk)
                    lbl = b_labels[idx]
                    bx1, by1, bx2, by2 = x1[idx], y1[idx], x2[idx], y2[idx]

                    img_id = int(img_ids[b].item())

                    for s, l, xx1, yy1, xx2, yy2 in zip(
                            sc, lbl, bx1, by1, bx2, by2):
                        if s.item() > INFER_THRESHOLD:
                            results.append({
                                "image_id": img_id,
                                "category_id": int(l.item()) + 1,
                                "bbox": [float(xx1),
                                         float(yy1),
                                         float(xx2 - xx1),
                                         float(yy2 - yy1)],
                                "score": float(s.item())
                            })

    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} predictions -> {out_path}")


# ──────────────────────────── Main ─────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", default="best_model234.pth")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--version", type=str, default="v1",
                        choices=["v1", "v2"])
    args = parser.parse_args()

    set_version_config(args.version)
    print(f"Running mode: {VERSION.upper()}")

    model = DABDETR(
        num_classes=NUM_CLS,
        num_queries=NUM_QUERIES,
        d_model=D_MODEL,
        nhead=NHEAD,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        dim_ff=DIM_FF,
        dropout=0.1,
    ).to(DEVICE)

    if args.infer:
        ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        infer(model, os.path.join(DATA_DIR, "test"))
        return

    # ── Load datasets ──
    train_ds = DigitDataset(os.path.join(DATA_DIR, "train"),
                            os.path.join(DATA_DIR, "train.json"),
                            train=True, version=VERSION)
    val_ds = DigitDataset(os.path.join(DATA_DIR, "valid"),
                          os.path.join(DATA_DIR, "valid.json"),
                          train=False, version=VERSION)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True,
                            persistent_workers=True, collate_fn=collate_fn)

    # ── Optimizer ──
    backbone_ids = ({id(p) for p in model.backbone.parameters()} | {
                    id(p) for p in model.input_proj.parameters()})
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if id(
                p) in backbone_ids], "lr": LR_BACKBONE},
            {"params": [p for p in model.parameters() if id(
                p) not in backbone_ids], "lr": LR_MAIN},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_DROP, gamma=0.1)

    # ── AMP scaler ──
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 1
    best_map = 0.0

    if args.resume and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']} (best mAP={best_map:.4f})")

    val_ann = os.path.join(DATA_DIR, "valid.json")

    for epoch in range(start_epoch, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        scheduler.step()

        if epoch % 1 == 0 or epoch == args.epochs:
            mAP = evaluate(model, val_loader, val_ann)
            print(f"  -> Val mAP@[.5:.95] = {mAP:.4f}")
            if mAP > best_map:
                best_map = mAP
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_map": best_map,
                }, args.ckpt)
                print(f"  Saved best model (mAP={best_map:.4f})")

    print(f"\nDone. Best val mAP = {best_map:.4f}")


if __name__ == "__main__":
    main()
