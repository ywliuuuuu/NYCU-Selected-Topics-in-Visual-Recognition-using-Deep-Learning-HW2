"""
datasets.py - Unified Dataset, transforms, and collate for HW2 digit detection.
Supports both original and letterbox configurations for reproducibility.
"""

import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as T_v1
import torchvision.transforms.v2 as T_v2

# ──────────────────── Transform Pipelines ──────────────────────


def make_transforms_v1(train: bool):
    """v1 Image transform pipeline: 256x256 + ColorJitter."""
    IMG_SIZE = 256
    if train:
        return T_v2.Compose([
            T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            T_v2.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1),
            T_v2.ToImage(),
            T_v2.ToDtype(torch.float32, scale=True),
            T_v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return T_v2.Compose([
        T_v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T_v2.ToImage(),
        T_v2.ToDtype(torch.float32, scale=True),
        T_v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_normalize_v2():
    """v2 Image transform pipeline: ToTensor + Normalize only."""
    return T_v1.Compose([
        T_v1.ToTensor(),
        T_v1.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# ──────────────────── Datasets ──────────────────────

class DigitDataset(Dataset):
    """COCO-format dataset for the digit detection task (Unified)."""

    def __init__(self, img_dir: str, ann_file: str,
                 train: bool = True, version: str = 'v1'):
        if version not in ['v1', 'v2']:
            raise ValueError("version must be either 'v1' or 'v2'")

        self.img_dir = Path(img_dir)
        self.train = train
        self.version = version

        with open(ann_file) as f:
            data = json.load(f)

        self.images_list = data["images"]
        self.images_dict = {img["id"]: img for img in data["images"]}
        self.ids = [img["id"] for img in data["images"]]

        self.anns = {img_id: [] for img_id in self.ids}
        for ann in data["annotations"]:
            self.anns[ann["image_id"]].append(ann)

        if self.version == 'v1':
            self.transforms = make_transforms_v1(train)
        elif self.version == 'v2':
            self.img_size = 320
            self.normalize = get_normalize_v2()

    def __len__(self):
        return len(self.ids)

    def letterbox(self, img, boxes):
        w, h = img.size
        scale = min(self.img_size / w, self.img_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        pad_x = (self.img_size - new_w) // 2
        pad_y = (self.img_size - new_h) // 2
        canvas.paste(img_resized, (pad_x, pad_y))

        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            boxes *= scale
            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / self.img_size
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / self.img_size
            bw = (boxes[:, 2] - boxes[:, 0]) / self.img_size
            bh = (boxes[:, 3] - boxes[:, 1]) / self.img_size
            boxes = np.stack([cx, cy, bw, bh], axis=1)

        return canvas, boxes, scale, pad_x, pad_y

    def __getitem__(self, idx):
        if self.version == 'v1':
            img_id = self.ids[idx]
            info = self.images_dict[img_id]
            img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
            W_orig, H_orig = img.size

            boxes, labels = [], []
            for ann in self.anns[img_id]:
                x, y, w, h = ann["bbox"]
                boxes.append([(x + w / 2) / W_orig, (y + h / 2) /
                             H_orig, w / W_orig, h / H_orig])
                labels.append(ann["category_id"] - 1)

            img = self.transforms(img)

            return img, {
                "image_id": img_id,
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
                "orig_size": (H_orig, W_orig),
            }

        elif self.version == 'v2':
            img_info = self.images_list[idx]
            img_id = img_info["id"]
            img = Image.open(
                self.img_dir /
                img_info["file_name"]).convert("RGB")

            anns = self.anns.get(img_id, [])

            boxes, labels = [], []
            for ann in anns:
                boxes.append(ann["bbox"])  # xywh
                labels.append(ann["category_id"] - 1)

            boxes = np.array(boxes, dtype=np.float32) if len(
                boxes) else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(labels, dtype=np.int64) if len(
                labels) else np.zeros((0,), dtype=np.int64)

            img_padded, boxes_norm, scale, pad_x, pad_y = self.letterbox(
                img, boxes)
            img_tensor = self.normalize(img_padded)

            target = {
                "boxes": torch.tensor(boxes_norm, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": int(img_id),
                "orig_size": torch.tensor([img_info["height"],
                                           img_info["width"]]),
                "letterbox_params": torch.tensor([scale, pad_x, pad_y])
            }

            return img_tensor, target


class TestDataset(Dataset):
    """Dataset for the unlabelled test split (Unified)."""

    def __init__(self, img_dir: str, version: str = 'v1'):
        if version not in ['v1', 'v2']:
            raise ValueError("version must be either 'v1' or 'v2'")

        self.img_dir = Path(img_dir)
        self.files = sorted(
            self.img_dir.glob("*.png"),
            key=lambda p: int(
                p.stem))
        self.version = version

        if self.version == 'v1':
            self.transform = make_transforms_v1(train=False)
        elif self.version == 'v2':
            self.img_size = 320
            self.normalize = get_normalize_v2()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        W_orig, H_orig = img.size

        if self.version == 'v1':
            return self.transform(img), int(p.stem), H_orig, W_orig

        elif self.version == 'v2':
            scale = min(self.img_size / W_orig, self.img_size / H_orig)
            new_w = int(W_orig * scale)
            new_h = int(H_orig * scale)
            pad_x = (self.img_size - new_w) // 2
            pad_y = (self.img_size - new_h) // 2

            img_resized = img.resize((new_w, new_h), Image.BILINEAR)
            canvas = Image.new(
                "RGB", (self.img_size, self.img_size), (0, 0, 0))
            canvas.paste(img_resized, (pad_x, pad_y))

            img_tensor = self.normalize(canvas)

            return img_tensor, int(p.stem), H_orig, W_orig, scale, pad_x, pad_y


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)
