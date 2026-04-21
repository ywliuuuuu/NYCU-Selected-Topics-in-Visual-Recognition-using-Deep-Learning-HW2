"""
models.py - DAB-DETR model for digit detection (HW2).

Key idea of DAB-DETR (Liu et al., ICLR 2022):
  - Object queries are dynamic anchor boxes (cx, cy, w, h).
  - Each decoder layer refines the anchor and uses its (cx, cy) to modulate
    the cross-attention via sinusoidal positional encoding, giving the model
    explicit spatial priors and speeding up convergence significantly.
"""

import math
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


# ──────────────────── Positional Encoding ──────────────────────

def inverse_sigmoid(x, eps=1e-5):
    return torch.logit(x.clamp(min=eps, max=1 - eps))


def pos_encoding_1d(pos, num_feats, temperature=10000):
    """Sinusoidal encoding for a 1-D position tensor.

    Args:
        pos:       (*, 1) float tensor of positions in [0, 1].
        num_feats: output dimension (must be even).

    Returns:
        Tensor of shape (*, num_feats).
    """
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    enc = pos / dim_t
    enc = torch.stack([enc[..., 0::2].sin(),
                       enc[..., 1::2].cos()], dim=-1)
    return enc.flatten(-2)


class PositionEmbeddingSine2D(nn.Module):
    """2-D sine-cosine positional encoding for feature maps (B, C, H, W)."""

    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x):
        B, _, H, W = x.shape
        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=x.device)
        y_embed = y_embed / (H + 1e-6) * 2 * math.pi
        x_embed = x_embed / (W + 1e-6) * 2 * math.pi

        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_y = y_embed[:, None] / dim_t
        pos_x = x_embed[:, None] / dim_t
        pos_y = torch.stack([pos_y[:, 0::2].sin(),
                             pos_y[:, 1::2].cos()], -1).flatten(1)
        pos_x = torch.stack([pos_x[:, 0::2].sin(),
                             pos_x[:, 1::2].cos()], -1).flatten(1)

        pos = torch.cat([
            pos_y[:, None, :].expand(H, W, -1),
            pos_x[None, :, :].expand(H, W, -1),
        ], dim=-1)
        return pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)


# ──────────────────── Transformer Encoder ──────────────────────

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, src, pos):
        q = k = src + pos
        src = self.norm1(src + self.drop(self.self_attn(q, k, src)[0]))
        src = self.norm2(src + self.drop(self.ff(src)))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, pos):
        for layer in self.layers:
            src = layer(src, pos)
        return self.norm(src)


# ──────────────────── DAB-DETR Decoder ─────────────────────────

class DABDecoderLayer(nn.Module):
    """One DAB-DETR decoder layer.

    Compared to vanilla DETR:
    1. Self-attention is conditioned on the current anchor encoding so
       each query has a unique spatial identity from the start.
    2. Cross-attention query = content (from hidden state) + position
       (sinusoidal of anchor), so the model attends to the anchor region.
    3. A small MLP refines the anchor after each layer.
    """

    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.pos_proj = nn.Linear(d_model, d_model)
        self.query_scale = nn.Linear(d_model, d_model)

    def forward(self, tgt, memory, anchor_enc, encoder_pos):
        """
        tgt:         (Q, B, d)
        memory:      (HW, B, d)
        anchor_enc:  (Q, B, d)   sinusoidal encoding of current anchor
        encoder_pos: (HW, B, d)
        """
        q = k = tgt + anchor_enc
        tgt = self.norm1(tgt + self.drop(self.self_attn(q, k, tgt)[0]))

        pos_q = self.pos_proj(anchor_enc)
        q = self.query_scale(tgt) + pos_q
        k = memory + encoder_pos
        tgt = self.norm2(tgt + self.drop(self.cross_attn(q, k, memory)[0]))

        tgt = self.norm3(tgt + self.drop(self.ff(tgt)))
        return tgt


class DABTransformerDecoder(nn.Module):
    """Stacked DAB-DETR decoder with iterative anchor refinement."""

    def __init__(self, d_model, nhead, num_layers, dim_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DABDecoderLayer(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

        # Per-layer anchor refinement MLP
        self.anchor_refine = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, 4),
            )
            for _ in range(num_layers)
        ])

    def _anchor_to_enc(self, anchor_sigmoid):
        """
            Encode anchor (Q, B, 4) in [0,1] →
            sinusoidal embedding (Q, B, d_model).
        """
        d = self.d_model // 4
        cx_enc = pos_encoding_1d(anchor_sigmoid[..., 0:1], d)
        cy_enc = pos_encoding_1d(anchor_sigmoid[..., 1:2], d)
        w_enc = pos_encoding_1d(anchor_sigmoid[..., 2:3], d)
        h_enc = pos_encoding_1d(anchor_sigmoid[..., 3:4], d)
        return torch.cat([cx_enc, cy_enc, w_enc, h_enc], dim=-1)

    def forward(self, tgt, memory, anchor_sigmoid, encoder_pos):
        """
        tgt:            (Q, B, d)  initial query content (zeros)
        memory:         (HW, B, d) encoder output
        anchor_sigmoid: (Q, B, 4)  initial anchors in [0,1]
        encoder_pos:    (HW, B, d)

        Returns:
            all_hs:      (num_layers, B, Q, d)
            all_anchors: (num_layers, B, Q, 4)  anchors in [0,1] at each layer
        """
        all_hs = []
        all_anchors = []

        # Convert initial sigmoid anchors to logit space for stable additive
        # updates
        anchor_logit = inverse_sigmoid(anchor_sigmoid)

        for i, layer in enumerate(self.layers):
            anchor_enc = self._anchor_to_enc(anchor_sigmoid)
            tgt = layer(tgt, memory, anchor_enc, encoder_pos)

            # Additive refinement in logit space → sigmoid back to [0,1]
            delta = self.anchor_refine[i](tgt)
            anchor_logit = anchor_logit + delta
            anchor_sigmoid = anchor_logit.sigmoid()

            all_hs.append(self.norm(tgt).transpose(0, 1))
            all_anchors.append(anchor_sigmoid.transpose(0, 1))

        return torch.stack(all_hs), torch.stack(all_anchors)


# ────────────────────────── DAB-DETR ───────────────────────────

class DABDETR(nn.Module):
    """DAB-DETR with ResNet-50 backbone (HW2 compliant).

    Default architecture is intentionally lighter than the original paper
    (enc_layers=4, dec_layers=4, dim_ff=1024) to fit an RTX 3060 Ti (8 GB)
    and finish 50 epochs within ~10 hours.
    """

    def __init__(
        self,
        num_classes=10,
        num_queries=100,
        d_model=256,
        nhead=8,
        enc_layers=4,
        dec_layers=4,
        dim_ff=1024,
        dropout=0.1,
    ):
        super().__init__()

        # ── Backbone (pretrained ResNet-50) ──
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.input_proj = nn.Conv2d(2048, d_model, kernel_size=1)

        # ── Positional encoding ──
        self.pos_enc = PositionEmbeddingSine2D(d_model // 2)

        # ── Transformer ──
        self.encoder = TransformerEncoder(
            d_model, nhead, enc_layers, dim_ff, dropout)
        self.decoder = DABTransformerDecoder(
            d_model, nhead, dec_layers, dim_ff, dropout)

        # ── Learned initial anchor boxes (uniform in [0,1] after sigmoid) ──
        self.anchor_init = nn.Embedding(num_queries, 4)
        nn.init.uniform_(self.anchor_init.weight)

        # ── Initial query content (zeros; updated through decoder layers) ──
        self.query_content = nn.Embedding(num_queries, d_model)
        nn.init.zeros_(self.query_content.weight)

        # ── Prediction heads ──
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        # bbox_embed predicts a delta in logit space, added to the refined
        # anchor
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 4),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            dict:
              pred_logits: (B, Q, num_classes+1)
              pred_boxes:  (B, Q, 4) in [0,1] cxcywh
              aux_outputs: list of same dicts for intermediate decoder layers
        """
        B = x.shape[0]

        feat = self.backbone(x)
        src = self.input_proj(feat)
        pos = self.pos_enc(src)

        src = src.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)

        memory = self.encoder(src, pos)

        tgt = self.query_content.weight.unsqueeze(1).expand(-1, B, -1)
        anchor = (
            self.anchor_init.weight.sigmoid()
            .unsqueeze(1)
            .expand(-1, B, -1)
        )

        all_hs, all_anchors = self.decoder(tgt, memory, anchor, pos)

        outputs = []
        for hs, anch in zip(all_hs, all_anchors):
            delta = self.bbox_embed(hs)
            boxes = (inverse_sigmoid(anch) + delta).sigmoid()
            logits = self.class_embed(hs)
            outputs.append({"pred_logits": logits, "pred_boxes": boxes})

        return {
            "pred_logits": outputs[-1]["pred_logits"],
            "pred_boxes": outputs[-1]["pred_boxes"],
            "aux_outputs": outputs[:-1],
        }
