"""
models/skin_model.py

ResNet18 backbone with a shared FC + four sigmoid regression heads.

Output keys:
  acne_score
  redness_score
  texture_score
  dark_circle_score
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import timm


@dataclass
class SkinScores:
    acne_score:        torch.Tensor   # (B,)
    redness_score:     torch.Tensor
    texture_score:     torch.Tensor
    dark_circle_score: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "acne_score":        self.acne_score,
            "redness_score":     self.redness_score,
            "texture_score":     self.texture_score,
            "dark_circle_score": self.dark_circle_score,
        }

    def to_cpu_dict(self) -> Dict[str, float]:
        return {k: v.squeeze().item() for k, v in self.to_dict().items()}


class SkinAnalysisModel(nn.Module):
    """
    Architecture
    ────────────
    ResNet18 (ImageNet pre-trained)
      → Global Average Pool  (built-in to timm)
      → Dropout
      → Shared FC  (1408 → shared_fc_dim)
      → BatchNorm + ReLU
      → Four independent heads  (shared_fc_dim → 1)
      → Sigmoid
    """

    HEAD_NAMES = ("acne_score", "redness_score", "texture_score", "dark_circle_score")

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        shared_fc_dim: int = 256,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,        # Remove classifier head; returns feature vector
            global_pool="avg",    # Global Average Pooling
        )
        feat_dim = self.backbone.num_features   # EfficientNet-B2 → 1408

        # ── Shared trunk ──────────────────────────────────────────────────────
        self.shared_trunk = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feat_dim, shared_fc_dim),
            nn.BatchNorm1d(shared_fc_dim),
            nn.ReLU(inplace=True),
        )

        # ── Four regression heads ─────────────────────────────────────────────
        self.head_acne        = self._make_head(shared_fc_dim)
        self.head_redness     = self._make_head(shared_fc_dim)
        self.head_texture     = self._make_head(shared_fc_dim)
        self.head_dark_circle = self._make_head(shared_fc_dim)

        # ── Weight initialisation for new layers ──────────────────────────────
        self._init_weights()

    # ------------------------------------------------------------------

    @staticmethod
    def _make_head(in_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _init_weights(self):
        for module in [self.shared_trunk, self.head_acne, self.head_redness,
                       self.head_texture, self.head_dark_circle]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> SkinScores:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, 3, 224, 224)

        Returns
        -------
        SkinScores  (all scores in [0, 1])
        """
        features = self.backbone(x)
        shared   = self.shared_trunk(features) # (B, shared_fc_dim)

        return SkinScores(
            acne_score        = self.head_acne(shared).squeeze(1),         # (B,)
            redness_score     = self.head_redness(shared).squeeze(1),
            texture_score     = self.head_texture(shared).squeeze(1),
            dark_circle_score = self.head_dark_circle(shared).squeeze(1),
        )

    def freeze_backbone(self):
        """Freeze backbone weights (useful for early-stage fine-tuning)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def count_parameters(self) -> Dict[str, int]:
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
