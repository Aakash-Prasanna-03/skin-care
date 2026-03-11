"""
models/skin_model.py

Region-aware skin analysis model using:
- full face for acne/context
- cheek crop for redness
- under-eye crop for dark circles
- texture patch for texture
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import timm


@dataclass
class SkinScores:
    acne_score: torch.Tensor          # ordinal: (B, 3) cumulative probs during training
    redness_score: torch.Tensor
    texture_score: torch.Tensor
    dark_circle_score: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "acne_score": self.acne_score,
            "redness_score": self.redness_score,
            "texture_score": self.texture_score,
            "dark_circle_score": self.dark_circle_score,
        }

    def to_cpu_dict(self) -> Dict[str, float]:
        out = {}
        for key, value in self.to_dict().items():
            v = value.detach().cpu()
            if v.dim() > 1:
                # Ordinal head: sigmoid(logits) → cumulative probs → mean → scalar score
                out[key] = float(torch.sigmoid(v).squeeze(0).mean().item())
            else:
                out[key] = float(v.squeeze().item())
        return out


class RegionEncoder(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, proj_dim: int):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.num_features, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class OrdinalHead(nn.Module):
    """CORAL-style ordinal classification: K-1 cumulative binary classifiers."""

    def __init__(self, in_dim: int, num_classes: int = 4):
        super().__init__()
        self.num_thresholds = num_classes - 1  # 3 thresholds for 4 ordinal levels
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_thresholds),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits for cumulative thresholds. Shape: (B, 3)
        Use with binary_cross_entropy_with_logits during training."""
        return self.fc(x)

    def to_score(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to a [0, 1] scalar via sigmoid + mean. Shape: (B,)"""
        return torch.sigmoid(logits).mean(dim=-1)


class RegressionHead(nn.Sequential):
    """Standard Sigmoid regression head for continuous targets."""

    def __init__(self, in_dim: int):
        super().__init__(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )


class SkinAnalysisModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        shared_fc_dim: int = 256,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        region_dim = max(64, shared_fc_dim // 2)
        self.full_face_encoder = RegionEncoder(backbone, pretrained, region_dim)
        self.cheek_encoder = RegionEncoder(backbone, pretrained, region_dim)
        self.undereye_encoder = RegionEncoder(backbone, pretrained, region_dim)
        self.texture_encoder = RegionEncoder(backbone, pretrained, region_dim)

        fused_dim = region_dim * 4
        self.shared_trunk = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(fused_dim, shared_fc_dim),
            nn.BatchNorm1d(shared_fc_dim),
            nn.ReLU(inplace=True),
        )
        self.acne_head = OrdinalHead(region_dim + shared_fc_dim, num_classes=4)
        self.redness_head = RegressionHead(region_dim + shared_fc_dim)
        self.texture_head = RegressionHead(region_dim + shared_fc_dim)
        self.dark_circle_head = RegressionHead(region_dim + shared_fc_dim)
        self._init_weights()

    def _init_weights(self):
        for module in [
            self.full_face_encoder.proj,
            self.cheek_encoder.proj,
            self.undereye_encoder.proj,
            self.texture_encoder.proj,
            self.shared_trunk,
            self.acne_head,
            self.redness_head,
            self.texture_head,
            self.dark_circle_head,
        ]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> SkinScores:
        full_face = self.full_face_encoder(inputs["full_face"])
        cheek = self.cheek_encoder(inputs["cheek"])
        undereye = self.undereye_encoder(inputs["undereye"])
        texture = self.texture_encoder(inputs["texture"])

        fused = self.shared_trunk(torch.cat([full_face, cheek, undereye, texture], dim=1))

        acne_features = torch.cat([fused, full_face], dim=1)
        redness_features = torch.cat([fused, cheek], dim=1)
        texture_features = torch.cat([fused, texture], dim=1)
        dark_features = torch.cat([fused, undereye], dim=1)

        return SkinScores(
            acne_score=self.acne_head(acne_features),             # (B, 3) ordinal
            redness_score=self.redness_head(redness_features).squeeze(1),
            texture_score=self.texture_head(texture_features).squeeze(1),
            dark_circle_score=self.dark_circle_head(dark_features).squeeze(1),
        )

    def count_parameters(self) -> Dict[str, int]:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {"total": total, "trainable": trainable}
