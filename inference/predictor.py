"""
inference/predictor.py

End-to-end inference for uploaded face images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import torch

from config import InferenceConfig
from models.skin_model import SkinAnalysisModel
from preprocessing.face_pipeline import FacePreprocessor


@dataclass
class SkinReport:
    acne_score: float
    redness_score: float
    texture_score: float
    dark_circle_score: float
    overall_score: float
    face_detected: bool = True
    error: Optional[str] = None
    severity: Dict[str, str] = field(default_factory=dict)
    quality_checks: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "acne_score": self.acne_score,
            "redness_score": self.redness_score,
            "texture_score": self.texture_score,
            "dark_circle_score": self.dark_circle_score,
            "overall_score": self.overall_score,
            "face_detected": self.face_detected,
            "error": self.error,
            "severity": self.severity,
            "quality_checks": self.quality_checks,
        }

    def __str__(self) -> str:
        if self.error:
            return f"Skin analysis failed: {self.error}"
        lines = [
            "Skin Analysis Report",
            f"  Acne severity:     {self.acne_score:.3f} ({self.severity.get('acne', '')})",
            f"  Redness:           {self.redness_score:.3f} ({self.severity.get('redness', '')})",
            f"  Texture roughness: {self.texture_score:.3f} ({self.severity.get('texture', '')})",
            f"  Dark circles:      {self.dark_circle_score:.3f} ({self.severity.get('dark_circle', '')})",
            f"  Overall skin score:{self.overall_score:.3f} ({self.severity.get('overall', '')})",
        ]
        return "\n".join(lines)


def _score_to_severity(score: float) -> str:
    if score < 0.25:
        return "Clear"
    if score < 0.50:
        return "Mild"
    if score < 0.75:
        return "Moderate"
    return "Severe"


class SkinPredictor:
    MIN_BRIGHTNESS = 60.0
    MAX_BRIGHTNESS = 200.0
    MIN_BLUR_VARIANCE = 80.0

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        cfg: Optional[InferenceConfig] = None,
        device: Optional[str] = None,
    ):
        self.cfg = cfg or InferenceConfig()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[SkinPredictor] Device: {self.device}")

        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_config = checkpoint.get("model_config", {})
        self.model = SkinAnalysisModel(
            backbone=model_config.get("backbone", "resnet18"),
            pretrained=False,
            shared_fc_dim=model_config.get("shared_fc_dim", 256),
            dropout_rate=model_config.get("dropout_rate", 0.3),
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[SkinPredictor] Loaded weights from {checkpoint_path}")

        self.preprocessor = FacePreprocessor()

    def predict_from_path(self, image_path: Union[str, Path]) -> SkinReport:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return SkinReport(
                acne_score=0.0,
                redness_score=0.0,
                texture_score=0.0,
                dark_circle_score=0.0,
                overall_score=0.0,
                face_detected=False,
                error=f"Could not read image: {image_path}",
            )
        return self.predict_from_array(bgr)

    def predict_from_array(self, bgr: np.ndarray) -> SkinReport:
        quality = self.preprocessor.assess_image_quality(bgr)
        quality_error = self._quality_error(quality)
        if quality_error is not None:
            return SkinReport(
                acne_score=0.0,
                redness_score=0.0,
                texture_score=0.0,
                dark_circle_score=0.0,
                overall_score=0.0,
                face_detected=bool(quality.get("face_detected", False)),
                error=quality_error,
                quality_checks=quality,
            )

        tensor, _ = self.preprocessor(bgr)
        if tensor is None:
            return SkinReport(
                acne_score=0.0,
                redness_score=0.0,
                texture_score=0.0,
                dark_circle_score=0.0,
                overall_score=0.0,
                face_detected=False,
                error="Face alignment failed for this image.",
                quality_checks=quality,
            )

        return self._run_model(tensor, quality)

    def predict_from_tensor(self, tensor: torch.Tensor) -> SkinReport:
        return self._run_model(tensor, quality={})

    def _quality_error(self, quality: Dict[str, object]) -> Optional[str]:
        if not quality.get("face_detected", False):
            return "No face detected in the uploaded image."
        brightness = float(quality.get("brightness", 0.0))
        if brightness < self.MIN_BRIGHTNESS:
            return "Image is too dark. Please upload a brighter photo."
        if brightness > self.MAX_BRIGHTNESS:
            return "Image is too bright. Please reduce exposure and try again."
        blur_variance = float(quality.get("blur_variance", 0.0))
        if blur_variance < self.MIN_BLUR_VARIANCE:
            return "Image is too blurry. Please upload a sharper photo."
        return None

    @torch.no_grad()
    def _run_model(self, tensor: torch.Tensor, quality: Dict[str, object]) -> SkinReport:
        x = tensor.unsqueeze(0).to(self.device)
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            scores = self.model(x)

        raw = scores.to_cpu_dict()
        acne = float(raw["acne_score"])
        redness = float(raw["redness_score"])
        texture = float(raw["texture_score"])
        dark_circle = float(raw["dark_circle_score"])
        overall = (
            self.cfg.acne_weight * acne
            + self.cfg.redness_weight * redness
            + self.cfg.texture_weight * texture
            + self.cfg.dark_circle_weight * dark_circle
        )

        severity = {
            "acne": _score_to_severity(acne),
            "redness": _score_to_severity(redness),
            "texture": _score_to_severity(texture),
            "dark_circle": _score_to_severity(dark_circle),
            "overall": _score_to_severity(overall),
        }
        return SkinReport(
            acne_score=acne,
            redness_score=redness,
            texture_score=texture,
            dark_circle_score=dark_circle,
            overall_score=overall,
            face_detected=True,
            severity=severity,
            quality_checks=quality,
        )
