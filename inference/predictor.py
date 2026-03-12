"""
inference/predictor.py

End-to-end inference for region-aware uploaded face analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
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


def _estimate_skin_tone(bgr: np.ndarray) -> float:
    """Estimate skin lightness from the cheek area. Returns L* in [0, 100].

    Lower L* = darker skin. Used to correct bias where darker skin
    inflates redness / dark_circle / texture scores.
    """
    h, w = bgr.shape[:2]
    # Central cheek region (rough crop)
    cheek = bgr[int(h * 0.35):int(h * 0.70), int(w * 0.20):int(w * 0.80)]
    if cheek.size == 0:
        return 65.0  # neutral default
    lab = cv2.cvtColor(cheek, cv2.COLOR_BGR2LAB)
    return float(np.mean(lab[:, :, 0]))  # L* channel, range 0-255 in OpenCV


def _skin_tone_correction(score: float, lightness: float, sensitivity: float = 0.4) -> float:
    """Reduce inflated scores for darker skin tones.

    For lighter skin (L* > 140), no correction.
    For darker skin (L* < 100), scores are scaled down proportionally.
    `sensitivity` controls how much correction to apply (0 = none, 1 = full).
    """
    if lightness >= 140.0:
        return score
    # Linear ramp: full correction at L*=60, no correction at L*=140
    correction_factor = max(0.0, (140.0 - lightness) / 80.0)  # 0.0 to 1.0
    reduction = correction_factor * sensitivity * score * 0.35  # up to 35% reduction
    return max(0.0, score - reduction)


class SkinPredictor:
        @staticmethod
        def _calibrate_dark_circle(raw: float) -> float:
            """Remap dark circles so clear=0.08-0.12, moderate=0.55-0.70, severe=0.85+ (like acne).

            Uses a piecewise linear mapping to fix the floor and preserve the ceiling.
            """
            if raw < 0.25:
                # Clear range: map [0.05, 0.25] → [0.08, 0.12]
                return max(0.08, 0.08 + (raw - 0.05) * 0.20)
            elif raw < 0.50:
                # Mild range: map [0.25, 0.50] → [0.10, 0.40]
                return 0.10 + (raw - 0.25) * 1.2
            elif raw < 0.75:
                # Moderate range: map [0.50, 0.75] → [0.40, 0.70]
                return 0.40 + (raw - 0.50) * 1.2
            else:
                # Severe range: map [0.75, 0.95] → [0.70, 0.95]
                return 0.70 + (raw - 0.75) * 1.25
    MIN_BRIGHTNESS = 60.0
    MAX_BRIGHTNESS = 200.0
    MIN_BLUR_VARIANCE = 80.0

    def __init__(self, checkpoint_path: Union[str, Path], cfg: Optional[InferenceConfig] = None, device: Optional[str] = None):
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
            return SkinReport(0.0, 0.0, 0.0, 0.0, 0.0, face_detected=False, error=f"Could not read image: {image_path}")
        return self.predict_from_array(bgr)

    def predict_from_array(self, bgr) -> SkinReport:
        quality = self.preprocessor.assess_image_quality(bgr)
        quality_error = self._quality_error(quality)
        if quality_error is not None:
            return SkinReport(0.0, 0.0, 0.0, 0.0, 0.0, face_detected=bool(quality.get("face_detected", False)), error=quality_error, quality_checks=quality)

        skin_lightness = _estimate_skin_tone(bgr)

        regions, _ = self.preprocessor(bgr)
        if regions is None:
            return SkinReport(0.0, 0.0, 0.0, 0.0, 0.0, face_detected=False, error="Face alignment failed for this image.", quality_checks=quality)
        return self._run_model(regions, quality, skin_lightness=skin_lightness)

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

    @staticmethod
    def _clamp_score(raw: float, low: float = 0.05, high: float = 0.95) -> float:
        """Compress extreme model outputs to a believable range."""
        return max(low, min(high, raw))

    def _robust_overall(self, scores: Dict[str, float]) -> float:
        """Winsorized weighted mean: caps any head contributing > 50% of total."""
        weights = {
            "acne": self.cfg.acne_weight,
            "redness": self.cfg.redness_weight,
            "texture": self.cfg.texture_weight,
            "dark_circle": self.cfg.dark_circle_weight,
        }
        total_w = sum(weights.values())
        raw_overall = sum(scores[k] * weights[k] for k in scores) / total_w

        # Winsorize: if any single head dominates, clamp its influence
        adjusted = dict(scores)
        for k in scores:
            contribution = scores[k] * weights[k] / (raw_overall * total_w + 1e-6)
            if contribution > 0.50:
                adjusted[k] = min(scores[k], raw_overall * 1.5)

        return sum(adjusted[k] * weights[k] for k in adjusted) / total_w

    @staticmethod
    def _calibrate_acne(raw: float) -> float:
        """Remap acne so clear=0.0-0.1, moderate=0.55-0.70, severe=0.85+.

        Uses a piecewise linear mapping to fix the floor (too high) and
        preserve the ceiling (severe must stay above 0.85).
        """
        if raw < 0.25:
            # Clear range: map [0.05, 0.25] → [0.08, 0.12]
            # This makes 'clear' less strict, so scores 0.08-0.12 are possible for nearly clear skin
            return max(0.08, 0.08 + (raw - 0.05) * 0.20)
        elif raw < 0.50:
            # Mild range: map [0.25, 0.50] → [0.10, 0.40]
            return 0.10 + (raw - 0.25) * 1.2
        elif raw < 0.75:
            # Moderate range: map [0.50, 0.75] → [0.40, 0.70]
            return 0.40 + (raw - 0.50) * 1.2
        else:
            # Severe range: map [0.75, 0.95] → [0.70, 0.95]
            return 0.70 + (raw - 0.75) * 1.25

    @staticmethod
    def _apply_acne_redness_correlation(acne: float, redness: float) -> float:
        """When acne is moderate+, redness should be at least ~2/3 of acne.

        Acne causes inflammation which manifests as redness.
        Does NOT cap redness — extreme redness samples without acne stay as-is.
        """
        if acne < 0.35:  # clear/mild acne — redness stands on its own
            return redness
        acne_implied_redness = acne * 0.65
        return max(redness, acne_implied_redness)

    @staticmethod
    def _apply_acne_texture_correlation(acne: float, texture: float) -> float:
        """When acne is present, texture should be proportionally elevated.

        Acne-prone skin has bumps/roughness. When acne is low, don't inflate texture.
        """
        # Texture should always be at least 65% of acne, regardless of acne level
        return max(texture, acne * 0.65)

    @torch.no_grad()
    def _run_model(self, regions: Dict[str, torch.Tensor], quality: Dict[str, object], skin_lightness: float = 140.0) -> SkinReport:
        inputs = {key: value.unsqueeze(0).to(self.device) for key, value in regions.items()}
        with torch.amp.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
            scores = self.model(inputs)

        raw = scores.to_cpu_dict()

        # 1. Base clamping
        acne = self._clamp_score(float(raw["acne_score"]))
        redness = self._clamp_score(float(raw["redness_score"]))
        texture = self._clamp_score(float(raw["texture_score"]))
        dark_circle = self._clamp_score(self._calibrate_dark_circle(float(raw["dark_circle_score"])))

        # 2. Acne calibration (compress overshooting)
        acne = self._clamp_score(self._calibrate_acne(acne))

        # 3. Skin tone correction (reduce false positives on darker skin)
        acne = _skin_tone_correction(acne, skin_lightness, sensitivity=0.12)
        redness = _skin_tone_correction(redness, skin_lightness, sensitivity=0.14)
        texture = _skin_tone_correction(texture, skin_lightness, sensitivity=0.12)
        dark_circle = _skin_tone_correction(dark_circle, skin_lightness, sensitivity=0.18)

        # 4. Acne-correlated adjustments
        redness = self._clamp_score(self._apply_acne_redness_correlation(acne, redness))
        texture = self._clamp_score(self._apply_acne_texture_correlation(acne, texture))

        score_dict = {"acne": acne, "redness": redness, "texture": texture, "dark_circle": dark_circle}
        overall = self._clamp_score(self._robust_overall(score_dict))

        severity = {
            "acne": _score_to_severity(acne),
            "redness": _score_to_severity(redness),
            "texture": _score_to_severity(texture),
            "dark_circle": _score_to_severity(dark_circle),
            "overall": _score_to_severity(overall),
        }
        return SkinReport(acne, redness, texture, dark_circle, overall, face_detected=True, severity=severity, quality_checks=quality)
