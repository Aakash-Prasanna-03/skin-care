"""
inference/predictor.py

End-to-end inference for region-aware uploaded face analysis.
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
    if score < 0.10:
        return "Clear"
    if score < 0.35:
        return "Mild"
    if score < 0.70:
        return "Moderate"
    return "Severe"


<<<<<<< HEAD
def _estimate_skin_tone(bgr: np.ndarray) -> float:
    """Estimate skin lightness from the cheek area. Returns OpenCV L* in [0, 255].

    Lower L* = darker skin. Used to correct bias where darker skin
    inflates redness / dark_circle / texture scores.
    """
    h, w = bgr.shape[:2]
    # Central cheek region (rough crop)
    cheek = bgr[int(h * 0.35):int(h * 0.70), int(w * 0.20):int(w * 0.80)]
    if cheek.size == 0:
        return 140.0  # neutral default (no correction)
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
        """Remap dark circles:
        - Clear:    0.00–0.10
        - Mild:     0.10–0.35
        - Moderate: 0.35–0.70
        - Severe:   0.70–1.00
        """
        if raw < 0.25:
            # Clear: compress [0.0, 0.25] → [0.0, 0.10]
            return raw * 0.40
        elif raw < 0.50:
            # Mild: map [0.25, 0.50] → [0.10, 0.35]
            return 0.10 + (raw - 0.25) * 1.0
        elif raw < 0.75:
            # Moderate: map [0.50, 0.75] → [0.35, 0.70]
            return 0.35 + (raw - 0.50) * 1.4
        else:
            # Severe: map [0.75, 1.0] → [0.70, 1.0]
            return 0.70 + (raw - 0.75) * 1.2

=======
class SkinPredictor:
>>>>>>> parent of 7e1e563 (more detailed for acne and all)
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

        regions, _ = self.preprocessor(bgr)
        if regions is None:
            return SkinReport(0.0, 0.0, 0.0, 0.0, 0.0, face_detected=False, error="Face alignment failed for this image.", quality_checks=quality)
        return self._run_model(regions, quality)

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
    def _clamp_score(raw: float, low: float = 0.0, high: float = 1.0) -> float:
        """Clamp model outputs to valid range."""
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

<<<<<<< HEAD
    @staticmethod
    def _calibrate_acne(raw: float) -> float:
        """Remap acne so:
        - Clear:    0.00–0.10
        - Mild:     0.10–0.35
        - Moderate: 0.35–0.70
        - Severe:   0.70–1.00
        """
        if raw < 0.25:
            # Clear skin: compress [0.0, 0.25] → [0.0, 0.10]
            return raw * 0.40
        elif raw < 0.50:
            # Mild: map [0.25, 0.50] → [0.10, 0.35]
            return 0.10 + (raw - 0.25) * 1.0
        elif raw < 0.75:
            # Moderate: map [0.50, 0.75] → [0.35, 0.70]
            return 0.35 + (raw - 0.50) * 1.4
        else:
            # Severe: map [0.75, 1.0] → [0.70, 1.0]
            return 0.70 + (raw - 0.75) * 1.2

    @staticmethod
    def _calibrate_redness(raw: float) -> float:
        """Remap redness — less compression than acne so red faces stay high:
        - Clear:    0.00–0.10
        - Mild:     0.10–0.35
        - Moderate: 0.35–0.70
        - Severe:   0.70–1.00
        """
        if raw < 0.15:
            # Clear: compress [0.0, 0.15] → [0.0, 0.10]
            return raw * 0.667
        elif raw < 0.35:
            # Mild: map [0.15, 0.35] → [0.10, 0.35]
            return 0.10 + (raw - 0.15) * 1.25
        elif raw < 0.65:
            # Moderate: map [0.35, 0.65] → [0.35, 0.70]
            return 0.35 + (raw - 0.35) * 1.167
        else:
            # Severe: map [0.65, 1.0] → [0.70, 1.0]
            return 0.70 + (raw - 0.65) * 0.857

    @staticmethod
    def _calibrate_texture(raw: float) -> float:
        """Remap texture to severity buckets:
        - Clear:    0.00–0.10
        - Mild:     0.10–0.35
        - Moderate: 0.35–0.70
        - Severe:   0.70–1.00
        """
        if raw < 0.20:
            # Clear: compress [0.0, 0.20] → [0.0, 0.10]
            return raw * 0.50
        elif raw < 0.45:
            # Mild: map [0.20, 0.45] → [0.10, 0.35]
            return 0.10 + (raw - 0.20) * 1.0
        elif raw < 0.70:
            # Moderate: map [0.45, 0.70] → [0.35, 0.70]
            return 0.35 + (raw - 0.45) * 1.4
        else:
            # Severe: map [0.70, 1.0] → [0.70, 1.0]
            return 0.70 + (raw - 0.70) * 1.0

    @staticmethod
    def _apply_acne_redness_correlation(acne: float, redness: float) -> float:
        """Redness is largely independent — rosacea, irritation, sunburn exist without acne.

        Only when acne is moderate+ does inflammation guarantee *some* redness.
        Does NOT cap redness — high redness without acne stays as-is.
        """
        if acne < 0.36:  # clear/mild acne — redness stands entirely on its own
            return redness
        # Moderate+ acne implies significant redness from inflammation
        acne_implied_redness = acne * 0.62
        return max(redness, acne_implied_redness)

    @staticmethod
    def _apply_acne_texture_correlation(acne: float, texture: float) -> float:
        """Texture is mostly independent — pores, wrinkles, sun damage exist without acne.

        Only when acne is moderate+ does inflammation imply some texture roughness.
        Does NOT cap texture — high texture without acne stays as-is.
        """
        if acne < 0.35:
            # Clear/mild acne — texture stands entirely on its own
            return texture
        # Moderate+ acne implies texture roughness (scales up with severity)
        acne_implied_texture = acne * (0.23 + acne * 0.47)
        return max(texture, acne_implied_texture)

=======
>>>>>>> parent of 7e1e563 (more detailed for acne and all)
    @torch.no_grad()
    def _run_model(self, regions: Dict[str, torch.Tensor], quality: Dict[str, object]) -> SkinReport:
        inputs = {key: value.unsqueeze(0).to(self.device) for key, value in regions.items()}
        with torch.amp.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
            scores = self.model(inputs)

        raw = scores.to_cpu_dict()
        acne = self._clamp_score(float(raw["acne_score"]))
        redness = self._clamp_score(float(raw["redness_score"]))
        texture = self._clamp_score(float(raw["texture_score"]))
        dark_circle = self._clamp_score(float(raw["dark_circle_score"]))
<<<<<<< HEAD

        # 2. Calibrate all scores to proper ranges
        acne = self._clamp_score(self._calibrate_acne(acne))
        dark_circle = self._clamp_score(self._calibrate_dark_circle(dark_circle))
        redness = self._clamp_score(self._calibrate_redness(redness))
        texture = self._clamp_score(self._calibrate_texture(texture))

        # 3. Skin tone correction (reduce false positives on darker skin)
        acne = _skin_tone_correction(acne, skin_lightness, sensitivity=0.15)
        redness = _skin_tone_correction(redness, skin_lightness, sensitivity=0.20)
        texture = _skin_tone_correction(texture, skin_lightness, sensitivity=0.25)
        dark_circle = _skin_tone_correction(dark_circle, skin_lightness, sensitivity=0.35)

        # 4. Acne-correlated adjustments
        redness = self._clamp_score(self._apply_acne_redness_correlation(acne, redness))
        texture = self._clamp_score(self._apply_acne_texture_correlation(acne, texture))
=======
>>>>>>> parent of 7e1e563 (more detailed for acne and all)

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
