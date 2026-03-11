"""
datasets/pseudo_label_generator.py

Generates pseudo-labels for FFHQ images (no ground-truth available):
  - texture_score    : LBP entropy + gradient magnitude variance
  - dark_circle_score: under-eye vs cheek brightness contrast
  - redness_score    : mean a* channel over cheek region (LAB)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm

from preprocessing.face_pipeline import (
    FacePreprocessor,
    extract_cheek_region,
    extract_texture_region,
    extract_undereye_region,
)

# ── LBP parameters ────────────────────────────────────────────────────────────
_LBP_RADIUS = 3
_LBP_N_POINTS = 8 * _LBP_RADIUS
_LBP_METHOD = "uniform"


class PseudoLabelGenerator:
    """
    Compute pseudo-labels for unlabelled facial images.

    Parameters
    ----------
    cache_dir : str | Path
        Directory where computed labels are cached as JSON.
    """

    def __init__(self, cache_dir: str = "data/cache/pseudo_labels_region_v2"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._preprocessor = FacePreprocessor()
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
        )

    # ------------------------------------------------------------------
    # Batch generation (with disk caching)
    # ------------------------------------------------------------------

    def generate_for_dataset(
        self, image_paths: List[str], dataset_name: str = "ffhq"
    ) -> Dict[str, Dict[str, float]]:
        """
        Parameters
        ----------
        image_paths : list of absolute/relative image paths
        dataset_name : used for the cache filename

        Returns
        -------
        labels : {image_path: {"texture_score": float, "redness_score": float,
                                "dark_circle_score": float}}
                Missing entries = failed processing (no face detected).
        """
        cache_file = self.cache_dir / f"{dataset_name}_pseudo_labels.json"
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
            # Only return entries that are in the current image list
            result = {p: cached[p] for p in image_paths if p in cached}
            missing = [p for p in image_paths if p not in cached]
        else:
            result = {}
            missing = image_paths

        if missing:
            new_labels = {}
            for path in tqdm(missing, desc=f"Generating pseudo-labels ({dataset_name})"):
                labels = self._compute_single(path)
                if labels is not None:
                    new_labels[path] = labels

            result.update(new_labels)

            # Merge with existing cache and save
            if cache_file.exists():
                with open(cache_file) as f:
                    old = json.load(f)
                old.update(new_labels)
                with open(cache_file, "w") as f:
                    json.dump(old, f, indent=2)
            else:
                with open(cache_file, "w") as f:
                    json.dump(new_labels, f, indent=2)

        return result

    # ------------------------------------------------------------------
    # Single image
    # ------------------------------------------------------------------

    def _compute_single(self, image_path: str) -> Optional[Dict[str, float]]:
        bgr = cv2.imread(image_path)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Landmarks
        mesh_result = self._mesh.process(rgb)
        lm = None
        if mesh_result.multi_face_landmarks:
            raw = mesh_result.multi_face_landmarks[0].landmark
            lm = np.array([[p.x, p.y] for p in raw], dtype=np.float32)

        texture_region = extract_texture_region(rgb, lm) if lm is not None else None
        if texture_region is None or texture_region.size == 0:
            h, w = rgb.shape[:2]
            texture_region = rgb[int(h * 0.15):int(h * 0.45), int(w * 0.22):int(w * 0.78)]

        texture_score     = self._texture_score(texture_region)
        redness_score     = self._redness_score(rgb, lm)
        dark_circle_score = self._dark_circle_score(rgb, lm)

        return {
            "texture_score":      float(texture_score),
            "redness_score":      float(redness_score),
            "dark_circle_score":  float(dark_circle_score),
        }

    # ------------------------------------------------------------------
    # Individual metric computations
    # ------------------------------------------------------------------

    @staticmethod
    def _texture_score(rgb: np.ndarray) -> float:
        """
        Combine LBP entropy and gradient magnitude variance.
        Higher value → rougher texture.
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # LBP entropy
        lbp = local_binary_pattern(gray, _LBP_N_POINTS, _LBP_RADIUS, method=_LBP_METHOD)
        n_bins = _LBP_N_POINTS + 2
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        hist = hist[hist > 0]
        lbp_entropy = float(-np.sum(hist * np.log2(hist)))
        lbp_entropy_norm = np.clip(lbp_entropy / 5.0, 0.0, 1.0)   # empirical max ~5 bits

        # Gradient magnitude variance
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        grad_var = float(np.var(mag))
        grad_var_norm = np.clip(grad_var / 3000.0, 0.0, 1.0)      # empirical max

        score = 0.5 * lbp_entropy_norm + 0.5 * grad_var_norm
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _redness_score(rgb: np.ndarray, lm: Optional[np.ndarray]) -> float:
        """
        Mean a* value over cheek regions converted to [0, 1].
        a* range in OpenCV uint8 LAB: 0-255, neutral=128.
        """
        if lm is not None:
            region = extract_cheek_region(rgb, lm, side="both")
        else:
            # Fallback: use central strip of the image
            h, w = rgb.shape[:2]
            region = rgb[h // 3: 2 * h // 3, w // 4: 3 * w // 4]

        if region is None or region.size == 0:
            h, w = rgb.shape[:2]
            region = rgb[h // 3: 2 * h // 3, w // 4: 3 * w // 4]

        lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
        a_channel = lab[:, :, 1].astype(np.float32)     # 0-255, neutral=128
        mean_a = float(np.mean(a_channel))

        # Map [133, 165] → [0, 1]  (tighter range for facial skin redness sensitivity)
        score = (mean_a - 133.0) / 32.0
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _dark_circle_score(rgb: np.ndarray, lm: Optional[np.ndarray]) -> float:
        """
        Normalised contrast: (cheek_brightness - undereye_brightness) / cheek_brightness
        """
        if lm is None:
            return 0.0

        undereye = extract_undereye_region(rgb, lm, side="both")
        cheek    = extract_cheek_region(rgb, lm, side="both")

        def mean_luminance(region: Optional[np.ndarray]) -> Optional[float]:
            if region is None or region.size == 0:
                return None
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            return float(np.mean(gray))

        l_eye   = mean_luminance(undereye)
        l_cheek = mean_luminance(cheek)

        if l_eye is None or l_cheek is None or l_cheek < 1e-6:
            return 0.0

        contrast = (l_cheek - l_eye) / (l_cheek + 1e-6)
        # Amplify subtle differences (raw contrast often < 0.1)
        score = contrast * 2.5
        return float(np.clip(score, 0.0, 1.0))



