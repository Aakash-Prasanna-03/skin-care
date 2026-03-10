"""
preprocessing/face_pipeline.py

End-to-end face preprocessing:
  1. Face detection (MediaPipe FaceDetection)
  2. Landmark detection (MediaPipe FaceMesh)
  3. Eye-based alignment
  4. Face crop
  5. Resize to 224x224
  6. ImageNet normalization -> tensor (3, 224, 224)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch


_LEFT_EYE_IDX = 33
_RIGHT_EYE_IDX = 263

_LEFT_UNDEREYE = [144, 145, 153, 154, 155]
_RIGHT_UNDEREYE = [373, 374, 380, 381, 382]

_LEFT_CHEEK = [205, 36, 47, 50, 101]
_RIGHT_CHEEK = [425, 266, 277, 280, 330]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_SIZE = (224, 224)


class FacePreprocessor:
    """Face detection, alignment, cropping, and normalization."""

    def __init__(
        self,
        target_size: Tuple[int, int] = TARGET_SIZE,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        padding_ratio: float = 0.25,
    ):
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence,
        )
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __call__(
        self, bgr: np.ndarray
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face_crop, _ = self._detect_and_crop(rgb, require_face=True)
        if face_crop is None:
            return None, None

        lm_norm = self._get_landmarks(face_crop)
        if lm_norm is None:
            return None, None

        face_aligned = self._align(face_crop, lm_norm)
        face_resized = cv2.resize(
            face_aligned,
            self.target_size,
            interpolation=cv2.INTER_AREA,
        )
        landmarks_px = (
            lm_norm * np.array(face_crop.shape[:2][::-1], dtype=np.float32)
        ).astype(int)
        return self._to_tensor(face_resized), landmarks_px

    def preprocess_for_inference(self, bgr: np.ndarray) -> Optional[torch.Tensor]:
        tensor, _ = self(bgr)
        if tensor is None:
            return None
        return tensor.unsqueeze(0)

    def detect_face_bbox(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        _, bbox = self._detect_and_crop(rgb, require_face=True)
        return bbox

    def assess_image_quality(self, bgr: np.ndarray) -> Dict[str, object]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        bbox = self.detect_face_bbox(bgr)
        face_detected = bbox is not None
        face_centered = False

        if bbox is not None:
            h, w = gray.shape[:2]
            x1, y1, x2, y2 = bbox
            face_cx = (x1 + x2) / 2.0
            face_cy = (y1 + y2) / 2.0
            face_centered = (
                abs(face_cx - (w / 2.0)) <= 0.2 * w
                and abs(face_cy - (h / 2.0)) <= 0.2 * h
            )

        return {
            "brightness": brightness,
            "blur_variance": blur_variance,
            "face_detected": face_detected,
            "face_centered": face_centered,
            "bbox": bbox,
        }

    def _detect_and_crop(
        self, rgb: np.ndarray, require_face: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        h, w = rgb.shape[:2]
        results = self._detector.process(rgb)
        if not results.detections:
            if require_face:
                return None, None
            return rgb, (0, 0, w, h)

        det = results.detections[0]
        bb = det.location_data.relative_bounding_box
        pad = self.padding_ratio

        x1 = max(0, int((bb.xmin - pad * bb.width) * w))
        y1 = max(0, int((bb.ymin - pad * bb.height) * h))
        x2 = min(w, int((bb.xmin + (1 + pad) * bb.width) * w))
        y2 = min(h, int((bb.ymin + (1 + pad) * bb.height) * h))

        if x2 <= x1 or y2 <= y1:
            if require_face:
                return None, None
            return rgb, (0, 0, w, h)

        return rgb[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def _get_landmarks(self, rgb_crop: np.ndarray) -> Optional[np.ndarray]:
        results = self._mesh.process(rgb_crop)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        return np.array([[p.x, p.y] for p in lm], dtype=np.float32)

    def _align(self, rgb: np.ndarray, lm: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        left_pt = lm[_LEFT_EYE_IDX] * np.array([w, h])
        right_pt = lm[_RIGHT_EYE_IDX] * np.array([w, h])

        dx = right_pt[0] - left_pt[0]
        dy = right_pt[1] - left_pt[1]
        angle = math.degrees(math.atan2(dy, dx))

        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            rgb,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    @staticmethod
    def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
        img = rgb.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        return torch.from_numpy(img.transpose(2, 0, 1))


def extract_undereye_region(
    image_rgb: np.ndarray, lm: np.ndarray, side: str = "both"
) -> Optional[np.ndarray]:
    h, w = image_rgb.shape[:2]
    indices = {
        "left": _LEFT_UNDEREYE,
        "right": _RIGHT_UNDEREYE,
        "both": _LEFT_UNDEREYE + _RIGHT_UNDEREYE,
    }[side]
    pts = (lm[indices] * np.array([w, h])).astype(int)
    if len(pts) == 0:
        return None
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    x1, y1 = max(0, x1 - 5), max(0, y1 - 5)
    x2, y2 = min(w, x2 + 5), min(h, y2 + 5)
    if x2 <= x1 or y2 <= y1:
        return None
    return image_rgb[y1:y2, x1:x2]


def extract_cheek_region(
    image_rgb: np.ndarray, lm: np.ndarray, side: str = "both"
) -> Optional[np.ndarray]:
    h, w = image_rgb.shape[:2]
    indices = {
        "left": _LEFT_CHEEK,
        "right": _RIGHT_CHEEK,
        "both": _LEFT_CHEEK + _RIGHT_CHEEK,
    }[side]
    pts = (lm[indices] * np.array([w, h])).astype(int)
    if len(pts) == 0:
        return None
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    pad = 10
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return image_rgb[y1:y2, x1:x2]
