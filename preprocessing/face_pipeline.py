"""
preprocessing/face_pipeline.py

Region-aware face preprocessing for:
- full face (acne/context)
- cheek region (redness)
- under-eye region (dark circles)
- texture patch (texture)
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

_LEFT_UNDEREYE = [144, 145, 153, 154, 155, 157, 158, 159, 160, 7, 163, 173]
_RIGHT_UNDEREYE = [373, 374, 380, 381, 382, 384, 385, 386, 387, 249, 390, 398]

_LEFT_CHEEK = [205, 36, 47, 50, 101, 116, 123, 147, 187, 207]
_RIGHT_CHEEK = [425, 266, 277, 280, 330, 345, 352, 376, 411, 427]

_FOREHEAD = [9, 10, 151, 337, 108]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_SIZE = (224, 224)
REGION_KEYS = ("full_face", "cheek", "undereye", "texture")


class FacePreprocessor:
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

    def __call__(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face_crop, _ = self._detect_and_crop(rgb, require_face=True)
        if face_crop is None:
            return None, None

        landmarks = self._get_landmarks(face_crop)
        if landmarks is None:
            return None, None

        region_tensors = self.extract_region_tensors(face_crop, landmarks)
        landmarks_px = (
            landmarks * np.array(face_crop.shape[:2][::-1], dtype=np.float32)
        ).astype(int)
        return region_tensors, landmarks_px

    def extract_region_tensors(self, face_rgb: np.ndarray, landmarks: np.ndarray) -> Dict[str, torch.Tensor]:
        aligned_face = self._align(face_rgb, landmarks)
        full_face = cv2.resize(aligned_face, self.target_size, interpolation=cv2.INTER_AREA)

        cheek = extract_cheek_region(face_rgb, landmarks, side="both")
        undereye = extract_undereye_region(face_rgb, landmarks, side="both")
        texture = extract_texture_region(face_rgb, landmarks)

        if cheek is None:
            cheek = _central_patch(face_rgb, top=0.35, bottom=0.72, left=0.18, right=0.82)
        if undereye is None:
            undereye = _central_patch(face_rgb, top=0.22, bottom=0.46, left=0.18, right=0.82)
        if texture is None:
            texture = _central_patch(face_rgb, top=0.12, bottom=0.42, left=0.22, right=0.78)

        return {
            "full_face": self._to_tensor(full_face),
            "cheek": self._to_tensor(cv2.resize(cheek, self.target_size, interpolation=cv2.INTER_AREA)),
            "undereye": self._to_tensor(cv2.resize(undereye, self.target_size, interpolation=cv2.INTER_AREA)),
            "texture": self._to_tensor(cv2.resize(texture, self.target_size, interpolation=cv2.INTER_AREA)),
        }

    def preprocess_for_inference(self, bgr: np.ndarray):
        tensors, _ = self(bgr)
        if tensors is None:
            return None
        return {key: value.unsqueeze(0) for key, value in tensors.items()}

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

    def _detect_and_crop(self, rgb: np.ndarray, require_face: bool = True):
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


def _central_patch(image_rgb: np.ndarray, top: float, bottom: float, left: float, right: float) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    y1 = max(0, int(h * top))
    y2 = min(h, int(h * bottom))
    x1 = max(0, int(w * left))
    x2 = min(w, int(w * right))
    if x2 <= x1 or y2 <= y1:
        return image_rgb.copy()
    return image_rgb[y1:y2, x1:x2]


def _crop_from_indices(image_rgb: np.ndarray, lm: np.ndarray, indices, pad: int = 8):
    h, w = image_rgb.shape[:2]
    pts = (lm[indices] * np.array([w, h])).astype(int)
    if len(pts) == 0:
        return None
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return image_rgb[y1:y2, x1:x2]


def extract_undereye_region(image_rgb: np.ndarray, lm: np.ndarray, side: str = "both") -> Optional[np.ndarray]:
    indices = {
        "left": _LEFT_UNDEREYE,
        "right": _RIGHT_UNDEREYE,
        "both": _LEFT_UNDEREYE + _RIGHT_UNDEREYE,
    }[side]
    return _crop_from_indices(image_rgb, lm, indices, pad=20)


def extract_cheek_region(image_rgb: np.ndarray, lm: np.ndarray, side: str = "both") -> Optional[np.ndarray]:
    indices = {
        "left": _LEFT_CHEEK,
        "right": _RIGHT_CHEEK,
        "both": _LEFT_CHEEK + _RIGHT_CHEEK,
    }[side]
    return _crop_from_indices(image_rgb, lm, indices, pad=25)


def extract_texture_region(image_rgb: np.ndarray, lm: np.ndarray) -> Optional[np.ndarray]:
    region = _crop_from_indices(image_rgb, lm, _FOREHEAD, pad=10)
    if region is not None:
        return region
    cheek = extract_cheek_region(image_rgb, lm, side="both")
    if cheek is not None:
        return cheek
    return _central_patch(image_rgb, top=0.15, bottom=0.45, left=0.22, right=0.78)
