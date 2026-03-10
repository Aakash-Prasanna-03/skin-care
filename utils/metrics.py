"""
utils/metrics.py  —  Evaluation metrics for regression heads.
utils/visualize.py — Visualization helpers.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# metrics.py
# ──────────────────────────────────────────────────────────────────────────────

class RunningMetrics:
    """
    Accumulates per-head regression metrics over batches.

    Tracked: MAE, RMSE (masked — NaN labels excluded).
    """

    HEADS = ("acne_score", "redness_score", "texture_score", "dark_circle_score")

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum_ae:  Dict[str, float] = {k: 0.0 for k in self.HEADS}
        self._sum_se:  Dict[str, float] = {k: 0.0 for k in self.HEADS}
        self._counts:  Dict[str, int]   = {k: 0   for k in self.HEADS}

    def update(
        self,
        preds: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ):
        for key in self.HEADS:
            if key not in preds or key not in labels:
                continue
            p = preds[key].detach().cpu()
            t = labels[key].detach().cpu()
            mask = ~torch.isnan(t)
            if mask.sum() == 0:
                continue
            diff = (p[mask] - t[mask]).abs()
            self._sum_ae[key]  += diff.sum().item()
            self._sum_se[key]  += (diff ** 2).sum().item()
            self._counts[key]  += int(mask.sum().item())

    def compute(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for key in self.HEADS:
            n = self._counts[key]
            if n == 0:
                result[key] = {"mae": float("nan"), "rmse": float("nan"), "n": 0}
            else:
                result[key] = {
                    "mae":  self._sum_ae[key] / n,
                    "rmse": (self._sum_se[key] / n) ** 0.5,
                    "n":    n,
                }
        return result

    def summary_str(self) -> str:
        metrics = self.compute()
        lines = ["Metric Results:"]
        for k, v in metrics.items():
            lines.append(
                f"  {k:<20s}: MAE={v['mae']:.4f}  RMSE={v['rmse']:.4f}  (n={v['n']})"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# visualize.py
# ──────────────────────────────────────────────────────────────────────────────

def draw_skin_report(
    bgr_image: "np.ndarray",
    report: "object",          # SkinReport
    font_scale: float = 0.6,
) -> "np.ndarray":
    """
    Overlay skin metric bars on the original BGR image.

    Parameters
    ----------
    bgr_image : np.ndarray  (H, W, 3) BGR
    report    : SkinReport

    Returns
    -------
    annotated : np.ndarray BGR
    """
    import cv2  # local import to avoid hard dependency in tests

    img = bgr_image.copy()
    h, w = img.shape[:2]

    # Semi-transparent panel on right side
    panel_w = 260
    overlay = img.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    metrics = [
        ("Acne",        report.acne_score,        (0,   150, 255)),
        ("Redness",     report.redness_score,      (0,    80, 220)),
        ("Texture",     report.texture_score,      (0,   200, 100)),
        ("Dark Circles",report.dark_circle_score,  (150,  80,   0)),
        ("Overall",     report.overall_score,      (255, 255, 255)),
    ]

    y = 30
    cv2.putText(img, "Skin Analysis", (w - panel_w + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for label, score, color in metrics:
        y += 35
        cv2.putText(img, f"{label}", (w - panel_w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)
        bar_x = w - panel_w + 10
        bar_y = y + 8
        bar_len = int((panel_w - 20) * score)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + panel_w - 20, bar_y + 10),
                      (80, 80, 80), -1)
        if bar_len > 0:
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_len, bar_y + 10),
                          color, -1)
        cv2.putText(img, f"{score:.2f}", (w - 45, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    return img
