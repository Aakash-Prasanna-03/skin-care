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


class DistributionChecker:
    """Track prediction distributions per head to detect collapse and cross-head leakage."""

    HEADS = RunningMetrics.HEADS

    def __init__(self):
        self.reset()

    def reset(self):
        self._preds: Dict[str, List[float]] = {k: [] for k in self.HEADS}

    def update(self, preds: Dict[str, "torch.Tensor"]):
        for key in self.HEADS:
            if key not in preds:
                continue
            p = preds[key].detach().cpu()
            if p.dim() > 1:
                # Ordinal head: convert logits to probs then to scalar
                p = torch.sigmoid(p).mean(dim=-1)
            self._preds[key].extend(p.tolist())

    def compute(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for key in self.HEADS:
            vals = self._preds[key]
            if not vals:
                result[key] = {"n": 0}
                continue
            arr = np.array(vals)
            result[key] = {
                "n": len(arr),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "pct_low": float((arr < 0.1).mean()),     # % collapsed near 0
                "pct_high": float((arr > 0.9).mean()),    # % collapsed near 1
            }
        return result

    def cross_head_correlations(self) -> Dict[str, float]:
        """Pearson correlation between all head pairs. High correlation = potential leakage."""
        import itertools
        result = {}
        for a, b in itertools.combinations(self.HEADS, 2):
            va, vb = self._preds[a], self._preds[b]
            n = min(len(va), len(vb))
            if n < 10:
                continue
            aa, bb = np.array(va[:n]), np.array(vb[:n])
            if aa.std() < 1e-6 or bb.std() < 1e-6:
                result[f"{a}_vs_{b}"] = 0.0
                continue
            corr = float(np.corrcoef(aa, bb)[0, 1])
            result[f"{a}_vs_{b}"] = corr
        return result

    def summary_str(self) -> str:
        dist = self.compute()
        lines = ["Prediction Distribution:"]
        for k, v in dist.items():
            if v.get("n", 0) == 0:
                lines.append(f"  {k:<20s}: no data")
                continue
            flag = ""
            if v["pct_low"] > 0.5:
                flag = " ⚠️ COLLAPSED LOW"
            elif v["pct_high"] > 0.5:
                flag = " ⚠️ COLLAPSED HIGH"
            lines.append(
                f"  {k:<20s}: μ={v['mean']:.3f}  σ={v['std']:.3f}  "
                f"[{v['min']:.3f}, {v['max']:.3f}]  "
                f"lo={v['pct_low']:.0%} hi={v['pct_high']:.0%}{flag}"
            )

        corrs = self.cross_head_correlations()
        if corrs:
            lines.append("Cross-Head Correlations:")
            for pair, r in corrs.items():
                flag = " ⚠️ HIGH" if abs(r) > 0.8 else ""
                lines.append(f"  {pair:<45s}: r={r:+.3f}{flag}")

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
