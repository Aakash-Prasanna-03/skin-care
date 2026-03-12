"""
config.py — Central configuration for the skin analysis pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import os


@dataclass
class DataConfig:
    # ── Dataset roots ──────────────────────────────────────────────────────────
    acne04_root: str = "data/acne_1024"
    celeba_root: str = "data/img_align_celeba"
    celeba_attr_file: str = "data/img_align_celeba/list_attr_celeba.csv"
    ffhq_root: str = "data/ffhq"
    extreme_samples_root: str = "data/extreme_samples"

    # ── Sampling ───────────────────────────────────────────────────────────────
    celeba_sample_size: int = 2_000
    ffhq_sample_size: int = 1_000

    # ── Preprocessing ─────────────────────────────────────────────────────────
    image_size: Tuple[int, int] = (260, 260)
    imagenet_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    # ── Cache directory for pseudo-labels ─────────────────────────────────────
    pseudo_label_cache_dir: str = "data/cache/pseudo_labels_region_v2"


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b2"
    pretrained: bool = True
    shared_fc_dim: int = 256
    dropout_rate: float = 0.3
    num_heads: int = 4                     # acne / redness / texture / dark_circles


@dataclass
class TrainConfig:
    # ── Paths ──────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # ── Hyper-parameters ──────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"          # "cosine" | "step"
    warmup_epochs: int = 3

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn: str = "smooth_l1"            # "smooth_l1" | "mse"
    beta: float = 0.1                     # Smooth L1 beta (lower = more L1-like, outlier-robust)

    # ── Per-head loss weights ─────────────────────────────────────────────────
    acne_weight: float = 3.0
    redness_weight: float = 1.2
    texture_weight: float = 0.8
    dark_circle_weight: float = 1.5

    # Progressive acne weight: linearly ramps from acne_weight to this value
    acne_weight_final: float = 4.5

    # Differential LR: backbone/shared trunk use lr * this factor
    backbone_lr_factor: float = 0.3

    acne_dataset_weight: float = 4.0      # Higher: ACNE04 is the only acne source
    celeba_dataset_weight: float = 1.0
    ffhq_dataset_weight: float = 1.5

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
    val_split: float = 0.1

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp: bool = True


@dataclass
class InferenceConfig:
    checkpoint_path: str = "checkpoints/best_model.pth"
    device: str = "cuda"                  # "cuda" | "cpu"

    # ── Overall score weights ─────────────────────────────────────────────────
    acne_weight: float = 0.35
    redness_weight: float = 0.25
    texture_weight: float = 0.20
    dark_circle_weight: float = 0.20


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def setup_dirs(self):
        """Create all necessary directories."""
        for d in [
            self.train.checkpoint_dir,
            self.train.log_dir,
            self.data.pseudo_label_cache_dir,
        ]:
            os.makedirs(d, exist_ok=True)


# ── Convenience singleton ──────────────────────────────────────────────────────
cfg = Config()

