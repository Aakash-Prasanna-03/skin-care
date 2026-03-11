"""
train.py — Entry-point for training the skin analysis model.

Usage
-----
# Train with default config:
python train.py

# Override data paths:
python train.py --acne04_root /path/to/ACNE04 \
                --celeba_root /path/to/CelebA \
                --ffhq_root   /path/to/FFHQ

# Full options:
python train.py --help
"""

import argparse
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, DataConfig, ModelConfig, TrainConfig
from training.trainer import SkinModelTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train skin analysis model")

    # Data
    p.add_argument("--acne04_root",    default="data/acne_1024")
    p.add_argument("--celeba_root",    default="data/img_align_celeba")
    p.add_argument("--celeba_attr",    default="data/img_align_celeba/list_attr_celeba.csv")
    p.add_argument("--ffhq_root",      default="data/ffhq")
    p.add_argument("--celeba_samples", type=int, default=2000)
    p.add_argument("--ffhq_samples",   type=int, default=1000)
    p.add_argument("--pseudo_cache",   default="data/cache/pseudo_labels_region_v2")

    # Model
    p.add_argument("--backbone",       default="resnet18")
    p.add_argument("--shared_fc",      type=int, default=256)
    p.add_argument("--dropout",        type=float, default=0.3)

    # Training
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--loss",           choices=["smooth_l1", "mse"], default="smooth_l1")
    p.add_argument("--acne_weight",    type=float, default=2.0)
    p.add_argument("--redness_weight", type=float, default=1.2)
    p.add_argument("--texture_weight", type=float, default=0.8)
    p.add_argument("--dark_weight",    type=float, default=1.5)
    p.add_argument("--acne_dataset_weight",   type=float, default=4.0)
    p.add_argument("--celeba_dataset_weight", type=float, default=1.0)
    p.add_argument("--ffhq_dataset_weight",   type=float, default=1.5)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no_amp",         action="store_true")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_dir",        default="logs")
    p.add_argument("--resume",         default=None, help="Path to checkpoint to resume from")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        data=DataConfig(
            acne04_root=args.acne04_root,
            celeba_root=args.celeba_root,
            celeba_attr_file=args.celeba_attr,
            ffhq_root=args.ffhq_root,
            celeba_sample_size=args.celeba_samples,
            ffhq_sample_size=args.ffhq_samples,
            pseudo_label_cache_dir=args.pseudo_cache,
        ),
        model=ModelConfig(
            backbone=args.backbone,
            shared_fc_dim=args.shared_fc,
            dropout_rate=args.dropout,
        ),
        train=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            loss_fn=args.loss,
            acne_weight=args.acne_weight,
            redness_weight=args.redness_weight,
            texture_weight=args.texture_weight,
            dark_circle_weight=args.dark_weight,
            acne_dataset_weight=args.acne_dataset_weight,
            celeba_dataset_weight=args.celeba_dataset_weight,
            ffhq_dataset_weight=args.ffhq_dataset_weight,
            num_workers=args.num_workers,
            seed=args.seed,
            use_amp=not args.no_amp,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
        ),
    )
    cfg.setup_dirs()

    trainer = SkinModelTrainer(
        train_cfg=cfg.train,
        data_cfg=cfg.data,
        model_cfg=cfg.model,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()

