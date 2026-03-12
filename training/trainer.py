"""
training/trainer.py

Balanced multi-task training with acne-focused sampling and checkpointing.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import DataConfig, ModelConfig, TrainConfig
from datasets.loaders import ACNE04Dataset, CelebADataset, CombinedSkinDataset, ExtremeSamplesDataset, FFHQDataset
from models.skin_model import SkinAnalysisModel
from utils.metrics import DistributionChecker


class MaskedRegressionLoss(nn.Module):
    def __init__(self, loss_type: str = "smooth_l1", beta: float = 0.1):
        super().__init__()
        if loss_type == "smooth_l1":
            self._fn = nn.SmoothL1Loss(reduction="none", beta=beta)
        elif loss_type == "mse":
            self._fn = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, int]:
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True), 0
        loss = self._fn(pred[mask], target[mask])
        return loss.mean(), int(mask.sum().item())


class OrdinalCrossEntropy(nn.Module):
    """Loss for CORAL-style ordinal classification.

    Converts a scalar target (0.0 / 0.33 / 0.66 / 1.0) to ordinal class index
    (0/1/2/3), then computes BCE on cumulative binary indicators.

    Label smoothing shifts hard 0/1 targets towards 0.5 to prevent overconfidence.
    """
    # Map from regression-style target → ordinal class index
    _CLASS_TARGETS = torch.tensor([0.0, 0.33, 0.66, 1.0])  # ordinal class centers

    def __init__(self, label_smoothing: float = 0.05):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """logits: (B, 3) raw logits, target: (B,) with values in {0.0, 0.33, 0.66, 1.0, NaN}"""
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True), 0

        valid_logits = logits[mask]            # (N, 3)
        valid_targets = target[mask]           # (N,)

        # Convert regression target → nearest ordinal class index
        # Uses argmin distance so any value (even slightly jittered) maps correctly
        class_targets = self._CLASS_TARGETS.to(valid_targets.device)  # (4,)
        dists = torch.abs(valid_targets.unsqueeze(-1) - class_targets.unsqueeze(0))  # (N, 4)
        ordinal_classes = dists.argmin(dim=-1)  # (N,)

        # Build cumulative binary targets: [y>=1, y>=2, y>=3]
        thresholds = torch.arange(1, 4, device=logits.device).float().unsqueeze(0)  # (1, 3)
        binary_targets = (ordinal_classes.unsqueeze(-1) >= thresholds).float()          # (N, 3)

        # Label smoothing: shift hard 0/1 towards 0.5 to prevent overconfidence
        if self.label_smoothing > 0:
            binary_targets = binary_targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss = F.binary_cross_entropy_with_logits(valid_logits, binary_targets, reduction="mean")
        return loss, int(mask.sum().item())


class SkinModelTrainer:
    HEAD_KEYS = ("acne_score", "redness_score", "texture_score", "dark_circle_score")

    def __init__(self, train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig):
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg

        self._set_seed(train_cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Using device: {self.device}")

        self.model = SkinAnalysisModel(
            backbone=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            shared_fc_dim=model_cfg.shared_fc_dim,
            dropout_rate=model_cfg.dropout_rate,
        ).to(self.device)
        stats = self.model.count_parameters()
        print(f"[Trainer] Parameters - total: {stats['total']:,}, trainable: {stats['trainable']:,}")

        self.criterion = MaskedRegressionLoss(loss_type=train_cfg.loss_fn, beta=train_cfg.beta)
        self.acne_criterion = OrdinalCrossEntropy()
        self.head_weights = {
            "acne_score": train_cfg.acne_weight,
            "redness_score": train_cfg.redness_weight,
            "texture_score": train_cfg.texture_weight,
            "dark_circle_score": train_cfg.dark_circle_weight,
        }
        self.dataset_weights = {
            "acne04": train_cfg.acne_dataset_weight,
            "celeba": train_cfg.celeba_dataset_weight,
            "ffhq": train_cfg.ffhq_dataset_weight,
            "extreme": 3.0,  # Only source of real severe dark_circle / redness images
        }

        # Differential learning rates: slower backbone/shared trunk, faster task heads
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(name.startswith(p) for p in ("acne_head.", "redness_head.", "texture_head.", "dark_circle_head.")):
                head_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": train_cfg.learning_rate * train_cfg.backbone_lr_factor},
                {"params": head_params, "lr": train_cfg.learning_rate},
            ],
            weight_decay=train_cfg.weight_decay,
        )
        print(f"[Trainer] LR groups: backbone={train_cfg.learning_rate * train_cfg.backbone_lr_factor:.2e}, heads={train_cfg.learning_rate:.2e}")

        self.train_loader, self.val_loader = self._build_loaders()
        total_steps = max(1, len(self.train_loader) * train_cfg.epochs)
        self.scheduler = self._build_scheduler(total_steps)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=train_cfg.use_amp and self.device.type == "cuda"
        )

        Path(train_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(train_cfg.log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=train_cfg.log_dir)
        self.best_val_loss = float("inf")
        self.best_val_mae = float("inf")
        self.best_acne_val_loss = float("inf")
        self.global_step = 0
        self.dist_checker = DistributionChecker()

    def _build_base_datasets(self) -> List:
        datasets = []

        try:
            datasets.append(ACNE04Dataset(
                root=self.data_cfg.acne04_root,
                train=False,
                pseudo_label_cache_dir=self.data_cfg.pseudo_label_cache_dir,
            ))
        except FileNotFoundError as exc:
            print(f"[Warning] ACNE04 skipped: {exc}")

        try:
            datasets.append(
                CelebADataset(
                    root=self.data_cfg.celeba_root,
                    attr_file=self.data_cfg.celeba_attr_file,
                    sample_size=self.data_cfg.celeba_sample_size,
                    train=False,
                    seed=self.train_cfg.seed,
                )
            )
        except FileNotFoundError as exc:
            print(f"[Warning] CelebA skipped: {exc}")

        try:
            datasets.append(
                FFHQDataset(
                    root=self.data_cfg.ffhq_root,
                    sample_size=self.data_cfg.ffhq_sample_size,
                    train=False,
                    seed=self.train_cfg.seed,
                    pseudo_label_cache_dir=self.data_cfg.pseudo_label_cache_dir,
                )
            )
        except FileNotFoundError as exc:
            print(f"[Warning] FFHQ skipped: {exc}")

        try:
            datasets.append(ExtremeSamplesDataset(
                root=self.data_cfg.extreme_samples_root,
                train=False,
                pseudo_label_cache_dir=self.data_cfg.pseudo_label_cache_dir,
                seed=self.train_cfg.seed,
            ))
        except FileNotFoundError as exc:
            print(f"[Warning] Extreme samples skipped: {exc}")

        if not datasets:
            raise RuntimeError("No datasets found. Check your data paths.")

        return datasets

    def _split_dataset(self, dataset) -> Tuple:
        size = len(dataset)
        if size == 1:
            return dataset.subset([0], train=True), dataset.subset([], train=False)

        n_val = max(1, int(round(size * self.train_cfg.val_split)))
        n_val = min(n_val, size - 1)
        indices = list(range(size))
        rng = random.Random(self.train_cfg.seed + size)
        rng.shuffle(indices)
        val_indices = sorted(indices[:n_val])
        train_indices = sorted(indices[n_val:])
        return dataset.subset(train_indices, train=True), dataset.subset(val_indices, train=False)

    def _build_train_sampler(self, train_datasets: List) -> WeightedRandomSampler:
        sample_weights = []
        for dataset in train_datasets:
            dataset_weight = self.dataset_weights.get(dataset.source_name, 1.0)
            per_sample_weight = dataset_weight / max(1, len(dataset))
            sample_weights.extend([per_sample_weight] * len(dataset))
        weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
        return WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)

    def _build_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_datasets = []
        val_datasets = []
        for dataset in self._build_base_datasets():
            train_split, val_split = self._split_dataset(dataset)
            if len(train_split) > 0:
                train_datasets.append(train_split)
            if len(val_split) > 0:
                val_datasets.append(val_split)

        if not train_datasets:
            raise RuntimeError("No training samples available after dataset split.")
        if not val_datasets:
            raise RuntimeError("No validation samples available after dataset split.")

        train_ds = CombinedSkinDataset(train_datasets)
        val_ds = CombinedSkinDataset(val_datasets)
        train_sampler = self._build_train_sampler(train_datasets)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.train_cfg.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=self.train_cfg.num_workers,
            collate_fn=CombinedSkinDataset.collate_fn,
            pin_memory=self.device.type == "cuda",
            drop_last=len(train_ds) >= self.train_cfg.batch_size,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
            num_workers=self.train_cfg.num_workers,
            collate_fn=CombinedSkinDataset.collate_fn,
            pin_memory=self.device.type == "cuda",
        )

        print(f"[Trainer] Train: {len(train_ds)} | Val: {len(val_ds)}")
        print(f"[Trainer] Dataset weights: {self.dataset_weights}")
        return train_loader, val_loader

    def _build_scheduler(self, total_steps: int):
        warmup_steps = len(self.train_loader) * self.train_cfg.warmup_epochs

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    _REGRESSION_KEYS = ("redness_score", "texture_score", "dark_circle_score")

    def _get_acne_weight(self, epoch: int) -> float:
        """Linearly ramp acne weight from initial to final over training."""
        base = self.train_cfg.acne_weight
        final = self.train_cfg.acne_weight_final
        progress = epoch / max(1, self.train_cfg.epochs - 1)
        return base + (final - base) * progress

    def _compute_total_loss(
        self,
        preds: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = torch.tensor(0.0, device=self.device)
        per_head = {}

        # Ordinal loss for acne with progressive weight
        acne_loss, _ = self.acne_criterion(preds["acne_score"], labels["acne_score"].to(self.device))
        acne_w = self._get_acne_weight(epoch)
        total = total + acne_w * acne_loss
        per_head["acne_score"] = acne_loss.item()

        # Regression loss for other heads
        for key in self._REGRESSION_KEYS:
            head_loss, _ = self.criterion(preds[key], labels[key].to(self.device))
            total = total + self.head_weights[key] * head_loss
            per_head[key] = head_loss.item()

        return total, per_head

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [train]", leave=False)
        for images, labels in progress:
            images = {key: value.to(self.device, non_blocking=True) for key, value in images.items()}

            with torch.amp.autocast(
                device_type="cuda", enabled=self.train_cfg.use_amp and self.device.type == "cuda"
            ):
                scores = self.model(images)
                loss, per_head = self._compute_total_loss(scores.to_dict(), labels, epoch=epoch)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            self.writer.add_scalar("train/acne_weight_effective", self._get_acne_weight(epoch), self.global_step)
            for key, value in per_head.items():
                self.writer.add_scalar(f"train/{key}", value, self.global_step)
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        self.dist_checker.reset()

        acne_val_loss_sum = 0.0
        acne_val_batches = 0

        # MAE accumulators for regression heads
        mae_sums = {k: 0.0 for k in self._REGRESSION_KEYS}
        mae_counts = {k: 0 for k in self._REGRESSION_KEYS}

        for images, labels in tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [val]", leave=False):
            images = {key: value.to(self.device, non_blocking=True) for key, value in images.items()}
            with torch.amp.autocast(
                device_type="cuda", enabled=self.train_cfg.use_amp and self.device.type == "cuda"
            ):
                scores = self.model(images)
                preds = scores.to_dict()
                loss, per_head = self._compute_total_loss(preds, labels, epoch=epoch)

            # Accumulate per-head MAE for regression heads
            for key in self._REGRESSION_KEYS:
                target = labels[key].to(self.device)
                mask = ~torch.isnan(target)
                if mask.sum() > 0:
                    mae_sums[key] += (preds[key][mask] - target[mask]).abs().sum().item()
                    mae_counts[key] += int(mask.sum().item())

            self.dist_checker.update(preds)
            total_loss += loss.item()
            n_batches += 1
            if per_head.get("acne_score", 0.0) > 0:
                acne_val_loss_sum += per_head["acne_score"]
                acne_val_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_acne_loss = acne_val_loss_sum / max(1, acne_val_batches)

        # Average MAE across regression heads (weight-invariant metric)
        head_maes = []
        for key in self._REGRESSION_KEYS:
            if mae_counts[key] > 0:
                head_mae = mae_sums[key] / mae_counts[key]
                head_maes.append(head_mae)
                self.writer.add_scalar(f"val/mae_{key}", head_mae, epoch)
        avg_mae = sum(head_maes) / max(1, len(head_maes))

        self.writer.add_scalar("val/loss", avg_loss, epoch)
        self.writer.add_scalar("val/acne_loss", avg_acne_loss, epoch)
        self.writer.add_scalar("val/avg_mae", avg_mae, epoch)
        for key, value in per_head.items():
            self.writer.add_scalar(f"val/{key}", value, epoch)
        return avg_loss, avg_acne_loss, avg_mae

    def train(self):
        print(f"[Trainer] Starting training for {self.train_cfg.epochs} epochs.")
        print(f"[Trainer] Acne weight: {self.train_cfg.acne_weight:.1f} -> {self.train_cfg.acne_weight_final:.1f} (progressive)")
        for epoch in range(self.train_cfg.epochs):
            start = time.time()
            train_loss = self._train_epoch(epoch)
            val_loss, acne_val_loss, val_mae = self._val_epoch(epoch)
            elapsed = time.time() - start
            lr = self.scheduler.get_last_lr()[0]

            acne_w = self._get_acne_weight(epoch)
            print(
                f"Epoch {epoch + 1:3d}/{self.train_cfg.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"acne_val={acne_val_loss:.4f} | val_mae={val_mae:.4f} | "
                f"acne_w={acne_w:.1f} | lr={lr:.2e} | {elapsed:.1f}s"
            )
            self.writer.add_scalar("lr", lr, epoch)

            # Print distribution diagnostics every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(self.dist_checker.summary_str())

            # Best model selection: use weight-invariant MAE
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self._save_checkpoint(epoch, val_loss, best=True)
                print(f"  New best model (val_mae={val_mae:.4f})")

            # Track best weighted loss separately
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            # Save best acne checkpoint separately
            if acne_val_loss < self.best_acne_val_loss:
                self.best_acne_val_loss = acne_val_loss
                self._save_checkpoint(epoch, val_loss, best=False, tag="best_acne")
                print(f"  New best acne model (acne_val_loss={acne_val_loss:.4f})")

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_loss, best=False)

        self.writer.close()
        print(f"[Trainer] Done. Best val MAE: {self.best_val_mae:.4f}")
        print(f"[Trainer] Best val loss: {self.best_val_loss:.4f}")
        print(f"[Trainer] Best acne val loss: {self.best_acne_val_loss:.4f}")

    def _save_checkpoint(self, epoch: int, val_loss: float, best: bool, tag: str = ""):
        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "model_config": {
                "backbone": self.model_cfg.backbone,
                "shared_fc_dim": self.model_cfg.shared_fc_dim,
                "dropout_rate": self.model_cfg.dropout_rate,
            },
            "train_config": {
                "head_weights": self.head_weights,
                "dataset_weights": self.dataset_weights,
            },
        }
        if best:
            path = os.path.join(self.train_cfg.checkpoint_dir, "best_model.pth")
        elif tag:
            path = os.path.join(self.train_cfg.checkpoint_dir, f"{tag}.pth")
        else:
            path = os.path.join(self.train_cfg.checkpoint_dir, f"checkpoint_epoch{epoch + 1:03d}.pth")
        torch.save(checkpoint, path)
        if best:
            print(f"  New best model saved -> {path} (val_loss={val_loss:.4f})")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"[Trainer] Loaded checkpoint from {path} (epoch {checkpoint['epoch'] + 1})")

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
