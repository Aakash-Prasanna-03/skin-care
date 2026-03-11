"""
datasets/loaders.py

Region-aware datasets for acne, redness, texture, and dark-circle prediction.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from datasets.pseudo_label_generator import PseudoLabelGenerator
from preprocessing.face_pipeline import FacePreprocessor, REGION_KEYS, TARGET_SIZE


_TRAIN_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.4),
    A.RandomBrightnessContrast(p=0.3),
])

_ACNE_LABEL_MAP = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
NAN = float("nan")
_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _make_label_dict(acne: float = NAN, redness: float = NAN, texture: float = NAN, dark_circle: float = NAN):
    return {
        "acne_score": acne,
        "redness_score": redness,
        "texture_score": texture,
        "dark_circle_score": dark_circle,
    }


def _bgr_augment(bgr, aug_pipeline):
    if aug_pipeline is None:
        return bgr
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = aug_pipeline(image=rgb)["image"]
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def _fallback_regions(bgr) -> Dict[str, torch.Tensor]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    tensor = (tensor - _MEAN) / _STD
    return {key: tensor.clone() for key in REGION_KEYS}


class _BaseSkinDataset(Dataset):
    source_name = "base"

    def __init__(self, image_paths: List[str], labels: List[Dict[str, float]], train: bool = True, preprocessor: Optional[FacePreprocessor] = None):
        assert len(image_paths) == len(labels)
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.train = train
        self.preprocessor = preprocessor
        self.augment = _TRAIN_AUG if train else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def subset(self, indices: List[int], train: bool) -> "_BaseSkinDataset":
        subset = copy.copy(self)
        subset.image_paths = [self.image_paths[i] for i in indices]
        subset.labels = [self.labels[i] for i in indices]
        subset.train = train
        subset.augment = _TRAIN_AUG if train else None
        subset.preprocessor = None
        return subset

    def _get_preprocessor(self) -> FacePreprocessor:
        if self.preprocessor is None:
            self.preprocessor = FacePreprocessor()
        return self.preprocessor

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        bgr = cv2.imread(path)

        if bgr is None:
            regions = {key: torch.zeros(3, TARGET_SIZE[0], TARGET_SIZE[1], dtype=torch.float32) for key in REGION_KEYS}
        else:
            bgr = _bgr_augment(bgr, self.augment)
            regions, _ = self._get_preprocessor()(bgr)
            if regions is None:
                regions = _fallback_regions(bgr)

        label_tensors = {key: torch.tensor(value, dtype=torch.float32) for key, value in self.labels[idx].items()}
        return regions, label_tensors


class ACNE04Dataset(_BaseSkinDataset):
    source_name = "acne04"

    def __init__(self, root: str, train: bool = True, pseudo_label_cache_dir: str = "data/cache/pseudo_labels_region_v2", **kwargs):
        root = Path(root)
        image_paths, acne_scores = [], []
        folder_map = {
            0: ["acne0_1024", "0"],
            1: ["acne1_1024", "1"],
            2: ["acne2_1024", "2"],
            3: ["acne3_1024", "3", "acne3_512_selection"],
        }

        for severity, score in _ACNE_LABEL_MAP.items():
            for folder_name in folder_map[severity]:
                subdir = root / folder_name
                if not subdir.is_dir():
                    continue
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    for path in subdir.glob(ext):
                        image_paths.append(str(path))
                        acne_scores.append(score)

        if not image_paths:
            raise FileNotFoundError(f"No ACNE04 images found in {root}")

        # Generate pseudo-labels for redness/texture/dark_circles from acne images
        generator = PseudoLabelGenerator(cache_dir=pseudo_label_cache_dir)
        pseudo = generator.generate_for_dataset(image_paths, dataset_name="acne04")

        labels = []
        for path, acne in zip(image_paths, acne_scores):
            pl = pseudo.get(path)
            if pl is not None:
                labels.append(_make_label_dict(
                    acne=acne,
                    redness=pl["redness_score"],
                    texture=pl["texture_score"],
                    dark_circle=pl["dark_circle_score"],
                ))
            else:
                labels.append(_make_label_dict(acne=acne))

        super().__init__(image_paths, labels, train=train, **kwargs)
        print(f"[ACNE04] Loaded {len(self)} images (with pseudo-labels for redness/texture/dark_circles).")


class CelebADataset(_BaseSkinDataset):
    source_name = "celeba"
    ATTR_REDNESS = "Rosy_Cheeks"
    ATTR_DARK_CIRCLE = "Bags_Under_Eyes"  # Fixed: was "Dark_Circles" which doesn't exist in CelebA

    @staticmethod
    def _smooth_label(present: bool, rng: random.Random) -> float:
        """Randomised soft labels instead of hard binary values."""
        if present:
            return 0.60 + 0.15 * rng.random()   # range [0.60, 0.75]
        return 0.10 + 0.10 * rng.random()        # range [0.10, 0.20]

    def __init__(self, root: str, attr_file: str, sample_size: int = 4000, train: bool = True, seed: int = 42, **kwargs):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"CelebA image directory not found: {root}")

        df = pd.read_csv(attr_file)
        df.columns = [column.strip() for column in df.columns]
        filename_col = df.columns[0]

        if sample_size < len(df):
            df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

        label_rng = random.Random(seed + 7)
        image_paths, labels = [], []
        for _, row in df.iterrows():
            filename = str(row[filename_col]).strip()
            path = root / filename
            if not path.exists():
                continue
            redness_present = int(row.get(self.ATTR_REDNESS, -1)) == 1
            dark_present = int(row.get(self.ATTR_DARK_CIRCLE, -1)) == 1
            redness = self._smooth_label(redness_present, label_rng)
            dark = self._smooth_label(dark_present, label_rng)
            image_paths.append(str(path))
            # acne=NaN: only ACNE04 provides acne supervision
            labels.append(_make_label_dict(redness=redness, dark_circle=dark))

        if not image_paths:
            raise FileNotFoundError(f"No CelebA images matched rows from {attr_file}")

        super().__init__(image_paths, labels, train=train, **kwargs)
        print(f"[CelebA] Loaded {len(self)} images.")


class FFHQDataset(_BaseSkinDataset):
    source_name = "ffhq"

    def __init__(self, root: str, sample_size: int = 1000, train: bool = True, seed: int = 42, pseudo_label_cache_dir: str = "data/cache/pseudo_labels_region_v2", **kwargs):
        root = Path(root)
        all_paths = sorted(
            str(path)
            for path in root.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not all_paths:
            raise FileNotFoundError(f"No images found under {root}")

        rng = random.Random(seed)
        if sample_size < len(all_paths):
            all_paths = rng.sample(all_paths, sample_size)

        generator = PseudoLabelGenerator(cache_dir=pseudo_label_cache_dir)
        pseudo = generator.generate_for_dataset(all_paths, dataset_name="ffhq")

        image_paths, label_list = [], []
        for path in all_paths:
            labels = pseudo.get(path)
            if labels is None:
                continue
            image_paths.append(path)
            # acne=NaN: only ACNE04 provides acne supervision
            label_list.append(_make_label_dict(
                texture=labels["texture_score"],
                redness=labels["redness_score"],
                dark_circle=labels["dark_circle_score"],
            ))

        if not image_paths:
            raise FileNotFoundError(f"No FFHQ pseudo-labels could be generated for {root}")

        super().__init__(image_paths, label_list, train=train, **kwargs)
        print(f"[FFHQ] Loaded {len(self)} images with pseudo-labels.")


class ExtremeSamplesDataset(_BaseSkinDataset):
    """Folder-based dataset for extreme/clear examples of redness and dark circles.

    Expected structure:
        data/extreme_samples/
            dark_circles_severe/   → dark_circle = 0.85–0.95
            redness_severe/        → redness = 0.85–0.95
            clear_skin/            → dark_circle/redness = 0.05–0.15, texture = 0.15-0.25, acne = 0.0
    """
    source_name = "extreme"

    # folder_name → dict of labels to set
    _FOLDER_MAP = {
        "dark_circles_severe": {"dark_circle_score": 0.80},
        "redness_severe":      {"redness_score": 0.80, "texture_score": 0.575},
        "clear_skin":          {"dark_circle_score": 0.10, "redness_score": 0.10, "texture_score": 0.20, "acne_score": 0.0},
    }

    # Ordinal heads must use exact values — jitter would break class mapping
    _NO_JITTER_KEYS = {"acne_score"}

    def __init__(self, root: str, train: bool = True, pseudo_label_cache_dir: str = "data/cache/pseudo_labels_region_v2", seed: int = 42, **kwargs):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Extreme samples directory not found: {root}")

        rng = random.Random(seed + 99)
        image_paths, primary_labels = [], []

        for folder_name, labels_dict in self._FOLDER_MAP.items():
            folder = root / folder_name
            if not folder.is_dir():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                for path in folder.glob(ext):
                    # Jitter ±0.05 for natural variance for each label
                    jittered_labels = {}
                    for k, center in labels_dict.items():
                        if k in self._NO_JITTER_KEYS:
                            jittered_labels[k] = center  # exact ordinal value
                        else:
                            jittered_labels[k] = max(0.0, min(1.0, center + (rng.random() - 0.5) * 0.10))
                    image_paths.append(str(path))
                    primary_labels.append(jittered_labels)

        if not image_paths:
            raise FileNotFoundError(f"No extreme sample images found in {root}")

        # Generate pseudo-labels for OTHER heads (texture, etc.)
        generator = PseudoLabelGenerator(cache_dir=pseudo_label_cache_dir)
        pseudo = generator.generate_for_dataset(image_paths, dataset_name="extreme")

        labels = []
        for path, primary_lbls in zip(image_paths, primary_labels):
            pl = pseudo.get(path, {})
            label = _make_label_dict()  # all NaN by default

            # Set the primary labels from folder
            for k, val in primary_lbls.items():
                label[k] = val

            # Fill in pseudo-labels for other heads where available
            if pl:
                if "redness_score" not in primary_lbls and "redness_score" in pl:
                    label["redness_score"] = pl["redness_score"]
                if "dark_circle_score" not in primary_lbls and "dark_circle_score" in pl:
                    label["dark_circle_score"] = pl["dark_circle_score"]
                if "texture_score" not in primary_lbls and "texture_score" in pl:
                    label["texture_score"] = pl["texture_score"]

            labels.append(label)

        super().__init__(image_paths, labels, train=train, **kwargs)
        print(f"[ExtremeSamples] Loaded {len(self)} images from {root}.")

class CombinedSkinDataset(ConcatDataset):
    def __init__(self, datasets: List[Dataset]):
        super().__init__(datasets)

    @staticmethod
    def collate_fn(batch):
        region_keys = batch[0][0].keys()
        images = {key: torch.stack([item[0][key] for item in batch]) for key in region_keys}
        label_keys = batch[0][1].keys()
        labels = {key: torch.stack([item[1][key] for item in batch]) for key in label_keys}
        return images, labels

