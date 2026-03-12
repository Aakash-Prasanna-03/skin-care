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

    # Cap clear-skin (class 0) to a fixed maximum of 340 images.
    MAX_CLASS0: int = 340

    def __init__(self, root: str, train: bool = True, pseudo_label_cache_dir: str = "data/cache/pseudo_labels_region_v2", **kwargs):
        root = Path(root)
        # Collect images per severity class first
        per_class: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}
        folder_map = {
            0: ["acne0_1024", "0"],
            1: ["acne1_1024", "1"],
            2: ["acne2_1024", "2"],
            3: ["acne3_1024", "3", "acne3_512_selection"],
        }

        for severity in folder_map:
            for folder_name in folder_map[severity]:
                subdir = root / folder_name
                if not subdir.is_dir():
                    continue
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    for path in subdir.glob(ext):
                        per_class[severity].append(str(path))

        # Balance: cap class-0 so it doesn't dominate
        if len(per_class[0]) > self.MAX_CLASS0:
            rng = random.Random(42)
            per_class[0] = rng.sample(per_class[0], self.MAX_CLASS0)
            print(f"[ACNE04] Downsampled class-0 (clear) from original to {self.MAX_CLASS0} images (fixed cap)")

        image_paths, acne_scores = [], []
        for severity, paths in per_class.items():
            score = _ACNE_LABEL_MAP[severity]
            for p in paths:
                image_paths.append(p)
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

    # folder_name → dict of labels to set (used for clear_skin only now)
    _CLEAR_LABELS = {"dark_circle_score": 0.05, "redness_score": 0.05, "texture_score": 0.05, "acne_score": 0.0}

    # For ranked folders: pseudo-label key → (min, max) range to scale into
    _RANKED_FOLDERS = {
        "dark_circles_severe": {"dark_circle_score": (0.50, 0.90)},
        "redness_severe":      {"redness_score": (0.40, 0.80), "texture_score": (0.30, 0.60)},
    }

    # Ordinal heads must use exact values — jitter would break class mapping
    _NO_JITTER_KEYS = {"acne_score"}

    def __init__(self, root: str, train: bool = True, pseudo_label_cache_dir: str = "data/cache/pseudo_labels_region_v2", seed: int = 42, **kwargs):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Extreme samples directory not found: {root}")

        rng = random.Random(seed + 99)

        # Generate pseudo-labels first so we can rank images
        all_image_paths = []
        image_folder_map = {}  # path → folder_name
        for folder_name in list(self._RANKED_FOLDERS.keys()) + ["clear_skin"]:
            folder = root / folder_name
            if not folder.is_dir():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                for path in folder.glob(ext):
                    path_str = str(path)
                    all_image_paths.append(path_str)
                    image_folder_map[path_str] = folder_name

        if not all_image_paths:
            raise FileNotFoundError(f"No extreme sample images found in {root}")

        generator = PseudoLabelGenerator(cache_dir=pseudo_label_cache_dir)
        pseudo = generator.generate_for_dataset(all_image_paths, dataset_name="extreme")

        # For ranked folders, collect pseudo scores per key to rank and rescale
        folder_paths = {}  # folder_name → [path, ...]
        for path_str, folder_name in image_folder_map.items():
            folder_paths.setdefault(folder_name, []).append(path_str)

        # Compute rescaled labels for ranked folders
        ranked_labels = {}  # path → {key: rescaled_value}
        for folder_name, key_ranges in self._RANKED_FOLDERS.items():
            paths = folder_paths.get(folder_name, [])
            if not paths:
                continue
            for key, (lo, hi) in key_ranges.items():
                # Collect raw pseudo scores for this key across the folder
                raw_scores = []
                for p in paths:
                    pl = pseudo.get(p, {})
                    raw_scores.append(pl.get(key, 0.5))

                # Rank-based rescaling: sort, assign evenly spaced values in [lo, hi]
                n = len(raw_scores)
                sorted_indices = sorted(range(n), key=lambda i: raw_scores[i])
                for rank, idx in enumerate(sorted_indices):
                    if n > 1:
                        scaled = lo + (hi - lo) * (rank / (n - 1))
                    else:
                        scaled = (lo + hi) / 2.0
                    # Add small jitter ±0.03
                    scaled = max(0.0, min(1.0, scaled + (rng.random() - 0.5) * 0.06))
                    ranked_labels.setdefault(paths[idx], {})[key] = scaled

        # Build final label list
        image_paths, labels = [], []
        for path_str in all_image_paths:
            folder_name = image_folder_map[path_str]
            pl = pseudo.get(path_str, {})
            label = _make_label_dict()  # all NaN by default

            if folder_name == "clear_skin":
                # Fixed labels for clear skin
                for k, center in self._CLEAR_LABELS.items():
                    if k in self._NO_JITTER_KEYS:
                        label[k] = center
                    else:
                        label[k] = max(0.0, min(1.0, center + (rng.random() - 0.5) * 0.10))
            elif folder_name in self._RANKED_FOLDERS:
                # Use rank-scaled labels for primary keys
                rl = ranked_labels.get(path_str, {})
                for k, val in rl.items():
                    label[k] = val

            # Fill in pseudo-labels for non-primary heads
            if pl:
                for k in ("redness_score", "dark_circle_score", "texture_score"):
                    if label[k] != label[k]:  # is NaN
                        if k in pl:
                            label[k] = pl[k]

            image_paths.append(path_str)
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

