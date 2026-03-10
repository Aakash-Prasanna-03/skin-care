"""
datasets/loaders.py

PyTorch datasets for acne, redness, texture, and dark-circle prediction.
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
from preprocessing.face_pipeline import FacePreprocessor, TARGET_SIZE


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


def _make_label_dict(
    acne: float = NAN,
    redness: float = NAN,
    texture: float = NAN,
    dark_circle: float = NAN,
) -> Dict[str, float]:
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


def _fallback_tensor_from_bgr(bgr) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    return (tensor - _MEAN) / _STD


class _BaseSkinDataset(Dataset):
    source_name = "base"

    def __init__(
        self,
        image_paths: List[str],
        labels: List[Dict[str, float]],
        train: bool = True,
        preprocessor: Optional[FacePreprocessor] = None,
    ):
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        path = self.image_paths[idx]
        bgr = cv2.imread(path)

        if bgr is None:
            tensor = torch.zeros(3, TARGET_SIZE[0], TARGET_SIZE[1], dtype=torch.float32)
        else:
            bgr = _bgr_augment(bgr, self.augment)
            tensor, _ = self._get_preprocessor()(bgr)
            if tensor is None:
                tensor = _fallback_tensor_from_bgr(bgr)

        label_tensors = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in self.labels[idx].items()
        }
        return tensor, label_tensors


class ACNE04Dataset(_BaseSkinDataset):
    source_name = "acne04"

    def __init__(self, root: str, train: bool = True, **kwargs):
        root = Path(root)
        image_paths, labels = [], []
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
                        labels.append(_make_label_dict(acne=score))
                break

        if not image_paths:
            raise FileNotFoundError(f"No ACNE04 images found in {root}")

        super().__init__(image_paths, labels, train=train, **kwargs)
        print(f"[ACNE04] Loaded {len(self)} images.")


class CelebADataset(_BaseSkinDataset):
    source_name = "celeba"
    ATTR_REDNESS = "Rosy_Cheeks"
    ATTR_DARK_CIRCLE = "Dark_Circles"
    _WEAK_PRESENT = 0.85
    _WEAK_ABSENT = 0.15

    def __init__(
        self,
        root: str,
        attr_file: str,
        sample_size: int = 4000,
        train: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"CelebA image directory not found: {root}")

        df = pd.read_csv(attr_file)
        df.columns = [column.strip() for column in df.columns]
        filename_col = df.columns[0]

        if sample_size < len(df):
            df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

        image_paths, labels = [], []
        for _, row in df.iterrows():
            filename = str(row[filename_col]).strip()
            path = root / filename
            if not path.exists():
                continue

            redness = self._WEAK_PRESENT if int(row.get(self.ATTR_REDNESS, -1)) == 1 else self._WEAK_ABSENT
            dark = self._WEAK_PRESENT if int(row.get(self.ATTR_DARK_CIRCLE, -1)) == 1 else self._WEAK_ABSENT
            image_paths.append(str(path))
            labels.append(_make_label_dict(redness=redness, dark_circle=dark))

        if not image_paths:
            raise FileNotFoundError(f"No CelebA images matched rows from {attr_file}")

        super().__init__(image_paths, labels, train=train, **kwargs)
        print(f"[CelebA] Loaded {len(self)} images.")


class FFHQDataset(_BaseSkinDataset):
    source_name = "ffhq"

    def __init__(
        self,
        root: str,
        sample_size: int = 1000,
        train: bool = True,
        seed: int = 42,
        pseudo_label_cache_dir: str = "data/cache/pseudo_labels",
        **kwargs,
    ):
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
            label_list.append(
                _make_label_dict(
                    texture=labels["texture_score"],
                    redness=labels["redness_score"],
                    dark_circle=labels["dark_circle_score"],
                )
            )

        if not image_paths:
            raise FileNotFoundError(f"No FFHQ pseudo-labels could be generated for {root}")

        super().__init__(image_paths, label_list, train=train, **kwargs)
        print(f"[FFHQ] Loaded {len(self)} images with pseudo-labels.")


class CombinedSkinDataset(ConcatDataset):
    def __init__(self, datasets: List[Dataset]):
        super().__init__(datasets)

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        keys = batch[0][1].keys()
        labels = {key: torch.stack([item[1][key] for item in batch]) for key in keys}
        return images, labels
