# Skin Analysis ML Pipeline

A production-grade facial skin analysis system that predicts four skin health metrics from a single face image using a multi-task deep learning model.

## Metrics Predicted

| Metric | Range | Description |
|---|---|---|
| `acne_score` | 0–1 | Acne severity (0 = clear, 1 = severe) |
| `redness_score` | 0–1 | Redness / inflammation |
| `texture_score` | 0–1 | Skin roughness / texture |
| `dark_circle_score` | 0–1 | Under-eye dark circles |
| `overall_score` | 0–1 | Weighted composite health score |

**Lower values = healthier skin.**

---

## Architecture

```
Input (256×256×3)
    ↓
EfficientNet-B2 (ImageNet pretrained)  — backbone
    ↓  Global Average Pool  →  feature vector (1408-d)
    ↓  Dropout → Linear(1408, 256) → BN → ReLU   — shared trunk
    ↓
  ┌──────────────────────────────────────────────────┐
  │  head_acne         → Linear(256,64) → ReLU → Linear(64,1) → Sigmoid  │
  │  head_redness      → (same)                                           │
  │  head_texture      → (same)                                           │
  │  head_dark_circle  → (same)                                           │
  └──────────────────────────────────────────────────┘
```

---

## Project Structure

```
Project/
├── config.py                        # Centralised config dataclasses
├── train.py                         # Training entry-point
├── evaluate.py                      # Model evaluation on test sets
├── run_uploaded_images.py           # Batch inference on uploaded images
├── Pyproject.toml                   # Project metadata & dependencies
│
├── preprocessing/
│   └── face_pipeline.py             # MediaPipe face detection, alignment, crop, normalise
│
├── datasets/
│   ├── loaders.py                   # ACNE04Dataset, CelebADataset, FFHQDataset, CombinedSkinDataset
│   └── pseudo_label_generator.py    # LBP texture, redness (LAB a*), dark-circle contrast
│
├── models/
│   └── skin_model.py                # SkinAnalysisModel (EfficientNet-B2 + 4 heads)
│
├── training/
│   └── trainer.py                   # SkinModelTrainer (masked loss, AMP, cosine LR, checkpointing)
│
├── inference/
│   ├── predict.py                   # CLI inference entry-point
│   └── predictor.py                 # SkinPredictor + SkinReport
│
├── utils/
│   └── metrics.py                   # RunningMetrics (MAE/RMSE), draw_skin_report
│
├── data/                            # Datasets (git-ignored, see Dataset Setup)
├── checkpoints/                     # Saved model weights (git-ignored)
└── logs/                            # TensorBoard logs (git-ignored)
```

---

## Dataset Setup

### ACNE04
```
data/acne_1024/
    acne0_1024/   ← clear (label 0.0)
    acne1_1024/   ← mild  (label 0.33)
    acne2_1024/   ← moderate (label 0.66)
    acne3_1024/   ← severe (label 1.0)
```

### CelebA
```
data/img_align_celeba/
    000001.jpg
    ...
    list_attr_celeba.csv   ← must include Rosy_Cheeks and Dark_Circles columns
```

### FFHQ
```
data/ffhq/
    00000/
    01000/
    ...
    09000/
```

---

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install torch torchvision timm opencv-python mediapipe numpy albumentations \
    pandas scikit-learn scikit-image Pillow tqdm matplotlib tensorboard PyYAML
```

> Requires Python ≥ 3.9. CUDA 11.8+ recommended for GPU training.

---

## Training

```bash
# Default paths (data/ directory):
python train.py

# Custom dataset paths:
python train.py \
  --acne04_root /data/ACNE04 \
  --celeba_root /data/CelebA \
  --celeba_attr /data/CelebA/list_attr_celeba.csv \
  --ffhq_root   /data/FFHQ \
  --epochs 50 \
  --batch_size 32

# Resume from checkpoint:
python train.py --resume checkpoints/checkpoint_epoch020.pth
```

### Key training options

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 3e-4 | AdamW learning rate |
| `--loss` | smooth_l1 | Loss function (smooth_l1 or mse) |
| `--celeba_samples` | 15000 | CelebA sample size |
| `--ffhq_samples` | 10000 | FFHQ sample size |
| `--no_amp` | — | Disable mixed-precision training |

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs/
```

---

## Inference

```bash
# Single image:
python inference/predict.py --image face.jpg --checkpoint checkpoints/best_model.pth

# Batch directory + JSON output:
python run_uploaded_images.py --image_dir ./uploads/ \
                              --checkpoint checkpoints/best_model.pth
```

### Python API

```python
from inference.predictor import SkinPredictor

predictor = SkinPredictor("checkpoints/best_model.pth")
report = predictor.predict_from_path("face.jpg")

print(report)
# ┌─ Skin Analysis Report ──────────────────────┐
# │  Acne severity:     0.123        Clear │
# │  Redness:           0.241        Mild  │
# │  Texture roughness: 0.187        Clear │
# │  Dark circles:      0.312        Mild  │
# ├─────────────────────────────────────────────┤
# │  Overall skin score: 0.213       Clear │
# └─────────────────────────────────────────────┘

scores = report.to_dict()
```

---

## Overall Score Formula

```
overall_score = 0.35 × acne_score
              + 0.25 × redness_score
              + 0.20 × texture_score
              + 0.20 × dark_circle_score
```

---

## Training Details

### Loss Function
Smooth L1 loss (Huber loss) with NaN masking — only valid labels for each sample contribute to the loss, enabling true multi-task learning across heterogeneous datasets.

### Label Sources

| Dataset | acne | redness | texture | dark_circles |
|---|---|---|---|---|
| ACNE04 | ✓ supervised | — | — | — |
| CelebA | — | ✓ weak | — | ✓ weak |
| FFHQ | — | ✓ pseudo | ✓ pseudo | ✓ pseudo |

### Augmentations (training)
- Horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- Gaussian noise
- Gaussian blur
- Shift / scale / rotate
- Random brightness & contrast

---

## Pseudo-Label Generation

FFHQ pseudo-labels are generated automatically and cached to disk on first run:

- **Texture score**: 50% LBP entropy + 50% gradient magnitude variance (normalised)
- **Redness score**: Mean LAB a* channel over cheek landmarks, mapped [128, 200] → [0, 1]
- **Dark circle score**: (cheek_brightness − undereye_brightness) / cheek_brightness
