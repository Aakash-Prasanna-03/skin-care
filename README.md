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
skin_analysis/
├── config.py                        # Centralised config dataclasses
├── train.py                         # Training entry-point
├── predict.py                       # Inference entry-point
├── requirements.txt
│
├── preprocessing/
│   ├── __init__.py
│   └── face_pipeline.py             # MediaPipe face detection, alignment, crop, normalise
│
├── datasets/
│   ├── __init__.py
│   ├── loaders.py                   # ACNE04Dataset, CelebADataset, FFHQDataset, CombinedSkinDataset
│   └── pseudo_label_generator.py   # LBP texture, redness (LAB a*), dark-circle contrast
│
├── models/
│   ├── __init__.py
│   └── skin_model.py                # SkinAnalysisModel (EfficientNet-B2 + 4 heads)
│
├── training/
│   ├── __init__.py
│   └── trainer.py                   # SkinModelTrainer (masked loss, AMP, cosine LR, checkpointing)
│
├── inference/
│   ├── __init__.py
│   └── predictor.py                 # SkinPredictor + SkinReport
│
└── utils/
    ├── __init__.py
    └── metrics.py                   # RunningMetrics (MAE/RMSE), draw_skin_report
```

---

## Dataset Setup

### ACNE04
```
data/ACNE04/
    0/   ← clear (label 0.0)
    1/   ← mild  (label 0.33)
    2/   ← moderate (label 0.66)
    3/   ← severe (label 1.0)
```

### CelebA
```
data/CelebA/
    img_align_celeba/
        000001.jpg
        ...
    list_attr_celeba.csv   ← must include Rosy_Cheeks and Dark_Circles columns
```

### FFHQ
```
data/FFHQ/
    images/   ← or any nested subdirectory structure
        00000.png
        ...
```

---

## Installation

```bash
pip install -r requirements.txt
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
python predict.py --image face.jpg --checkpoint checkpoints/best_model.pth

# Save annotated output:
python predict.py --image face.jpg --checkpoint checkpoints/best_model.pth \
                  --save_annotated result.jpg

# Batch directory + JSON output:
python predict.py --image_dir ./faces/ \
                  --checkpoint checkpoints/best_model.pth \
                  --output_json results.json
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
