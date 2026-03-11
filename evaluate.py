"""
evaluate.py — Evaluate model quality on a manually curated test set.

Usage
-----
# Create a test set CSV (image_path, acne, redness, texture, dark_circle):
python evaluate.py --test_csv data/test_set.csv --checkpoint checkpoints/best_model.pth

# Or just run on a folder and inspect predicted distributions:
python evaluate.py --image_dir uploads --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from inference.predictor import SkinPredictor
from utils.metrics import DistributionChecker


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate skin analysis model")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--test_csv", help="CSV with columns: image_path, acne, redness, texture, dark_circle")
    group.add_argument("--image_dir", help="Directory of images (distribution check only, no ground truth)")

    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--device", default=None, help="'cuda' or 'cpu'")
    p.add_argument("--output_json", default=None, help="Save detailed results as JSON")
    return p.parse_args()


def evaluate_with_ground_truth(predictor: SkinPredictor, csv_path: str) -> Dict:
    """Evaluate against manually annotated ground truth."""
    df = pd.read_csv(csv_path)
    required = {"image_path"}
    optional = {"acne", "redness", "texture", "dark_circle"}
    assert required.issubset(set(df.columns)), f"CSV must have columns: {required}"

    head_map = {
        "acne": "acne_score",
        "redness": "redness_score",
        "texture": "texture_score",
        "dark_circle": "dark_circle_score",
    }
    present_heads = [h for h in optional if h in df.columns]

    errors: Dict[str, List[float]] = {h: [] for h in present_heads}
    predictions = []
    dist_checker = DistributionChecker()

    for _, row in df.iterrows():
        path = str(row["image_path"])
        if not Path(path).exists():
            print(f"  [SKIP] {path} not found")
            continue

        report = predictor.predict_from_path(path)
        if report.error:
            print(f"  [FAIL] {path}: {report.error}")
            continue

        pred_dict = report.to_dict()
        record = {"image": path}

        for h in present_heads:
            gt = float(row[h])
            pred = float(pred_dict[head_map[h]])
            errors[h].append(abs(pred - gt))
            record[f"{h}_gt"] = gt
            record[f"{h}_pred"] = pred
            record[f"{h}_err"] = abs(pred - gt)

        record["overall_pred"] = report.overall_score
        predictions.append(record)

        import torch
        fake_preds = {head_map[h]: torch.tensor([pred_dict[head_map[h]]]) for h in present_heads}
        dist_checker.update(fake_preds)

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Images evaluated: {len(predictions)}")

    results = {}
    for h in present_heads:
        if not errors[h]:
            continue
        arr = np.array(errors[h])
        mae = float(arr.mean())
        rmse = float(np.sqrt((arr ** 2).mean()))
        within_015 = float((arr < 0.15).mean())
        print(f"\n  {h.upper()}")
        print(f"    MAE:          {mae:.4f}")
        print(f"    RMSE:         {rmse:.4f}")
        print(f"    Within ±0.15: {within_015:.0%}")
        results[h] = {"mae": mae, "rmse": rmse, "within_015": within_015, "n": len(arr)}

    print(f"\n{dist_checker.summary_str()}")
    return {"metrics": results, "predictions": predictions}


def evaluate_distribution_only(predictor: SkinPredictor, image_dir: str) -> Dict:
    """No ground truth: just check prediction distributions."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(
        str(p) for p in Path(image_dir).iterdir()
        if p.suffix.lower() in extensions
    )
    if not image_files:
        print(f"No images found in {image_dir}")
        return {}

    dist_checker = DistributionChecker()
    predictions = []

    print(f"Running inference on {len(image_files)} images...")
    for path in image_files:
        report = predictor.predict_from_path(path)
        if report.error:
            continue
        pred_dict = report.to_dict()
        predictions.append({"image": path, **{k: pred_dict[k] for k in pred_dict if "score" in k}})

        import torch
        fake_preds = {}
        for key in ("acne_score", "redness_score", "texture_score", "dark_circle_score"):
            fake_preds[key] = torch.tensor([pred_dict[key]])
        dist_checker.update(fake_preds)

    print(f"\n{dist_checker.summary_str()}")
    return {"predictions": predictions}


def main():
    args = parse_args()
    predictor = SkinPredictor(checkpoint_path=args.checkpoint, device=args.device)

    if args.test_csv:
        results = evaluate_with_ground_truth(predictor, args.test_csv)
    else:
        results = evaluate_distribution_only(predictor, args.image_dir)

    if args.output_json and results:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved → {args.output_json}")


if __name__ == "__main__":
    main()
