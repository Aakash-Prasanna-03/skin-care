"""
predict.py — Run skin analysis inference on one or more images.

Usage
-----
# Single image:
python predict.py --image path/to/face.jpg --checkpoint checkpoints/best_model.pth

# Batch (all images in a directory):
python predict.py --image_dir path/to/images/ --checkpoint checkpoints/best_model.pth

# Save annotated image:
python predict.py --image face.jpg --checkpoint best_model.pth --save_annotated out.jpg
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from inference.predictor import SkinPredictor
from utils.metrics import draw_skin_report


def parse_args():
    p = argparse.ArgumentParser(description="Skin analysis inference")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",     help="Path to a single image")
    group.add_argument("--image_dir", help="Directory of images")

    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--device",     default=None,  help="'cuda' or 'cpu' (auto-detect if omitted)")
    p.add_argument("--save_annotated", default=None, help="Path to save annotated image (single image only)")
    p.add_argument("--output_json",    default=None, help="Save all results as JSON")
    return p.parse_args()


def run_single(predictor: SkinPredictor, image_path: str, save_annotated: str = None):
    report = predictor.predict_from_path(image_path)
    print(f"\nImage: {image_path}")
    print(report)

    if save_annotated:
        bgr = cv2.imread(image_path)
        if bgr is not None and report.face_detected:
            annotated = draw_skin_report(bgr, report)
            cv2.imwrite(save_annotated, annotated)
            print(f"Annotated image saved → {save_annotated}")

    return report


def main():
    args = parse_args()

    predictor = SkinPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    results = {}

    if args.image:
        report = run_single(predictor, args.image, args.save_annotated)
        results[args.image] = report.to_dict()

    elif args.image_dir:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted(
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if os.path.splitext(f)[1].lower() in extensions
        )
        if not image_files:
            print(f"No images found in {args.image_dir}")
            return

        for path in image_files:
            report = run_single(predictor, path)
            results[path] = report.to_dict()

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {args.output_json}")


if __name__ == "__main__":
    main()
