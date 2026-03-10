import argparse
import json
from pathlib import Path

import cv2

from inference.predictor import SkinPredictor
from utils.metrics import draw_skin_report

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the best skin model on all uploaded images")
    parser.add_argument("--input_dir", default="uploads", help="Folder containing images to score")
    parser.add_argument("--output_dir", default="prediction_results", help="Folder to save annotated results")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--summary", default="summary.json", help="Summary JSON filename inside output_dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    image_paths = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        print(f"No supported images found in {input_dir}")
        return

    predictor = SkinPredictor(args.checkpoint)
    results = {}

    for image_path in image_paths:
        report = predictor.predict_from_path(image_path)
        results[image_path.name] = report.to_dict()
        print(f"\nImage: {image_path.name}")
        print(report)

        bgr = cv2.imread(str(image_path))
        if bgr is not None:
            annotated = draw_skin_report(bgr, report) if report.face_detected else bgr
            output_path = output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
            cv2.imwrite(str(output_path), annotated)

    summary_path = output_dir / args.summary
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nSaved annotated images to: {output_dir}")
    print(f"Saved summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()
