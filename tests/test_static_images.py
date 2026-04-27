"""Run trained YOLOv8 model on static images."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test trained YOLOv8 model with static images.")
    parser.add_argument(
        "--model",
        default="runs/detect/train/weights/best.pt",
        help="Path to trained model weights (.pt).",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image file or directory path.",
    )
    parser.add_argument(
        "--output",
        default="runs/detect/static_test",
        help="Output directory for annotated predictions.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    return parser.parse_args()


def run_inference(model_path: Path, source: Path, output_dir: Path, conf: float, imgsz: int) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    model = YOLO(str(model_path))
    model.predict(
        source=str(source),
        conf=conf,
        imgsz=imgsz,
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )
    print(f"Inference complete. Results saved to: {output_dir}")


def main() -> None:
    args = parse_args()
    run_inference(
        model_path=Path(args.model),
        source=Path(args.source),
        output_dir=Path(args.output),
        conf=args.conf,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()

