"""Run trained YOLOv8 model on webcam or RTSP live stream."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test trained YOLOv8 model with live camera or RTSP stream.")
    parser.add_argument(
        "--model",
        default="best.pt",
        help="Path to trained model weights (.pt).",
    )
    parser.add_argument(
        "--source",
        default="0",
        help='Camera index (e.g. "0") or stream URL (e.g. rtsp://...).',
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


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def run_live_inference(model_path: Path, source: str, conf: float, imgsz: int) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(parse_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    prev_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stream ended or frame read failed.")
                break

            results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
                    
            annotated = results[0].plot()

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.imshow("YOLOv8 Live Test", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run_live_inference(
        model_path=Path(args.model),
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()

