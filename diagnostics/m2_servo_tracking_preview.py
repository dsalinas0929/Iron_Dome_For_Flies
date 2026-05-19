"""M2 software-only validation: detection -> tracking -> servo angle output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Allow running as `python diagnostics/m2_servo_tracking_preview.py` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.pan_tilt import PanTiltConfig, PanTiltController, ServoCommand
from tracking.controller import TargetTracker, TrackerConfig, TrackerOutput
from tracking.kalman_filter import KalmanConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO detection + Kalman prediction + servo-angle simulation. "
            "No physical PCA9685 control required."
        )
    )
    parser.add_argument("--source", default="0", help='Camera index, file path, or RTSP URL.')
    parser.add_argument(
        "--backend",
        choices=["default", "ffmpeg", "gstreamer"],
        default="default",
        help="OpenCV backend for stream input.",
    )
    parser.add_argument(
        "--model",
        default="models/best_v2_com.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--config",
        default="config/m2_tracking_config.json",
        help="JSON config path for detection/tracking parameters.",
    )
    parser.add_argument(
        "--save-video",
        default="",
        help="Optional output video path (e.g. result/m2_tracking_preview.mp4).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 means run until stream ends or q).",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print pan/tilt outputs every N frames.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2 window display (useful for headless run).",
    )
    servo_group = parser.add_mutually_exclusive_group()
    servo_group.add_argument(
        "--servo-enable",
        action="store_true",
        help="Force-enable servo output regardless of config file value.",
    )
    servo_group.add_argument(
        "--servo-disable",
        action="store_true",
        help="Force-disable servo output regardless of config file value.",
    )
    return parser.parse_args()


def parse_source(raw_source: str) -> int | str:
    return int(raw_source) if raw_source.isdigit() else raw_source


def parse_backend(raw_backend: str) -> int | None:
    if raw_backend == "default":
        return None
    if raw_backend == "ffmpeg":
        return getattr(cv2, "CAP_FFMPEG", None)
    if raw_backend == "gstreamer":
        return getattr(cv2, "CAP_GSTREAMER", None)
    return None


def load_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def pick_best_detection(
    result: object,
    width: int,
    height: int,
    target_class_id: int | None,
) -> tuple[tuple[float, float] | None, float | None]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    indices = list(range(len(boxes)))
    if target_class_id is not None and boxes.cls is not None:
        filtered = []
        for idx in indices:
            cls_id = int(boxes.cls[idx].item())
            if cls_id == target_class_id:
                filtered.append(idx)
        indices = filtered

    if not indices:
        return None, None

    best_idx = max(indices, key=lambda i: float(boxes.conf[i].item()))
    conf = float(boxes.conf[best_idx].item())
    x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[best_idx].detach().cpu().tolist()]

    cx = ((x1 + x2) * 0.5) / max(width, 1)
    cy = ((y1 + y2) * 0.5) / max(height, 1)
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    return (cx, cy), conf


def draw_overlay(
    frame: np.ndarray,
    tracking_state: TrackerOutput,
    servo_command: ServoCommand | None,
    servo_enabled: bool,
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()

    if tracking_state.measured_x is not None and tracking_state.measured_y is not None:
        mx = int(tracking_state.measured_x * w)
        my = int(tracking_state.measured_y * h)
        cv2.circle(out, (mx, my), 5, (0, 255, 0), -1)
        cv2.putText(out, "measured", (mx + 8, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    px = int(tracking_state.predicted_x * w)
    py = int(tracking_state.predicted_y * h)
    cv2.circle(out, (px, py), 6, (0, 0, 255), 2)
    cv2.putText(out, "predicted", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    lines = [
        f"frame: {frame_idx}",
        f"fps: {fps:.1f}",
        f"pan(deg): {tracking_state.pan_angle_deg:.2f}",
        f"tilt(deg): {tracking_state.tilt_angle_deg:.2f}",
        f"prediction_only: {tracking_state.prediction_only}",
        f"servo_enabled: {servo_enabled}",
    ]
    if tracking_state.detection_confidence is not None:
        lines.append(f"detection_conf: {tracking_state.detection_confidence:.2f}")
    if servo_command is not None:
        lines.append(
            f"servo_ticks: pan={servo_command.pan_ticks} tilt={servo_command.tilt_ticks} emitted={servo_command.emitted}"
        )

    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (12, 30 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def build_video_writer(path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_fps = fps if fps > 1.0 else 20.0
    return cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        safe_fps,
        (width, height),
    )


def run() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    detection_cfg = cfg.get("detection", {})
    tracking_cfg = cfg.get("tracking", {})
    kalman_cfg = cfg.get("kalman", {})
    servo_cfg = cfg.get("servo_output", {})

    detection_conf = float(detection_cfg.get("conf", 0.25))
    detection_imgsz = int(detection_cfg.get("imgsz", 640))
    target_class = detection_cfg.get("target_class_id", None)
    target_class_id = int(target_class) if target_class is not None else None

    tracker = TargetTracker(
        tracker_config=TrackerConfig.from_dict(tracking_cfg if isinstance(tracking_cfg, dict) else {}),
        kalman_config=KalmanConfig(
            process_var=float(kalman_cfg.get("process_var", 0.02)),
            measurement_var=float(kalman_cfg.get("measurement_var", 0.005)),
            initial_position_var=float(kalman_cfg.get("initial_position_var", 1.0)),
            initial_velocity_var=float(kalman_cfg.get("initial_velocity_var", 1.0)),
        ),
    )
    servo_cfg_dict = servo_cfg if isinstance(servo_cfg, dict) else {}
    servo_payload = dict(servo_cfg_dict)
    if args.servo_enable:
        servo_payload["enabled"] = True
    if args.servo_disable:
        servo_payload["enabled"] = False
    pan_tilt = PanTiltController(PanTiltConfig.from_dict(servo_payload))

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(str(model_path))

    source = parse_source(args.source)
    backend = parse_backend(args.backend)
    if backend is None:
        if args.backend != "default":
            raise RuntimeError(f"Backend '{args.backend}' is unavailable in this OpenCV build.")
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source} (backend={args.backend})")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    writer = None
    if args.save_video:
        writer = build_video_writer(args.save_video, width, height, source_fps)
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open video writer: {args.save_video}")

    window_name = "M2 Servo Tracking Preview"
    frame_idx = 0
    start_time = time.time()
    display = not args.no_display

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stream ended or frame read failed.")
                break

            frame_idx += 1
            result = model.predict(frame, conf=detection_conf, imgsz=detection_imgsz, verbose=False)[0]
            measurement, det_conf = pick_best_detection(result, frame.shape[1], frame.shape[0], target_class_id)
            tracking_state = tracker.update(measurement=measurement, detection_confidence=det_conf)
            servo_command = pan_tilt.move_to(
                tracking_state.pan_angle_deg,
                tracking_state.tilt_angle_deg,
            )

            annotated = result.plot()
            elapsed = max(time.time() - start_time, 1e-6)
            fps = frame_idx / elapsed
            overlay = draw_overlay(
                annotated,
                tracking_state,
                servo_command=servo_command,
                servo_enabled=pan_tilt.config.enabled,
                frame_idx=frame_idx,
                fps=fps,
            )

            if frame_idx % max(args.print_every, 1) == 0:
                print(
                    f"frame={frame_idx} fps={fps:.1f} "
                    f"pan={tracking_state.pan_angle_deg:.2f} "
                    f"tilt={tracking_state.tilt_angle_deg:.2f} "
                    f"prediction_only={tracking_state.prediction_only} "
                    f"servo_ticks=({servo_command.pan_ticks},{servo_command.tilt_ticks}) "
                    f"servo_emitted={servo_command.emitted}"
                )

            if display:
                cv2.imshow(window_name, overlay)
            if writer is not None:
                writer.write(overlay)

            if display and cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        pan_tilt.close()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
