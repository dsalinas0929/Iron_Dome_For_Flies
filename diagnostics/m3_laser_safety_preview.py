"""M3 validation: detection -> tracking -> servo -> safety -> laser gating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Allow running as `python diagnostics/m3_laser_safety_preview.py` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.pan_tilt import PanTiltConfig, PanTiltController, ServoCommand
from laser.controller import LaserConfig, LaserController, LaserStatus
from safety.interlocks import SafetyConfig, SafetyInterlocks
from tracking.controller import TargetTracker, TrackerConfig, TrackerOutput
from tracking.kalman_filter import KalmanConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end M3 preview with safety interlocks and laser gate decision. "
            "Use mock drivers locally; enable hardware drivers on Jetson."
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
        help="JSON config path for detection/tracking/servo/safety/laser parameters.",
    )
    parser.add_argument(
        "--save-video",
        nargs="?",
        const="result/m3_laser_safety_preview.mp4",
        default=None,
        help="Save preview as MP4 (optional custom path).",
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
        help="Print status every N frames.",
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
    laser_group = parser.add_mutually_exclusive_group()
    laser_group.add_argument(
        "--laser-enable",
        action="store_true",
        help="Force-enable laser output regardless of config file value.",
    )
    laser_group.add_argument(
        "--laser-disable",
        action="store_true",
        help="Force-disable laser output regardless of config file value.",
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


# Select best detection by confidence, optionally filtering by target class ID
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


def build_video_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_fps = fps if fps > 1.0 else 20.0
    return cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        safe_fps,
        (width, height),
    )


def normalize_mp4_path(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.suffix.lower() != ".mp4":
        path = path.with_suffix(".mp4")
    return path


def draw_overlay(
    frame: np.ndarray,
    tracking_state: TrackerOutput,
    servo_command: ServoCommand,
    laser_status: LaserStatus,
    safety: SafetyInterlocks,
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    if tracking_state.measured_x is not None and tracking_state.measured_y is not None:
        mx = int(tracking_state.measured_x * w)
        my = int(tracking_state.measured_y * h)
        cv2.circle(out, (mx, my), 5, (0, 255, 0), -1)
        cv2.putText(out, "measured", (mx + 8, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    px = int(tracking_state.predicted_x * w)
    py = int(tracking_state.predicted_y * h)
    cv2.circle(out, (px, py), 6, (0, 0, 255), 2)
    cv2.putText(out, "predicted", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    fire_allowed = safety.last_decision.allow_fire
    status_color = (0, 255, 0) if fire_allowed else (0, 0, 255)
    status_text = "FIRE: ALLOWED" if fire_allowed else "FIRE: BLOCKED"

    cv2.putText(out, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
    if fire_allowed:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)

    reasons_text = ", ".join(safety.last_decision.reasons[:3])
    lines = [
        f"frame: {frame_idx}",
        f"fps: {fps:.1f}",
        f"pan(deg): {tracking_state.pan_angle_deg:.2f}",
        f"tilt(deg): {tracking_state.tilt_angle_deg:.2f}",
        f"lock_frames: {safety.last_decision.lock_frames}",
        f"prediction_only: {tracking_state.prediction_only}",
        f"laser_on: {laser_status.is_on} ({laser_status.reason})",
        f"laser_emitted: {laser_status.emitted}",
        f"servo_ticks: pan={servo_command.pan_ticks} tilt={servo_command.tilt_ticks}",
        f"safety_reason: {reasons_text}",
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (12, 58 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def run() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    detection_cfg = cfg.get("detection", {})
    tracking_cfg = cfg.get("tracking", {})
    kalman_cfg = cfg.get("kalman", {})
    servo_cfg = cfg.get("servo_output", {})
    safety_cfg = cfg.get("safety", {})
    laser_cfg = cfg.get("laser_output", cfg.get("laser", {}))

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

    servo_payload = dict(servo_cfg if isinstance(servo_cfg, dict) else {})
    if args.servo_enable:
        servo_payload["enabled"] = True
    if args.servo_disable:
        servo_payload["enabled"] = False
    pan_tilt = PanTiltController(PanTiltConfig.from_dict(servo_payload))

    interlocks = SafetyInterlocks(config=SafetyConfig.from_dict(safety_cfg if isinstance(safety_cfg, dict) else {}))

    laser_payload = dict(laser_cfg if isinstance(laser_cfg, dict) else {})
    if args.laser_enable:
        laser_payload["enabled"] = True
    if args.laser_disable:
        laser_payload["enabled"] = False
    laser = LaserController(LaserConfig.from_dict(laser_payload))

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

    save_path = normalize_mp4_path(args.save_video)
    writer = None
    if save_path is not None:
        writer = build_video_writer(save_path, width, height, source_fps)
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open video writer: {save_path}")
        print(f"Saving M3 preview video to: {save_path}")

    window_name = "M3 Laser Safety Preview"
    display = not args.no_display
    frame_idx = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stream ended or frame read failed.")
                break

            frame_idx += 1
            result = model.predict(frame, conf=detection_conf, imgsz=detection_imgsz, verbose=False)[0]
            measurement, det_conf = pick_best_detection(
                result,
                width=frame.shape[1],
                height=frame.shape[0],
                target_class_id=target_class_id,
            )
            tracking_state = tracker.update(measurement=measurement, detection_confidence=det_conf)
            servo_command = pan_tilt.move_to(tracking_state.pan_angle_deg, tracking_state.tilt_angle_deg)

            target_xy = (
                measurement
                if measurement is not None
                else (tracking_state.predicted_x, tracking_state.predicted_y)
            )
            decision = interlocks.evaluate(
                detection_confidence=det_conf,
                pan_angle=tracking_state.pan_angle_deg,
                tilt_angle=tracking_state.tilt_angle_deg,
                target_xy=target_xy,
                prediction_only=tracking_state.prediction_only,
            )
            laser_status = laser.update(decision.allow_fire)

            annotated = result.plot()
            elapsed = max(time.time() - start_time, 1e-6)
            fps = frame_idx / elapsed
            overlay = draw_overlay(
                annotated,
                tracking_state=tracking_state,
                servo_command=servo_command,
                laser_status=laser_status,
                safety=interlocks,
                frame_idx=frame_idx,
                fps=fps,
            )

            if frame_idx % max(args.print_every, 1) == 0:
                print(
                    f"frame={frame_idx} fps={fps:.1f} "
                    f"fire_allowed={decision.allow_fire} "
                    f"reasons={decision.reasons} "
                    f"laser_on={laser_status.is_on} "
                    f"laser_reason={laser_status.reason}"
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
        laser.close()
        if writer is not None:
            writer.release()
            print(f"Saved M3 preview video: {save_path} (frames={frame_idx})")
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run()

