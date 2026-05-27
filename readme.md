# Iron Dome for Flies

Autonomous insect detection and targeting MVP that combines real-time computer vision, pan-tilt tracking, and laser actuation with safety interlocks.

## Project Overview

This project builds a hobby-level outdoor prototype that:

- detects flying insects from a live camera feed,
- tracks a selected target using a 2-axis pan-tilt mechanism,
- activates a TTL laser only when lock/safety conditions are satisfied.

Core pipeline:

`Camera -> YOLO Detection -> Target Selection -> Servo Tracking -> Laser Activation -> Safety Control`

## Milestones and Deliverables

### Milestone 1: Real-Time Fly Detection

- Run YOLOv8 (or similar) on edge hardware (Jetson Nano or Raspberry Pi 4).
- Detect small insects/flies from live video.
- Target performance: 15-20 FPS (not strict perfection).
- Deliverable: demo video with live detection and bounding boxes.

### Milestone 2: Pan-Tilt Tracking System

- Integrate PCA9685 servo controller.
- Control 2-axis pan-tilt mount.
- Convert detection centroid into servo pan/tilt commands.
- Achieve smooth real-time target following.
- Deliverable: demo video of tracking a moving insect.

### Milestone 3: Laser Targeting + Safety System

- Integrate TTL laser control via GPIO.
- Fire only when target lock condition is true.
- Enforce safety constraints:
  - no upward firing,
  - restricted firing zones,
  - basic interlock logic.
- Outdoor test demonstration required.
- Deliverable: full system demo video.

## Hardware BOM

- NVIDIA Jetson Nano Developer Kit (preferred for GPU acceleration).
- Logitech C920 USB camera.
- Adafruit PCA9685 16-channel PWM/servo driver (I2C).
- SunFounder pan-tilt kit.
- 2x MG996R high-torque servos.
- 5W 450nm TTL laser module.
- 5V 4A power supply for Jetson Nano.
- 6V external battery pack for servos (isolated from compute power).
- Jumper wires, breadboard or terminal block.

Recommended safety additions:

- Physical laser kill switch.
- Basic weather-protective enclosure.

## Software Stack

- Python
- Ultralytics YOLOv8
- OpenCV
- Jetson.GPIO (or equivalent GPIO library on selected hardware)
- I2C/PCA9685 control library

## Key Technical Requirements

- Real-time performance target: 15+ FPS.
- Stable end-to-end detection-to-tracking behavior.
- Low-latency servo response with minimal jitter.
- Safe laser control with software interlocks.
- Modular, maintainable project structure.

## Proposed Module Structure

As implementation begins, organize code by responsibility:

- `vision/` - camera ingest, inference, filtering, target selection.
- `tracking/` - centroid-to-angle mapping, PID/smoothing logic.
- `control/` - pan/tilt servo driver + PCA9685 abstraction.
- `laser/` - TTL trigger control, lock conditions, interlocks.
- `safety/` - firing-zone validation, angle limits, emergency states.
- `app/` - main loop, config loading, telemetry/logging.
- `tests/` - unit/integration tests for vision, tracking, and safety logic.

## Development Plan

1. Stand up camera + YOLO inference loop with FPS metrics.
2. Add deterministic target selection (single target lock strategy).
3. Implement pan-tilt mapping with smoothing and servo limits.
4. Gate laser trigger behind lock confidence + safety checks.
5. Run field tests, tune thresholds, and document behavior.

## Safety Notes

- Do not allow laser fire when pan/tilt is outside permitted angles.
- Disable firing if detection confidence or lock stability drops.
- Keep an emergency stop path (software and physical kill switch).
- Never run outdoor tests without controlled backstop and exclusion zone.

## Current Repository Status

Initial project scaffold is in place with module folders for `app`, `vision`, `tracking`, `control`, `laser`, `safety`, `tests`, and `config`.

## Quick Test with Static Images

Run your trained YOLO model (`best.pt`) against a single image or an image folder:

```bash
python3 tests/test_static_images.py --source /path/to/images_or_image
```

Optional arguments:

- `--model` (default: `runs/detect/train/weights/best.pt`)
- `--output` (default: `runs/detect/static_test`)
- `--conf` (default: `0.25`)
- `--imgsz` (default: `640`)

## Quick Test with Live Camera or RTSP

Webcam (device 0):

```bash
python3 tests/test_live_stream.py --source 0
```

RTSP stream:

```bash
python3 tests/test_live_stream.py --source "rtsp://username:password@ip:554/stream"
```

Notes:

- Press `q` to stop the live window.
- Optional arguments: `--model`, `--conf`, `--imgsz`

## M2 Software-Only Tracking Validation

Use this to validate M2 on your side before real PCA9685 control:

`detection -> Kalman prediction -> simulated pan/tilt angle output`

Run with camera/RTSP/video source:

```bash
python3 diagnostics/m2_servo_tracking_preview.py \
  --source "rtsp://username:password@ip:554/stream" \
  --model models/best_v2_com.pt \
  --config config/m2_tracking_config.json \
  --save-video result/m2_tracking_preview.mp4
```

Quick default save (auto path):

```bash
python3 diagnostics/m2_servo_tracking_preview.py \
  --source 0 \
  --model models/best_v2_com.pt \
  --save-video
```

Config fields exposed in [`config/m2_tracking_config.json`](config/m2_tracking_config.json):

- servo limits: `pan_min_deg`, `pan_max_deg`, `tilt_min_deg`, `tilt_max_deg`
- smoothing: `smoothing_alpha`
- prediction horizon: `prediction_horizon_sec`
- Kalman params: `process_var`, `measurement_var`, `initial_position_var`, `initial_velocity_var`
- hardware output (PCA9685): `servo_output.*` (`enabled`, `driver`, `i2c_*`, channels, pulse range, deadband)

Useful runtime options:

- `--backend` (`default`, `ffmpeg`, `gstreamer`)
- `--no-display` (headless run)
- `--max-frames`, `--print-every`
- `--servo-enable` / `--servo-disable` to override config at runtime

Client-side Jetson hardware run example:

```bash
python3 diagnostics/m2_servo_tracking_preview.py \
  --source 0 \
  --model models/best_v2_com.pt \
  --config config/m2_tracking_config.json \
  --backend default \
  --servo-enable
```

```bash
python3 diagnostics/m2_servo_tracking_preview.py \
  --source "rtsp://127.0.0.1:8554/mystream" \
  --model models/best_v2_com.pt \
  --config config/m2_tracking_config.json
```

Notes:

- Default is safe/off (`"servo_output.enabled": false`).
- For real hardware control via PCA9685, install dependency: `pip install smbus2`.

## Servo Sweep Diagnostic

Before running full tracking with hardware, verify servo direction/range quickly:

```bash
python3 diagnostics/servo_sweep.py \
  --config config/m2_tracking_config.json \
  --servo-enable \
  --mode both \
  --cycles 1
```

Useful options:

- `--driver mock` (safe dry-run without hardware)
- `--step-deg`, `--step-delay-sec`, `--hold-sec`
- `--mode pan|tilt|both`

## M3 Laser + Safety Validation

Run end-to-end M3 logic with interlocks and laser gate decision overlay:

```bash
python3 diagnostics/m3_laser_safety_preview.py \
  --source "rtsp://username:password@ip:554/stream" \
  --model models/best_v2_com.pt \
  --config config/m2_tracking_config.json \
  --save-video result/m3_laser_safety_preview.mp4
```

Safe local run (no hardware output):

```bash
python3 diagnostics/m3_laser_safety_preview.py \
  --source 0 \
  --config config/m2_tracking_config.json \
  --servo-disable \
  --laser-disable
```

Jetson hardware run:

```bash
python3 diagnostics/m3_laser_safety_preview.py \
  --source 0 \
  --config config/m2_tracking_config.json \
  --servo-enable \
  --laser-enable
```

M3 config sections in `config/m2_tracking_config.json`:

- `safety.*`: confidence threshold, lock frames, no-upward tilt rule, restricted zones
- `laser_output.*`: driver, GPIO pin/mode, active level, pulse/cooldown/failsafe

Notes:

- Keep `"laser_output.enabled": false` by default until final hardware checks.
- For Jetson GPIO output, install `Jetson.GPIO` on the target device and verify GPIO permissions.
