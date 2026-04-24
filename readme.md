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