"""Main loop skeleton for the Iron Dome for Flies MVP."""

from __future__ import annotations

from control.pan_tilt import PanTiltController
from laser.controller import LaserController
from safety.interlocks import SafetyInterlocks
from tracking.controller import TargetTracker
from vision.detector import FlyDetector


def run() -> None:
    """Orchestrate detection, tracking, and safe firing."""
    detector = FlyDetector()
    tracker = TargetTracker()
    pan_tilt = PanTiltController()
    interlocks = SafetyInterlocks()
    laser = LaserController()

    try:
        for detection in detector.stream():
            if detection is None:
                laser.update(False)
                continue

            pan_angle, tilt_angle = tracker.compute_angles(detection)
            pan_tilt.move_to(pan_angle, tilt_angle)

            decision = interlocks.evaluate(
                detection_confidence=detection.confidence,
                pan_angle=pan_angle,
                tilt_angle=tilt_angle,
                target_xy=(detection.centroid_x, detection.centroid_y),
                prediction_only=False,
            )
            laser.update(decision.allow_fire)
    finally:
        pan_tilt.close()
        laser.close()


if __name__ == "__main__":
    run()
