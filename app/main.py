"""Main loop skeleton for the Iron Dome for Flies MVP."""

from __future__ import annotations

from control.pan_tilt import PanTiltController
from laser.controller import LaserController
from safety.interlocks import SafetyInterlocks
from tracking.controller import TargetTracker
from vision.detector import Detection, FlyDetector


def run() -> None:
    """Orchestrate detection, tracking, and safe firing."""
    detector = FlyDetector()
    tracker = TargetTracker()
    pan_tilt = PanTiltController()
    interlocks = SafetyInterlocks()
    laser = LaserController()

    for detection in detector.stream():
        if detection is None:
            laser.disable()
            continue

        pan_angle, tilt_angle = tracker.compute_angles(detection)
        pan_tilt.move_to(pan_angle, tilt_angle)

        if interlocks.can_fire(detection, pan_angle, tilt_angle):
            laser.enable()
        else:
            laser.disable()


if __name__ == "__main__":
    run()

