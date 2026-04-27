"""Safety interlock checks before laser firing."""

from __future__ import annotations

from vision.detector import Detection


class SafetyInterlocks:
    """Apply basic confidence/zone/angle firing constraints."""

    def __init__(
        self,
        min_confidence: float = 0.6,
        pan_min: float = 10.0,
        pan_max: float = 170.0,
        tilt_min: float = 5.0,
        tilt_max: float = 70.0,
    ) -> None:
        self.min_confidence = min_confidence
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max

    def can_fire(self, detection: Detection, pan_angle: float, tilt_angle: float) -> bool:
        """Return True only if confidence and angle constraints pass."""
        if detection.confidence < self.min_confidence:
            return False

        if not (self.pan_min <= pan_angle <= self.pan_max):
            return False

        if not (self.tilt_min <= tilt_angle <= self.tilt_max):
            return False

        return True

