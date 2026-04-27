"""Target tracking and angle generation."""

from __future__ import annotations

from vision.detector import Detection


class TargetTracker:
    """Convert target centroid into pan/tilt angles."""

    def compute_angles(self, detection: Detection) -> tuple[float, float]:
        """
        Convert normalized centroid values to servo angles.

        This placeholder assumes detection centroid is normalized to [0, 1].
        """
        pan_angle = detection.centroid_x * 180.0
        tilt_angle = detection.centroid_y * 90.0
        return pan_angle, tilt_angle

