"""PCA9685-backed pan/tilt control placeholders."""

from __future__ import annotations


class PanTiltController:
    """Send movement commands to pan/tilt servos."""

    def move_to(self, pan_angle: float, tilt_angle: float) -> None:
        # TODO: map angles to PWM values and send over I2C/PCA9685.
        _ = pan_angle, tilt_angle

