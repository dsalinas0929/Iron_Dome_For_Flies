"""Target tracking, prediction, and servo-angle mapping."""

from __future__ import annotations

from dataclasses import dataclass
import time

from tracking.kalman_filter import ConstantVelocityKalman2D, KalmanConfig
from vision.detector import Detection


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class TrackerConfig:
    pan_min_deg: float = 10.0
    pan_max_deg: float = 170.0
    tilt_min_deg: float = 5.0
    tilt_max_deg: float = 70.0
    smoothing_alpha: float = 0.25
    prediction_horizon_sec: float = 0.12
    invert_pan: bool = False
    invert_tilt: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "TrackerConfig":
        payload = payload or {}
        return cls(
            pan_min_deg=float(payload.get("pan_min_deg", cls.pan_min_deg)),
            pan_max_deg=float(payload.get("pan_max_deg", cls.pan_max_deg)),
            tilt_min_deg=float(payload.get("tilt_min_deg", cls.tilt_min_deg)),
            tilt_max_deg=float(payload.get("tilt_max_deg", cls.tilt_max_deg)),
            smoothing_alpha=float(payload.get("smoothing_alpha", cls.smoothing_alpha)),
            prediction_horizon_sec=float(
                payload.get("prediction_horizon_sec", cls.prediction_horizon_sec)
            ),
            invert_pan=bool(payload.get("invert_pan", cls.invert_pan)),
            invert_tilt=bool(payload.get("invert_tilt", cls.invert_tilt)),
        )


@dataclass
class TrackerOutput:
    pan_angle_deg: float
    tilt_angle_deg: float
    predicted_x: float
    predicted_y: float
    measured_x: float | None
    measured_y: float | None
    detection_confidence: float | None
    prediction_only: bool


class TargetTracker:
    """Convert normalized detections into smooth, predicted servo commands."""

    def __init__(
        self,
        tracker_config: TrackerConfig | None = None,
        kalman_config: KalmanConfig | None = None,
    ) -> None:
        self.config = tracker_config or TrackerConfig()
        self.kalman = ConstantVelocityKalman2D(kalman_config)
        self._last_pan: float | None = None
        self._last_tilt: float | None = None

    def compute_angles(self, detection: Detection) -> tuple[float, float]:
        """
        Backward-compatible helper used by existing app skeleton.

        Uses the same prediction/smoothing path as `update`.
        """
        state = self.update(
            measurement=(detection.centroid_x, detection.centroid_y),
            detection_confidence=detection.confidence,
        )
        return state.pan_angle_deg, state.tilt_angle_deg

    def update(
        self,
        measurement: tuple[float, float] | None,
        detection_confidence: float | None = None,
        timestamp: float | None = None,
    ) -> TrackerOutput:
        now = timestamp if timestamp is not None else time.monotonic()

        measured_x: float | None = None
        measured_y: float | None = None
        prediction_only = measurement is None

        if measurement is not None:
            measured_x = _clamp(float(measurement[0]), 0.0, 1.0)
            measured_y = _clamp(float(measurement[1]), 0.0, 1.0)
            self.kalman.update(measured_x, measured_y, now)
        elif self.kalman.initialized:
            self.kalman.predict(now)

        if self.kalman.initialized:
            predicted_x, predicted_y = self.kalman.predict_ahead(self.config.prediction_horizon_sec)
        else:
            predicted_x = 0.5
            predicted_y = 0.5

        predicted_x = _clamp(predicted_x, 0.0, 1.0)
        predicted_y = _clamp(predicted_y, 0.0, 1.0)

        raw_pan, raw_tilt = self._normalized_to_angles(predicted_x, predicted_y)
        pan_angle, tilt_angle = self._smooth_angles(raw_pan, raw_tilt)

        return TrackerOutput(
            pan_angle_deg=pan_angle,
            tilt_angle_deg=tilt_angle,
            predicted_x=predicted_x,
            predicted_y=predicted_y,
            measured_x=measured_x,
            measured_y=measured_y,
            detection_confidence=detection_confidence,
            prediction_only=prediction_only,
        )

    def _normalized_to_angles(self, x: float, y: float) -> tuple[float, float]:
        x = 1.0 - x if self.config.invert_pan else x
        y = 1.0 - y if self.config.invert_tilt else y

        pan = self.config.pan_min_deg + x * (self.config.pan_max_deg - self.config.pan_min_deg)
        tilt = self.config.tilt_min_deg + y * (self.config.tilt_max_deg - self.config.tilt_min_deg)

        pan = _clamp(pan, self.config.pan_min_deg, self.config.pan_max_deg)
        tilt = _clamp(tilt, self.config.tilt_min_deg, self.config.tilt_max_deg)
        return pan, tilt

    def _smooth_angles(self, raw_pan: float, raw_tilt: float) -> tuple[float, float]:
        alpha = _clamp(self.config.smoothing_alpha, 0.0, 1.0)

        if self._last_pan is None or self._last_tilt is None:
            smoothed_pan = raw_pan
            smoothed_tilt = raw_tilt
        else:
            smoothed_pan = self._last_pan + alpha * (raw_pan - self._last_pan)
            smoothed_tilt = self._last_tilt + alpha * (raw_tilt - self._last_tilt)

        smoothed_pan = _clamp(smoothed_pan, self.config.pan_min_deg, self.config.pan_max_deg)
        smoothed_tilt = _clamp(smoothed_tilt, self.config.tilt_min_deg, self.config.tilt_max_deg)

        self._last_pan = smoothed_pan
        self._last_tilt = smoothed_tilt
        return smoothed_pan, smoothed_tilt
