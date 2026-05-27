"""Safety interlock logic for laser firing decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from vision.detector import Detection


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class RestrictedZone:
    """Normalized screen-space blocked zone [0..1] coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str = "restricted_zone"

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RestrictedZone":
        return cls(
            x_min=float(payload.get("x_min", 0.0)),
            y_min=float(payload.get("y_min", 0.0)),
            x_max=float(payload.get("x_max", 1.0)),
            y_max=float(payload.get("y_max", 1.0)),
            label=str(payload.get("label", "restricted_zone")),
        )


@dataclass
class SafetyConfig:
    min_confidence: float = 0.6
    lock_required_frames: int = 3
    max_prediction_only_frames: int = 0
    pan_min: float = 10.0
    pan_max: float = 170.0
    tilt_min: float = 5.0
    tilt_max: float = 70.0
    no_upward_tilt_min: float = 5.0
    restricted_zones: list[RestrictedZone] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "SafetyConfig":
        payload = payload or {}
        zones_payload = payload.get("restricted_zones", [])
        zones: list[RestrictedZone] = []
        if isinstance(zones_payload, list):
            for item in zones_payload:
                if isinstance(item, dict):
                    zones.append(RestrictedZone.from_dict(item))

        tilt_min = float(payload.get("tilt_min", cls.tilt_min))
        return cls(
            min_confidence=float(payload.get("min_confidence", cls.min_confidence)),
            lock_required_frames=int(payload.get("lock_required_frames", cls.lock_required_frames)),
            max_prediction_only_frames=int(
                payload.get("max_prediction_only_frames", cls.max_prediction_only_frames)
            ),
            pan_min=float(payload.get("pan_min", cls.pan_min)),
            pan_max=float(payload.get("pan_max", cls.pan_max)),
            tilt_min=tilt_min,
            tilt_max=float(payload.get("tilt_max", cls.tilt_max)),
            no_upward_tilt_min=float(payload.get("no_upward_tilt_min", tilt_min)),
            restricted_zones=zones,
        )


@dataclass
class SafetyDecision:
    allow_fire: bool
    reasons: list[str]
    lock_frames: int
    prediction_only_frames: int


class SafetyInterlocks:
    """Apply confidence, lock, angle, and restricted-zone constraints.
    
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        pan_min: float = 10.0,
        pan_max: float = 170.0,
        tilt_min: float = 5.0,
        tilt_max: float = 70.0,
        lock_required_frames: int = 1,
        max_prediction_only_frames: int = 0,
        no_upward_tilt_min: float | None = None,
        restricted_zones: list[RestrictedZone] | None = None,
        config: SafetyConfig | None = None,
    ) -> None:
        if config is None:
            config = SafetyConfig(
                min_confidence=min_confidence,
                lock_required_frames=lock_required_frames,
                max_prediction_only_frames=max_prediction_only_frames,
                pan_min=pan_min,
                pan_max=pan_max,
                tilt_min=tilt_min,
                tilt_max=tilt_max,
                no_upward_tilt_min=tilt_min
                if no_upward_tilt_min is None
                else float(no_upward_tilt_min),
                restricted_zones=restricted_zones or [],
            )

        self.config = config
        self._lock_frames = 0
        self._prediction_only_frames = 0
        self.last_decision = SafetyDecision(
            allow_fire=False,
            reasons=["not_evaluated"],
            lock_frames=0,
            prediction_only_frames=0,
        )

    @property
    def lock_frames(self) -> int:
        return self._lock_frames

    def evaluate(
        self,
        detection_confidence: float | None,
        pan_angle: float,
        tilt_angle: float,
        target_xy: tuple[float, float] | None,
        prediction_only: bool = False,
    ) -> SafetyDecision:
        reasons: list[str] = []

        confidence_ok = (
            detection_confidence is not None
            and float(detection_confidence) >= self.config.min_confidence
        )

        if prediction_only:
            self._prediction_only_frames += 1
        else:
            self._prediction_only_frames = 0

        if confidence_ok and not prediction_only:
            self._lock_frames += 1
        elif prediction_only and self._prediction_only_frames <= self.config.max_prediction_only_frames:
            # Keep lock counter for short inference gaps.
            self._lock_frames = max(self._lock_frames, 0)
        else:
            self._lock_frames = 0

        if not confidence_ok:
            reasons.append("low_confidence")

        if self._lock_frames < max(self.config.lock_required_frames, 1):
            reasons.append("target_not_locked")

        if not (self.config.pan_min <= pan_angle <= self.config.pan_max):
            reasons.append("pan_out_of_range")

        if not (self.config.tilt_min <= tilt_angle <= self.config.tilt_max):
            reasons.append("tilt_out_of_range")

        if tilt_angle < self.config.no_upward_tilt_min:
            reasons.append("upward_fire_blocked")

        if prediction_only and self._prediction_only_frames > self.config.max_prediction_only_frames:
            reasons.append("prediction_only_timeout")

        if target_xy is None:
            reasons.append("missing_target_xy")
        else:
            x = _clamp(float(target_xy[0]), 0.0, 1.0)
            y = _clamp(float(target_xy[1]), 0.0, 1.0)
            for zone in self.config.restricted_zones:
                if zone.contains(x, y):
                    reasons.append(zone.label)
                    break

        decision = SafetyDecision(
            allow_fire=(len(reasons) == 0),
            reasons=reasons if reasons else ["ok_to_fire"],
            lock_frames=self._lock_frames,
            prediction_only_frames=self._prediction_only_frames,
        )
        self.last_decision = decision
        return decision

    def can_fire(self, detection: Detection, pan_angle: float, tilt_angle: float) -> bool:
        """
        Backward-compatible helper for existing call sites.

        For advanced diagnostics, use `evaluate(...)` to read detailed reasons.
        """
        decision = self.evaluate(
            detection_confidence=detection.confidence,
            pan_angle=pan_angle,
            tilt_angle=tilt_angle,
            target_xy=(detection.centroid_x, detection.centroid_y),
            prediction_only=False,
        )
        return decision.allow_fire
