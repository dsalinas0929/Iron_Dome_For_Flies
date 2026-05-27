from safety.interlocks import RestrictedZone, SafetyConfig, SafetyInterlocks
from vision.detector import Detection


def test_can_fire_when_in_bounds_and_confident() -> None:
    interlocks = SafetyInterlocks(min_confidence=0.5)
    detection = Detection(0.9, 0.5, 0.5, 0.1, 0.1)
    assert interlocks.can_fire(detection, pan_angle=90.0, tilt_angle=40.0)


def test_blocks_when_confidence_too_low() -> None:
    interlocks = SafetyInterlocks(min_confidence=0.8)
    detection = Detection(0.6, 0.5, 0.5, 0.1, 0.1)
    assert not interlocks.can_fire(detection, pan_angle=90.0, tilt_angle=40.0)


def test_blocks_when_angles_out_of_bounds() -> None:
    interlocks = SafetyInterlocks()
    detection = Detection(0.9, 0.5, 0.5, 0.1, 0.1)
    assert not interlocks.can_fire(detection, pan_angle=179.0, tilt_angle=40.0)


def test_lock_required_frames() -> None:
    interlocks = SafetyInterlocks(
        config=SafetyConfig(
            min_confidence=0.5,
            lock_required_frames=2,
            pan_min=10,
            pan_max=170,
            tilt_min=5,
            tilt_max=70,
            no_upward_tilt_min=10,
            restricted_zones=[],
        )
    )
    first = interlocks.evaluate(
        detection_confidence=0.9,
        pan_angle=90.0,
        tilt_angle=40.0,
        target_xy=(0.5, 0.5),
        prediction_only=False,
    )
    second = interlocks.evaluate(
        detection_confidence=0.9,
        pan_angle=90.0,
        tilt_angle=40.0,
        target_xy=(0.5, 0.5),
        prediction_only=False,
    )
    assert not first.allow_fire
    assert second.allow_fire


def test_blocked_in_restricted_zone() -> None:
    interlocks = SafetyInterlocks(
        config=SafetyConfig(
            min_confidence=0.5,
            lock_required_frames=1,
            pan_min=10,
            pan_max=170,
            tilt_min=5,
            tilt_max=70,
            no_upward_tilt_min=10,
            restricted_zones=[
                RestrictedZone(
                    x_min=0.0,
                    y_min=0.0,
                    x_max=0.3,
                    y_max=0.3,
                    label="top_left_no_fire",
                )
            ],
        )
    )
    decision = interlocks.evaluate(
        detection_confidence=0.9,
        pan_angle=90.0,
        tilt_angle=40.0,
        target_xy=(0.2, 0.2),
        prediction_only=False,
    )
    assert not decision.allow_fire
    assert "top_left_no_fire" in decision.reasons
