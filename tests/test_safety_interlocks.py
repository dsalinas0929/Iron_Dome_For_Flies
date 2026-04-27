from safety.interlocks import SafetyInterlocks
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

