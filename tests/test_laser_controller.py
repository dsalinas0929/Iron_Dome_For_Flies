import time

from laser.controller import LaserConfig, LaserController


def test_disabled_laser_stays_off() -> None:
    controller = LaserController(LaserConfig(enabled=False))
    status = controller.update(True)
    assert not status.is_on
    assert status.reason == "disabled_in_config"
    controller.close()


def test_continuous_mode_on_and_off() -> None:
    controller = LaserController(
        LaserConfig(
            enabled=True,
            driver="mock",
            fire_mode="continuous",
            fail_safe_max_on_ms=1000.0,
        )
    )
    on_state = controller.update(True)
    off_state = controller.update(False)
    assert on_state.is_on
    assert off_state.is_on is False
    controller.close()


def test_pulse_mode_honors_pulse_and_cooldown() -> None:
    controller = LaserController(
        LaserConfig(
            enabled=True,
            driver="mock",
            fire_mode="pulse",
            pulse_on_ms=60.0,
            cooldown_ms=120.0,
            fail_safe_max_on_ms=500.0,
        )
    )
    base = time.monotonic()

    start = controller.update(True, now=base)
    during = controller.update(True, now=base + 0.02)
    after_pulse = controller.update(True, now=base + 0.08)
    in_cooldown = controller.update(True, now=base + 0.15)
    next_pulse = controller.update(True, now=base + 0.25)

    assert start.is_on
    assert during.is_on
    assert not after_pulse.is_on
    assert in_cooldown.reason == "cooldown_active"
    assert next_pulse.is_on
    controller.close()

