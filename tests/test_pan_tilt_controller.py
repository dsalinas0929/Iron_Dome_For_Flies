from control.pan_tilt import PanTiltConfig, PanTiltController


def test_disabled_controller_never_emits() -> None:
    controller = PanTiltController(PanTiltConfig(enabled=False))
    cmd = controller.move_to(90.0, 30.0)
    assert not cmd.emitted


def test_mock_controller_emits_first_command() -> None:
    controller = PanTiltController(PanTiltConfig(enabled=True, driver="mock"))
    cmd = controller.move_to(90.0, 30.0)
    assert cmd.emitted


def test_deadband_suppresses_small_changes() -> None:
    controller = PanTiltController(
        PanTiltConfig(enabled=True, driver="mock", deadband_deg=1.0)
    )
    first = controller.move_to(90.0, 30.0)
    second = controller.move_to(90.4, 30.2)
    assert first.emitted
    assert not second.emitted


def test_angle_clamp_to_limits() -> None:
    controller = PanTiltController(
        PanTiltConfig(
            enabled=True,
            driver="mock",
            pan_min_deg=20.0,
            pan_max_deg=160.0,
            tilt_min_deg=10.0,
            tilt_max_deg=60.0,
        )
    )
    cmd = controller.move_to(999.0, -50.0)
    assert cmd.pan_angle_deg == 160.0
    assert cmd.tilt_angle_deg == 10.0

