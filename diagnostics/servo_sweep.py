"""Servo sweep diagnostic for PCA9685 pan/tilt verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Iterator

# Allow running as `python diagnostics/servo_sweep.py` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.pan_tilt import PanTiltConfig, PanTiltController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep pan/tilt servos to validate direction, range, and PWM mapping."
    )
    parser.add_argument(
        "--config",
        default="config/m2_tracking_config.json",
        help="JSON config file containing `servo_output`.",
    )
    parser.add_argument(
        "--mode",
        choices=["pan", "tilt", "both"],
        default="both",
        help="Which axis pattern to run.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="How many sweep cycles to run.",
    )
    parser.add_argument(
        "--step-deg",
        type=float,
        default=2.0,
        help="Angle step size per update.",
    )
    parser.add_argument(
        "--step-delay-sec",
        type=float,
        default=0.03,
        help="Delay between servo commands.",
    )
    parser.add_argument(
        "--hold-sec",
        type=float,
        default=0.35,
        help="Hold time at key positions (center/min/max).",
    )
    parser.add_argument(
        "--deadband-deg",
        type=float,
        default=None,
        help="Optional override for deadband.",
    )
    parser.add_argument(
        "--driver",
        choices=["mock", "pca9685"],
        default=None,
        help="Optional driver override.",
    )
    toggle = parser.add_mutually_exclusive_group()
    toggle.add_argument(
        "--servo-enable",
        action="store_true",
        help="Force-enable servo output regardless of config.",
    )
    toggle.add_argument(
        "--servo-disable",
        action="store_true",
        help="Force-disable servo output regardless of config.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def frange(start: float, stop: float, step: float) -> Iterator[float]:
    if step <= 0.0:
        raise ValueError("step must be > 0")

    if start <= stop:
        value = start
        while value < stop:
            yield value
            value += step
        yield stop
        return

    value = start
    while value > stop:
        yield value
        value -= step
    yield stop


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def move_and_print(controller: PanTiltController, pan: float, tilt: float) -> None:
    cmd = controller.move_to(pan, tilt)
    print(
        f"[{ts()}] pan={cmd.pan_angle_deg:7.2f} "
        f"tilt={cmd.tilt_angle_deg:7.2f} "
        f"ticks=({cmd.pan_ticks:4d},{cmd.tilt_ticks:4d}) "
        f"emitted={cmd.emitted}"
    )


def sweep_axis(
    controller: PanTiltController,
    axis: str,
    min_deg: float,
    max_deg: float,
    fixed_other_axis: float,
    step_deg: float,
    step_delay_sec: float,
) -> None:
    if axis == "pan":
        for pan in frange(min_deg, max_deg, step_deg):
            move_and_print(controller, pan, fixed_other_axis)
            time.sleep(step_delay_sec)
        for pan in frange(max_deg, min_deg, step_deg):
            move_and_print(controller, pan, fixed_other_axis)
            time.sleep(step_delay_sec)
        return

    for tilt in frange(min_deg, max_deg, step_deg):
        move_and_print(controller, fixed_other_axis, tilt)
        time.sleep(step_delay_sec)
    for tilt in frange(max_deg, min_deg, step_deg):
        move_and_print(controller, fixed_other_axis, tilt)
        time.sleep(step_delay_sec)


def run() -> None:
    args = parse_args()
    payload = load_json(Path(args.config))
    servo_payload = dict(payload.get("servo_output", {}))

    if args.driver is not None:
        servo_payload["driver"] = args.driver
    if args.servo_enable:
        servo_payload["enabled"] = True
    if args.servo_disable:
        servo_payload["enabled"] = False
    if args.deadband_deg is not None:
        servo_payload["deadband_deg"] = args.deadband_deg

    config = PanTiltConfig.from_dict(servo_payload)
    controller = PanTiltController(config)

    pan_min = config.pan_min_deg
    pan_max = config.pan_max_deg
    tilt_min = config.tilt_min_deg
    tilt_max = config.tilt_max_deg
    pan_center = 0.5 * (pan_min + pan_max)
    tilt_center = 0.5 * (tilt_min + tilt_max)

    print(f"[{ts()}] Servo sweep starting")
    print(
        f"[{ts()}] enabled={config.enabled} driver={config.driver} "
        f"i2c_bus={config.i2c_bus} i2c_address=0x{config.i2c_address:02X} "
        f"pan_ch={config.pan_channel} tilt_ch={config.tilt_channel}"
    )
    print(
        f"[{ts()}] pan=[{pan_min:.1f},{pan_max:.1f}] tilt=[{tilt_min:.1f},{tilt_max:.1f}] "
        f"pulse_us=[{config.servo_min_pulse_us:.0f},{config.servo_max_pulse_us:.0f}] "
        f"deadband={config.deadband_deg:.2f}"
    )

    try:
        move_and_print(controller, pan_center, tilt_center)
        time.sleep(args.hold_sec)

        for cycle in range(1, max(args.cycles, 1) + 1):
            print(f"\n[{ts()}] ---- Cycle {cycle}/{max(args.cycles, 1)} ----")

            if args.mode in {"pan", "both"}:
                print(f"[{ts()}] Pan axis sweep")
                sweep_axis(
                    controller=controller,
                    axis="pan",
                    min_deg=pan_min,
                    max_deg=pan_max,
                    fixed_other_axis=tilt_center,
                    step_deg=args.step_deg,
                    step_delay_sec=args.step_delay_sec,
                )
                time.sleep(args.hold_sec)

            if args.mode in {"tilt", "both"}:
                print(f"[{ts()}] Tilt axis sweep")
                sweep_axis(
                    controller=controller,
                    axis="tilt",
                    min_deg=tilt_min,
                    max_deg=tilt_max,
                    fixed_other_axis=pan_center,
                    step_deg=args.step_deg,
                    step_delay_sec=args.step_delay_sec,
                )
                time.sleep(args.hold_sec)

            if args.mode == "both":
                print(f"[{ts()}] Diagonal sweep")
                diag_steps = max(2, int(round(max(pan_max - pan_min, tilt_max - tilt_min) / args.step_deg)))
                for i in range(diag_steps + 1):
                    r = i / diag_steps
                    pan = pan_min + r * (pan_max - pan_min)
                    tilt = tilt_min + r * (tilt_max - tilt_min)
                    move_and_print(controller, pan, tilt)
                    time.sleep(args.step_delay_sec)
                for i in range(diag_steps + 1):
                    r = i / diag_steps
                    pan = pan_max - r * (pan_max - pan_min)
                    tilt = tilt_max - r * (tilt_max - tilt_min)
                    move_and_print(controller, pan, tilt)
                    time.sleep(args.step_delay_sec)
                time.sleep(args.hold_sec)

        move_and_print(controller, pan_center, tilt_center)
        time.sleep(args.hold_sec)
        print(f"[{ts()}] Servo sweep complete")
    finally:
        controller.close()


if __name__ == "__main__":
    run()

