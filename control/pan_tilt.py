"""Pan/tilt command layer with optional PCA9685 hardware output."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class PanTiltConfig:
    enabled: bool = False
    driver: str = "mock"  # "mock" or "pca9685"
    i2c_bus: int = 1
    i2c_address: int = 0x40
    pwm_frequency_hz: int = 50
    pan_channel: int = 0
    tilt_channel: int = 1
    pan_min_deg: float = 10.0
    pan_max_deg: float = 170.0
    tilt_min_deg: float = 5.0
    tilt_max_deg: float = 70.0
    pan_invert: bool = False
    tilt_invert: bool = False
    pan_offset_deg: float = 0.0
    tilt_offset_deg: float = 0.0
    servo_min_pulse_us: float = 500.0
    servo_max_pulse_us: float = 2500.0
    deadband_deg: float = 0.2
    startup_center: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "PanTiltConfig":
        payload = payload or {}
        return cls(
            enabled=bool(payload.get("enabled", cls.enabled)),
            driver=str(payload.get("driver", cls.driver)),
            i2c_bus=int(payload.get("i2c_bus", cls.i2c_bus)),
            i2c_address=int(payload.get("i2c_address", cls.i2c_address)),
            pwm_frequency_hz=int(payload.get("pwm_frequency_hz", cls.pwm_frequency_hz)),
            pan_channel=int(payload.get("pan_channel", cls.pan_channel)),
            tilt_channel=int(payload.get("tilt_channel", cls.tilt_channel)),
            pan_min_deg=float(payload.get("pan_min_deg", cls.pan_min_deg)),
            pan_max_deg=float(payload.get("pan_max_deg", cls.pan_max_deg)),
            tilt_min_deg=float(payload.get("tilt_min_deg", cls.tilt_min_deg)),
            tilt_max_deg=float(payload.get("tilt_max_deg", cls.tilt_max_deg)),
            pan_invert=bool(payload.get("pan_invert", cls.pan_invert)),
            tilt_invert=bool(payload.get("tilt_invert", cls.tilt_invert)),
            pan_offset_deg=float(payload.get("pan_offset_deg", cls.pan_offset_deg)),
            tilt_offset_deg=float(payload.get("tilt_offset_deg", cls.tilt_offset_deg)),
            servo_min_pulse_us=float(payload.get("servo_min_pulse_us", cls.servo_min_pulse_us)),
            servo_max_pulse_us=float(payload.get("servo_max_pulse_us", cls.servo_max_pulse_us)),
            deadband_deg=float(payload.get("deadband_deg", cls.deadband_deg)),
            startup_center=bool(payload.get("startup_center", cls.startup_center)),
        )


@dataclass
class ServoCommand:
    pan_angle_deg: float
    tilt_angle_deg: float
    pan_ticks: int
    tilt_ticks: int
    emitted: bool


class _PwmDriver(Protocol):
    def set_channel_ticks(self, channel: int, off_ticks: int) -> None: ...

    def close(self) -> None: ...


class _MockPwmDriver:
    def set_channel_ticks(self, channel: int, off_ticks: int) -> None:
        _ = channel, off_ticks

    def close(self) -> None:
        return None


class _PCA9685Driver:
    MODE1 = 0x00
    MODE2 = 0x01
    LED0_ON_L = 0x06
    PRE_SCALE = 0xFE
    MODE1_SLEEP = 0x10
    MODE1_AI = 0x20
    MODE1_RESTART = 0x80
    MODE2_OUTDRV = 0x04
    OSC_CLOCK_HZ = 25_000_000

    def __init__(self, i2c_bus: int, i2c_address: int, pwm_frequency_hz: int) -> None:
        try:
            from smbus2 import SMBus
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "smbus2 is required for PCA9685 control. Install with: pip install smbus2"
            ) from exc

        self._bus = SMBus(i2c_bus)
        self._address = i2c_address
        self._configure(pwm_frequency_hz)

    def _read8(self, reg: int) -> int:
        return int(self._bus.read_byte_data(self._address, reg))

    def _write8(self, reg: int, value: int) -> None:
        self._bus.write_byte_data(self._address, reg, value & 0xFF)

    def _configure(self, pwm_frequency_hz: int) -> None:
        self._write8(self.MODE1, self.MODE1_AI)
        self._write8(self.MODE2, self.MODE2_OUTDRV)
        self._set_pwm_frequency(pwm_frequency_hz)

    def _set_pwm_frequency(self, frequency_hz: int) -> None:
        freq = int(_clamp(float(frequency_hz), 24.0, 1526.0))
        prescale_val = int(round(self.OSC_CLOCK_HZ / (4096.0 * freq) - 1.0))
        prescale_val = int(_clamp(float(prescale_val), 3.0, 255.0))

        old_mode = self._read8(self.MODE1)
        sleep_mode = (old_mode & 0x7F) | self.MODE1_SLEEP
        self._write8(self.MODE1, sleep_mode)
        self._write8(self.PRE_SCALE, prescale_val)
        self._write8(self.MODE1, old_mode)
        time.sleep(0.005)
        self._write8(self.MODE1, old_mode | self.MODE1_AI | self.MODE1_RESTART)

    def set_channel_ticks(self, channel: int, off_ticks: int) -> None:
        channel = int(_clamp(float(channel), 0.0, 15.0))
        off_ticks = int(_clamp(float(off_ticks), 0.0, 4095.0))
        reg = self.LED0_ON_L + 4 * channel

        self._write8(reg + 0, 0)  # ON_L
        self._write8(reg + 1, 0)  # ON_H
        self._write8(reg + 2, off_ticks & 0xFF)  # OFF_L
        self._write8(reg + 3, (off_ticks >> 8) & 0x0F)  # OFF_H (12-bit)

    def close(self) -> None:
        self._bus.close()


class PanTiltController:
    """Map pan/tilt angles to PWM ticks and optionally write to PCA9685."""

    def __init__(self, config: PanTiltConfig | None = None) -> None:
        self.config = config or PanTiltConfig()
        self._last_pan: float | None = None
        self._last_tilt: float | None = None
        self.last_command: ServoCommand | None = None

        if not self.config.enabled:
            self._driver: _PwmDriver = _MockPwmDriver()
            return

        if self.config.driver == "mock":
            self._driver = _MockPwmDriver()
        elif self.config.driver == "pca9685":
            self._driver = _PCA9685Driver(
                i2c_bus=self.config.i2c_bus,
                i2c_address=self.config.i2c_address,
                pwm_frequency_hz=self.config.pwm_frequency_hz,
            )
        else:
            raise ValueError(f"Unsupported pan/tilt driver: {self.config.driver}")

        if self.config.startup_center:
            self.move_to(
                (self.config.pan_min_deg + self.config.pan_max_deg) * 0.5,
                (self.config.tilt_min_deg + self.config.tilt_max_deg) * 0.5,
            )

    def move_to(self, pan_angle: float, tilt_angle: float) -> ServoCommand:
        pan = self._normalize_pan(float(pan_angle))
        tilt = self._normalize_tilt(float(tilt_angle))
        pan_ticks = self._angle_to_ticks(
            pan,
            self.config.pan_min_deg,
            self.config.pan_max_deg,
        )
        tilt_ticks = self._angle_to_ticks(
            tilt,
            self.config.tilt_min_deg,
            self.config.tilt_max_deg,
        )

        emitted = False
        if self.config.enabled:
            emitted = self._should_emit(pan, tilt)
        if emitted:
            self._driver.set_channel_ticks(self.config.pan_channel, pan_ticks)
            self._driver.set_channel_ticks(self.config.tilt_channel, tilt_ticks)
            self._last_pan = pan
            self._last_tilt = tilt

        command = ServoCommand(
            pan_angle_deg=pan,
            tilt_angle_deg=tilt,
            pan_ticks=pan_ticks,
            tilt_ticks=tilt_ticks,
            emitted=emitted,
        )
        self.last_command = command
        return command

    def close(self) -> None:
        self._driver.close()

    def _should_emit(self, pan: float, tilt: float) -> bool:
        if self._last_pan is None or self._last_tilt is None:
            return True
        return (
            abs(pan - self._last_pan) >= self.config.deadband_deg
            or abs(tilt - self._last_tilt) >= self.config.deadband_deg
        )

    def _normalize_pan(self, angle: float) -> float:
        adjusted = (
            self.config.pan_min_deg + self.config.pan_max_deg - angle
            if self.config.pan_invert
            else angle
        )
        adjusted += self.config.pan_offset_deg
        return _clamp(adjusted, self.config.pan_min_deg, self.config.pan_max_deg)

    def _normalize_tilt(self, angle: float) -> float:
        adjusted = (
            self.config.tilt_min_deg + self.config.tilt_max_deg - angle
            if self.config.tilt_invert
            else angle
        )
        adjusted += self.config.tilt_offset_deg
        return _clamp(adjusted, self.config.tilt_min_deg, self.config.tilt_max_deg)

    def _angle_to_ticks(self, angle: float, angle_min: float, angle_max: float) -> int:
        span_deg = max(angle_max - angle_min, 1e-6)
        normalized = _clamp((angle - angle_min) / span_deg, 0.0, 1.0)
        pulse_us = self.config.servo_min_pulse_us + normalized * (
            self.config.servo_max_pulse_us - self.config.servo_min_pulse_us
        )
        period_us = 1_000_000.0 / max(float(self.config.pwm_frequency_hz), 1.0)
        ticks = int(round((pulse_us / period_us) * 4096.0))
        return int(_clamp(float(ticks), 0.0, 4095.0))
