"""Laser TTL controller with mock and Jetson GPIO backends."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class LaserConfig:
    enabled: bool = False
    driver: str = "mock"  # "mock" or "jetson_gpio"
    gpio_pin: int = 18
    gpio_mode: str = "BCM"  # BOARD, BCM, CVM, TEGRA_SOC
    active_high: bool = True
    fire_mode: str = "continuous"  # "continuous" or "pulse"
    pulse_on_ms: float = 80.0
    cooldown_ms: float = 120.0
    fail_safe_max_on_ms: float = 300.0
    cleanup_on_close: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "LaserConfig":
        payload = payload or {}
        return cls(
            enabled=bool(payload.get("enabled", cls.enabled)),
            driver=str(payload.get("driver", cls.driver)),
            gpio_pin=int(payload.get("gpio_pin", cls.gpio_pin)),
            gpio_mode=str(payload.get("gpio_mode", cls.gpio_mode)),
            active_high=bool(payload.get("active_high", cls.active_high)),
            fire_mode=str(payload.get("fire_mode", cls.fire_mode)),
            pulse_on_ms=float(payload.get("pulse_on_ms", cls.pulse_on_ms)),
            cooldown_ms=float(payload.get("cooldown_ms", cls.cooldown_ms)),
            fail_safe_max_on_ms=float(
                payload.get("fail_safe_max_on_ms", cls.fail_safe_max_on_ms)
            ),
            cleanup_on_close=bool(payload.get("cleanup_on_close", cls.cleanup_on_close)),
        )


@dataclass
class LaserStatus:
    is_on: bool
    allowed: bool
    emitted: bool
    reason: str


class _DigitalOutDriver(Protocol):
    def write(self, value_high: bool) -> None: ...

    def close(self) -> None: ...


class _MockDigitalOutDriver:
    def __init__(self) -> None:
        self.last_value_high = False

    def write(self, value_high: bool) -> None:
        self.last_value_high = bool(value_high)

    def close(self) -> None:
        return None


class _JetsonGpioDriver:
    def __init__(self, pin: int, mode_name: str, cleanup_on_close: bool) -> None:
        try:
            import Jetson.GPIO as GPIO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Jetson.GPIO is required for real laser GPIO control. "
                "Install on Jetson and ensure user has GPIO permissions."
            ) from exc

        self._gpio = GPIO
        self._pin = int(pin)
        self._cleanup_on_close = cleanup_on_close
        self._setup_mode(mode_name)
        self._gpio.setwarnings(False)
        self._gpio.setup(self._pin, self._gpio.OUT, initial=self._gpio.LOW)

    def _setup_mode(self, mode_name: str) -> None:
        mode_key = mode_name.upper()
        if not hasattr(self._gpio, mode_key):
            raise ValueError(
                f"Invalid gpio_mode '{mode_name}'. Use one of BOARD, BCM, CVM, TEGRA_SOC."
            )
        self._gpio.setmode(getattr(self._gpio, mode_key))

    def write(self, value_high: bool) -> None:
        level = self._gpio.HIGH if value_high else self._gpio.LOW
        self._gpio.output(self._pin, level)

    def close(self) -> None:
        if self._cleanup_on_close:
            self._gpio.cleanup(self._pin)


class LaserController:
    """Gate TTL laser output with cooldown and pulse/continuous modes."""

    def __init__(self, config: LaserConfig | None = None) -> None:
        self.config = config or LaserConfig()

        if not self.config.enabled:
            self._driver: _DigitalOutDriver = _MockDigitalOutDriver()
        elif self.config.driver == "mock":
            self._driver = _MockDigitalOutDriver()
        elif self.config.driver == "jetson_gpio":
            self._driver = _JetsonGpioDriver(
                pin=self.config.gpio_pin,
                mode_name=self.config.gpio_mode,
                cleanup_on_close=self.config.cleanup_on_close,
            )
        else:
            raise ValueError(f"Unsupported laser driver: {self.config.driver}")

        self._is_on = False
        self._last_on_ts: float | None = None
        self._last_off_ts = time.monotonic() - (max(self.config.cooldown_ms, 0.0) / 1000.0)
        self._pulse_end_ts: float | None = None
        self.last_status = LaserStatus(
            is_on=False,
            allowed=False,
            emitted=False,
            reason="startup_off",
        )
        self._write_output(False)

    def update(self, allow_fire: bool, now: float | None = None) -> LaserStatus:
        """
        Update laser state from interlock decision.
        Returns current output status and reason.
        """
        timestamp = now if now is not None else time.monotonic()

        if not self.config.enabled:
            self.disable(reason="disabled_in_config", now=timestamp)
            return self.last_status

        if not allow_fire:
            self.disable(reason="interlock_blocked", now=timestamp)
            return self.last_status

        if self.config.fire_mode == "continuous":
            self.enable(reason="continuous_allow", now=timestamp)
            self._fail_safe_guard(timestamp)
            return self.last_status

        if self.config.fire_mode != "pulse":
            raise ValueError(f"Unsupported fire_mode: {self.config.fire_mode}")

        cooldown_sec = max(self.config.cooldown_ms, 0.0) / 1000.0
        pulse_on_sec = max(self.config.pulse_on_ms, 0.0) / 1000.0

        if self._is_on:
            if self._pulse_end_ts is not None and timestamp >= self._pulse_end_ts:
                self.disable(reason="pulse_elapsed", now=timestamp)
            else:
                self._fail_safe_guard(timestamp)
                if self._is_on:
                    self.last_status = LaserStatus(
                        is_on=True,
                        allowed=True,
                        emitted=False,
                        reason="pulse_active",
                    )
            return self.last_status

        if (timestamp - self._last_off_ts) < cooldown_sec:
            self.last_status = LaserStatus(
                is_on=False,
                allowed=True,
                emitted=False,
                reason="cooldown_active",
            )
            return self.last_status

        self.enable(reason="pulse_start", now=timestamp)
        self._pulse_end_ts = timestamp + pulse_on_sec
        return self.last_status

    def enable(self, reason: str = "manual_enable", now: float | None = None) -> None:
        if not self.config.enabled:
            self.last_status = LaserStatus(
                is_on=False,
                allowed=False,
                emitted=False,
                reason="disabled_in_config",
            )
            return
        changed = not self._is_on
        self._write_output(True, now=now)
        self.last_status = LaserStatus(
            is_on=True,
            allowed=True,
            emitted=changed,
            reason=reason,
        )

    def disable(self, reason: str = "manual_disable", now: float | None = None) -> None:
        changed = self._is_on
        self._write_output(False, now=now)
        self.last_status = LaserStatus(
            is_on=False,
            allowed=False,
            emitted=changed,
            reason=reason,
        )

    def close(self) -> None:
        self.disable(reason="shutdown")
        self._driver.close()

    def _write_output(self, laser_on: bool, now: float | None = None) -> None:
        physical_high = laser_on if self.config.active_high else (not laser_on)
        self._driver.write(physical_high)

        timestamp = now if now is not None else time.monotonic()
        if laser_on and not self._is_on:
            self._last_on_ts = timestamp
        if not laser_on and self._is_on:
            self._last_off_ts = timestamp
            self._pulse_end_ts = None
        self._is_on = laser_on

    def _fail_safe_guard(self, now: float) -> None:
        if not self._is_on:
            return
        if self._last_on_ts is None:
            return

        max_on_sec = max(self.config.fail_safe_max_on_ms, 0.0) / 1000.0
        if max_on_sec <= 0.0:
            return
        if (now - self._last_on_ts) >= max_on_sec:
            self.disable(reason="fail_safe_max_on_exceeded", now=now)
