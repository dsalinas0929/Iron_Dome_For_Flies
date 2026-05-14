"""Constant-velocity 2D Kalman filter for target trajectory prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanConfig:
    process_var: float = 0.02
    measurement_var: float = 0.005
    initial_position_var: float = 1.0
    initial_velocity_var: float = 1.0


class ConstantVelocityKalman2D:
    """
    Kalman filter state:
        x = [pos_x, pos_y, vel_x, vel_y]^T
    Coordinates are normalized to [0, 1] frame space.
    """

    def __init__(self, config: KalmanConfig | None = None) -> None:
        self.config = config or KalmanConfig()
        self._x = np.zeros((4, 1), dtype=np.float64)
        self._p = np.eye(4, dtype=np.float64)
        self._initialized = False
        self._last_timestamp: float | None = None

        self._h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self._i = np.eye(4, dtype=np.float64)

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, x: float, y: float, timestamp: float) -> None:
        self._x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)
        self._p = np.diag(
            [
                self.config.initial_position_var,
                self.config.initial_position_var,
                self.config.initial_velocity_var,
                self.config.initial_velocity_var,
            ]
        )
        self._initialized = True
        self._last_timestamp = timestamp

    def predict(self, timestamp: float) -> tuple[float, float]:
        if not self._initialized:
            raise RuntimeError("Kalman filter must be initialized before predict.")

        dt = max(timestamp - (self._last_timestamp or timestamp), 1e-6)
        self._last_timestamp = timestamp

        f = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self.config.process_var
        q_matrix = q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

        self._x = f @ self._x
        self._p = f @ self._p @ f.T + q_matrix

        return float(self._x[0, 0]), float(self._x[1, 0])

    def update(self, x: float, y: float, timestamp: float) -> tuple[float, float]:
        if not self._initialized:
            self.initialize(x, y, timestamp)
            return x, y

        self.predict(timestamp)

        z = np.array([[x], [y]], dtype=np.float64)
        r = self.config.measurement_var * np.eye(2, dtype=np.float64)

        y_residual = z - self._h @ self._x
        s = self._h @ self._p @ self._h.T + r
        k = self._p @ self._h.T @ np.linalg.inv(s)

        self._x = self._x + k @ y_residual
        self._p = (self._i - k @ self._h) @ self._p

        return float(self._x[0, 0]), float(self._x[1, 0])

    def predict_ahead(self, horizon_sec: float) -> tuple[float, float]:
        if not self._initialized:
            raise RuntimeError("Kalman filter must be initialized before predict_ahead.")

        h = max(horizon_sec, 0.0)
        f = np.array(
            [
                [1.0, 0.0, h, 0.0],
                [0.0, 1.0, 0.0, h],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        predicted = f @ self._x
        return float(predicted[0, 0]), float(predicted[1, 0])

