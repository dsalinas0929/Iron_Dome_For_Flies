"""GPIO-backed laser controller placeholder."""

from __future__ import annotations


class LaserController:
    """Enable/disable TTL laser output."""

    def enable(self) -> None:
        # TODO: set TTL GPIO high after interlocks pass.
        return None

    def disable(self) -> None:
        # TODO: set TTL GPIO low.
        return None

