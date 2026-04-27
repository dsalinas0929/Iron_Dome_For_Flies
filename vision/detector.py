"""Detection interface for live fly inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Detection:
    confidence: float
    centroid_x: float
    centroid_y: float
    bbox_width: float
    bbox_height: float


class FlyDetector:
    """Thin placeholder around YOLO/OpenCV camera inference."""

    def stream(self) -> Iterator[Detection | None]:
        """
        Yield latest target detection for each frame.

        Replace this stub with camera capture + model inference and target
        selection logic.
        """
        while False:
            yield None

