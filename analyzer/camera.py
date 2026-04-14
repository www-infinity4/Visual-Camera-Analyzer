"""
Camera Capture Module

Handles real-time frame acquisition from a webcam or other camera device.
Designed to be used alongside a UV blacklight (365-395nm) for fluorescence imaging.

Usage notes:
- Operate in a darkened room to maximise fluorescence contrast.
- Position the UV flashlight 15-30 cm from the surface being inspected.
- A 365nm UV LED flashlight provides the best fluorescence excitation.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class CameraCapture:
    """
    Manages camera initialisation, frame capture, and release.

    Wraps OpenCV's VideoCapture for straightforward integration with
    the fluorescence detector and classifier.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        """
        Initialise the camera.

        Args:
            device_index: Camera device index (0 = first/default webcam).
            width: Requested frame width in pixels.
            height: Requested frame height in pixels.
            fps: Requested frames per second.
        """
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the camera and apply resolution/fps settings."""
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at device index {self.device_index}. "
                "Ensure a camera is connected and the device index is correct."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

    def is_open(self) -> bool:
        """Return True if the camera is currently open."""
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame from the camera.

        Returns:
            Tuple of (success flag, BGR frame or None on failure).
        """
        if not self.is_open():
            raise RuntimeError("Camera is not open. Call open() first.")
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @property
    def actual_width(self) -> int:
        """Actual frame width reported by the camera driver."""
        if not self.is_open():
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def actual_height(self) -> int:
        """Actual frame height reported by the camera driver."""
        if not self.is_open():
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
