"""
UV Fluorescence Detector

Detects cat urine by identifying the characteristic yellow-green fluorescence
that appears when dried urine is illuminated with 365-395nm UV (blacklight).

Key principles:
- UV blacklights (365-395nm) excite phosphorous compounds in dried urine
- Urine fluoresces with a distinct yellow-green color in low-light conditions
- OpenCV HSV color space analysis isolates the fluorescent signature
- Contour detection identifies and outlines individual stain regions

Limitations:
- Only detects DRIED urine (fresh urine does not fluoresce well)
- Other materials may also fluoresce: detergents, fabric brighteners, etc.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Detection:
    """Represents a single detected fluorescent region."""

    contour: np.ndarray
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    area: float
    centroid: Tuple[int, int]
    confidence: float = 0.0
    label: str = "unknown"

    @property
    def center_x(self) -> int:
        return self.centroid[0]

    @property
    def center_y(self) -> int:
        return self.centroid[1]


class UVFluorescenceDetector:
    """
    Detects UV-excited fluorescence patterns in camera images.

    Uses HSV color space thresholding to isolate the yellow-green glow
    emitted by dried cat urine under 365-395nm UV illumination.
    """

    # HSV range for urine fluorescence (yellow-green glow under UV)
    # In OpenCV HSV: Hue 0-179, Saturation 0-255, Value 0-255
    DEFAULT_HSV_LOWER = np.array([20, 40, 80])
    DEFAULT_HSV_UPPER = np.array([80, 255, 255])

    def __init__(
        self,
        hsv_lower: Optional[np.ndarray] = None,
        hsv_upper: Optional[np.ndarray] = None,
        min_contour_area: float = 100.0,
        blur_kernel_size: int = 5,
    ):
        """
        Initialize the UV fluorescence detector.

        Args:
            hsv_lower: Lower HSV bound for fluorescence color (yellow-green).
            hsv_upper: Upper HSV bound for fluorescence color.
            min_contour_area: Minimum pixel area to register as a detection.
            blur_kernel_size: Gaussian blur kernel size for noise reduction.
        """
        self.hsv_lower = hsv_lower if hsv_lower is not None else self.DEFAULT_HSV_LOWER
        self.hsv_upper = hsv_upper if hsv_upper is not None else self.DEFAULT_HSV_UPPER
        self.min_contour_area = min_contour_area
        self.blur_kernel_size = blur_kernel_size

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare a camera frame for fluorescence detection.

        Applies Gaussian blur to reduce sensor noise, then converts to HSV.

        Args:
            frame: BGR image from camera (as returned by OpenCV).

        Returns:
            HSV image suitable for color thresholding.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or None.")

        blurred = cv2.GaussianBlur(
            frame,
            (self.blur_kernel_size, self.blur_kernel_size),
            0,
        )
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return hsv

    def create_fluorescence_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        """
        Create a binary mask isolating fluorescent yellow-green regions.

        Args:
            hsv_frame: HSV image (output of preprocess()).

        Returns:
            Binary mask where fluorescent regions are white (255).
        """
        mask = cv2.inRange(hsv_frame, self.hsv_lower, self.hsv_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def find_detections(self, mask: np.ndarray) -> List[Detection]:
        """
        Extract individual detection regions from the fluorescence mask.

        Args:
            mask: Binary mask (output of create_fluorescence_mask()).

        Returns:
            List of Detection objects for each identified region.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Compute centroid using image moments
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                cx, cy = x + w // 2, y + h // 2
            else:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

            detections.append(
                Detection(
                    contour=contour,
                    bounding_box=(x, y, w, h),
                    area=area,
                    centroid=(cx, cy),
                )
            )

        return detections

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """
        Full detection pipeline: preprocess → mask → find detections.

        Args:
            frame: BGR camera frame.

        Returns:
            Tuple of (list of detections, fluorescence mask).
        """
        hsv = self.preprocess(frame)
        mask = self.create_fluorescence_mask(hsv)
        detections = self.find_detections(mask)
        return detections, mask

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels onto the frame for visualization.

        Args:
            frame: Original BGR camera frame.
            detections: Detections to annotate.
            color: BGR color for annotations.

        Returns:
            Annotated frame copy.
        """
        annotated = frame.copy()
        for i, det in enumerate(detections):
            x, y, w, h = det.bounding_box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label_text = f"{det.label} ({det.confidence:.0%})" if det.confidence > 0 else det.label
            cv2.putText(
                annotated,
                label_text,
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            # Mark centroid
            cv2.circle(annotated, det.centroid, 4, (0, 0, 255), -1)

        # Overlay detection count
        count_text = f"Detections: {len(detections)}"
        cv2.putText(
            annotated,
            count_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return annotated
