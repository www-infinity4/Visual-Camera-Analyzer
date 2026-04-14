"""
Detection Logger

Persists detection events to a structured JSON log file and saves
annotated images of each detection for later review.

Each log entry records:
- ISO-8601 timestamp
- Frame number
- Number of detections
- Per-detection metadata (bounding box, centroid, area, label, confidence)
- Path to the saved annotated image (if image saving is enabled)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from analyzer.detector import Detection


class DetectionLogger:
    """
    Records detection events to JSON and optionally saves annotated images.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        image_dir: str = "logs/images",
        log_filename: Optional[str] = None,
        save_images: bool = True,
        max_entries: int = 10000,
    ):
        """
        Args:
            log_dir: Directory for JSON log files.
            image_dir: Directory for annotated detection images.
            log_filename: Log file name.  Defaults to a timestamped name.
            save_images: Whether to save annotated frames for each detection.
            max_entries: Maximum entries before the log is rotated.
        """
        self.log_dir = log_dir
        self.image_dir = image_dir
        self.save_images = save_images
        self.max_entries = max_entries
        self._entries: List[Dict[str, Any]] = []
        self._frame_count: int = 0

        os.makedirs(log_dir, exist_ok=True)
        if save_images:
            os.makedirs(image_dir, exist_ok=True)

        if log_filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_filename = f"detections_{timestamp}.json"

        self.log_path = os.path.join(log_dir, log_filename)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Record a detection event.

        Args:
            detections: Detections from the current frame.
            frame: Optionally save this annotated frame image.

        Returns:
            The log entry dict that was recorded.
        """
        self._frame_count += 1
        timestamp = datetime.now(timezone.utc).isoformat()

        image_path: Optional[str] = None
        if self.save_images and frame is not None and len(detections) > 0:
            image_path = self._save_image(frame, self._frame_count)

        entry: Dict[str, Any] = {
            "timestamp": timestamp,
            "frame": self._frame_count,
            "detection_count": len(detections),
            "detections": [self._serialise_detection(d) for d in detections],
            "image_path": image_path,
        }

        self._entries.append(entry)
        self._flush()

        if len(self._entries) >= self.max_entries:
            self._rotate()

        return entry

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Return all in-memory log entries."""
        return list(self._entries)

    def total_detections(self) -> int:
        """Return total number of individual detections across all entries."""
        return sum(e["detection_count"] for e in self._entries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _serialise_detection(self, det: Detection) -> Dict[str, Any]:
        x, y, w, h = det.bounding_box
        return {
            "label": det.label,
            "confidence": round(det.confidence, 4),
            "centroid": {"x": det.center_x, "y": det.center_y},
            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
            "area": round(det.area, 2),
        }

    def _save_image(self, frame: np.ndarray, frame_count: int) -> str:
        filename = f"frame_{frame_count:06d}.jpg"
        path = os.path.join(self.image_dir, filename)
        cv2.imwrite(path, frame)
        return path

    def _flush(self) -> None:
        """Write all entries to the JSON log file."""
        with open(self.log_path, "w", encoding="utf-8") as fh:
            json.dump(self._entries, fh, indent=2, ensure_ascii=False)

    def _rotate(self) -> None:
        """Rotate the log file when it exceeds max_entries."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rotated_path = self.log_path.replace(".json", f"_{timestamp}_rotated.json")
        os.rename(self.log_path, rotated_path)
        self._entries = []
