"""
Tests for analyzer/logger.py

Covers:
- DetectionLogger.log() – creates JSON entries
- DetectionLogger.total_detections()
- DetectionLogger.get_all_entries()
- DetectionLogger log rotation
"""

import json
import os
import tempfile

import numpy as np
import pytest

from analyzer.detector import Detection
from analyzer.logger import DetectionLogger


def _make_detection(label: str = "urine", confidence: float = 0.9) -> Detection:
    return Detection(
        contour=np.array([[[10, 20]]]),
        bounding_box=(10, 20, 30, 40),
        area=1200.0,
        centroid=(25, 40),
        label=label,
        confidence=confidence,
    )


class TestDetectionLogger:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.img_dir = os.path.join(self.tmp_dir, "images")

    def _make_logger(self, **kwargs) -> DetectionLogger:
        return DetectionLogger(
            log_dir=self.tmp_dir,
            image_dir=self.img_dir,
            log_filename="test_log.json",
            save_images=False,
            **kwargs,
        )

    def test_log_creates_entry(self):
        logger = self._make_logger()
        det = _make_detection()
        entry = logger.log([det])
        assert entry["detection_count"] == 1
        assert entry["detections"][0]["label"] == "urine"

    def test_log_persists_to_json(self):
        logger = self._make_logger()
        det = _make_detection()
        logger.log([det])
        with open(logger.log_path, "r") as fh:
            data = json.load(fh)
        assert len(data) == 1
        assert data[0]["detection_count"] == 1

    def test_total_detections_accumulates(self):
        logger = self._make_logger()
        logger.log([_make_detection()])
        logger.log([_make_detection(), _make_detection()])
        assert logger.total_detections() == 3

    def test_get_all_entries_returns_copy(self):
        logger = self._make_logger()
        logger.log([_make_detection()])
        entries = logger.get_all_entries()
        entries.clear()
        # Original should still have the entry
        assert len(logger.get_all_entries()) == 1

    def test_log_empty_detections(self):
        logger = self._make_logger()
        entry = logger.log([])
        assert entry["detection_count"] == 0
        assert entry["detections"] == []

    def test_log_serialises_bounding_box(self):
        logger = self._make_logger()
        det = _make_detection()
        entry = logger.log([det])
        bb = entry["detections"][0]["bounding_box"]
        assert bb == {"x": 10, "y": 20, "width": 30, "height": 40}

    def test_log_rotation(self):
        logger = self._make_logger(max_entries=2)
        for _ in range(3):
            logger.log([_make_detection()])
        # After rotation, in-memory entries should be reset
        assert len(logger.get_all_entries()) < 3

    def test_save_images_creates_directory(self):
        # save_images=True should create the image directory
        DetectionLogger(
            log_dir=self.tmp_dir,
            image_dir=self.img_dir,
            log_filename="img_test.json",
            save_images=True,
        )
        assert os.path.isdir(self.img_dir)

    def test_log_with_frame_saves_image(self):
        logger = DetectionLogger(
            log_dir=self.tmp_dir,
            image_dir=self.img_dir,
            log_filename="frame_test.json",
            save_images=True,
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = _make_detection()
        entry = logger.log([det], frame=frame)
        assert entry["image_path"] is not None
        assert os.path.exists(entry["image_path"])
