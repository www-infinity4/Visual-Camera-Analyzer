"""
Tests for analyzer/detector.py

Covers:
- UVFluorescenceDetector.preprocess()
- UVFluorescenceDetector.create_fluorescence_mask()
- UVFluorescenceDetector.find_detections()
- UVFluorescenceDetector.detect()
- UVFluorescenceDetector.annotate_frame()
- Detection dataclass properties
"""

import numpy as np
import pytest

from analyzer.detector import Detection, UVFluorescenceDetector


def _make_black_frame(height: int = 100, width: int = 100) -> np.ndarray:
    """Return a completely black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _make_yellow_green_frame(height: int = 100, width: int = 100) -> np.ndarray:
    """
    Return a frame with a yellow-green rectangle in the centre.
    HSV yellow-green: H≈60°→OpenCV H=30, S=200, V=200 → BGR≈(0, 200, 200)
    We use a bright yellow-green that will fall in the DEFAULT_HSV_LOWER/UPPER range.
    """
    frame = _make_black_frame(height, width)
    # BGR for a vivid yellow-green (H≈60 in cv2 0-179 scale = 30, S=200, V=220)
    # Convert back: HSV(30,200,220) → approximately BGR(0, 220, 220) is greenish-yellow
    # Let's use BGR(0, 220, 200) which in HSV is H≈54°→cv2 H=27, well inside [20,80]
    frame[30:70, 30:70] = (0, 220, 200)
    return frame


class TestDetection:
    def test_centroid_properties(self):
        det = Detection(
            contour=np.array([[[10, 20]]]),
            bounding_box=(5, 10, 20, 30),
            area=600.0,
            centroid=(15, 25),
        )
        assert det.center_x == 15
        assert det.center_y == 25

    def test_default_label_and_confidence(self):
        det = Detection(
            contour=np.array([[[0, 0]]]),
            bounding_box=(0, 0, 10, 10),
            area=100.0,
            centroid=(5, 5),
        )
        assert det.label == "unknown"
        assert det.confidence == 0.0


class TestUVFluorescenceDetector:
    def setup_method(self):
        self.detector = UVFluorescenceDetector()

    # ------------------------------------------------------------------
    # preprocess
    # ------------------------------------------------------------------

    def test_preprocess_returns_hsv(self):
        frame = _make_yellow_green_frame()
        hsv = self.detector.preprocess(frame)
        assert hsv.shape == frame.shape
        assert hsv.dtype == np.uint8

    def test_preprocess_raises_on_empty_frame(self):
        with pytest.raises(ValueError, match="empty or None"):
            self.detector.preprocess(np.array([]))

    def test_preprocess_raises_on_none(self):
        with pytest.raises((ValueError, AttributeError)):
            self.detector.preprocess(None)

    # ------------------------------------------------------------------
    # create_fluorescence_mask
    # ------------------------------------------------------------------

    def test_mask_detects_yellow_green(self):
        frame = _make_yellow_green_frame()
        hsv = self.detector.preprocess(frame)
        mask = self.detector.create_fluorescence_mask(hsv)
        # At least some pixels in the yellow-green rectangle must be white
        assert mask.max() == 255

    def test_mask_ignores_black_background(self):
        frame = _make_black_frame()
        hsv = self.detector.preprocess(frame)
        mask = self.detector.create_fluorescence_mask(hsv)
        # No fluorescent pixels in an all-black frame
        assert mask.max() == 0

    # ------------------------------------------------------------------
    # find_detections
    # ------------------------------------------------------------------

    def test_find_detections_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        detections = self.detector.find_detections(mask)
        assert detections == []

    def test_find_detections_single_blob(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255  # 40×40 = 1600-pixel blob
        detections = self.detector.find_detections(mask)
        assert len(detections) == 1
        det = detections[0]
        assert det.area > 0
        assert det.bounding_box[2] > 0  # width > 0
        assert det.bounding_box[3] > 0  # height > 0

    def test_find_detections_filters_small_blobs(self):
        """Blobs smaller than min_contour_area should be filtered out."""
        detector = UVFluorescenceDetector(min_contour_area=500.0)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[48:52, 48:52] = 255  # 4×4 = 16 pixels – below threshold
        detections = detector.find_detections(mask)
        assert detections == []

    def test_find_detections_multiple_blobs(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:40, 10:40] = 255  # blob 1: 30×30 = 900 px
        mask[100:140, 100:140] = 255  # blob 2: 40×40 = 1600 px
        detections = self.detector.find_detections(mask)
        assert len(detections) == 2

    # ------------------------------------------------------------------
    # detect (end-to-end)
    # ------------------------------------------------------------------

    def test_detect_returns_correct_types(self):
        frame = _make_yellow_green_frame()
        detections, mask = self.detector.detect(frame)
        assert isinstance(detections, list)
        assert isinstance(mask, np.ndarray)

    def test_detect_finds_urine_spot(self):
        frame = _make_yellow_green_frame()
        detections, _ = self.detector.detect(frame)
        assert len(detections) >= 1

    def test_detect_no_detections_on_black(self):
        frame = _make_black_frame()
        detections, _ = self.detector.detect(frame)
        assert detections == []

    # ------------------------------------------------------------------
    # annotate_frame
    # ------------------------------------------------------------------

    def test_annotate_frame_does_not_modify_original(self):
        frame = _make_yellow_green_frame()
        original = frame.copy()
        detections, _ = self.detector.detect(frame)
        _ = self.detector.annotate_frame(frame, detections)
        np.testing.assert_array_equal(frame, original)

    def test_annotate_frame_same_shape(self):
        frame = _make_yellow_green_frame()
        detections, _ = self.detector.detect(frame)
        annotated = self.detector.annotate_frame(frame, detections)
        assert annotated.shape == frame.shape
