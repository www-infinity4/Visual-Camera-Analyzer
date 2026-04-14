"""
Visual Camera Analyzer - Main Entry Point

Runs the full detection pipeline:
  1. Opens the camera (use in a darkened room with a 365-395nm UV flashlight).
  2. Detects UV-fluorescent yellow-green regions in each frame.
  3. Classifies each region with the AI model (urine vs. false positive).
  4. Logs all detection events to JSON and optionally saves images.
  5. Sends email/webhook alerts when urine is detected.
  6. Displays a live annotated feed (press 'q' to quit, 's' to save snapshot).

Usage:
    python main.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

from analyzer.alerts import AlertManager
from analyzer.camera import CameraCapture
from analyzer.classifier import UrineClassifier
from analyzer.detector import UVFluorescenceDetector
from analyzer.logger import DetectionLogger


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_components(cfg: dict):
    """Instantiate all pipeline components from configuration."""
    cam_cfg = cfg.get("camera", {})
    camera = CameraCapture(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_cfg.get("width", 1280),
        height=cam_cfg.get("height", 720),
        fps=cam_cfg.get("fps", 30),
    )

    uv_cfg = cfg.get("uv_detection", {})
    detector = UVFluorescenceDetector(
        hsv_lower=np.array(
            [
                uv_cfg.get("hue_min", 20),
                uv_cfg.get("saturation_min", 40),
                uv_cfg.get("value_min", 80),
            ]
        ),
        hsv_upper=np.array(
            [
                uv_cfg.get("hue_max", 80),
                uv_cfg.get("saturation_max", 255),
                uv_cfg.get("value_max", 255),
            ]
        ),
        min_contour_area=uv_cfg.get("min_contour_area", 100.0),
        blur_kernel_size=uv_cfg.get("blur_kernel_size", 5),
    )

    cl_cfg = cfg.get("classifier", {})
    classifier = UrineClassifier(
        model_path=cl_cfg.get("model_path"),
        input_size=cl_cfg.get("input_size", 224),
        confidence_threshold=cl_cfg.get("confidence_threshold", 0.7),
        num_classes=cl_cfg.get("num_classes", 3),
        class_names=cl_cfg.get("class_names"),
    )

    log_cfg = cfg.get("logging", {})
    logger = DetectionLogger(
        log_dir=log_cfg.get("log_dir", "logs"),
        image_dir=log_cfg.get("image_dir", "logs/images"),
        save_images=True,
        max_entries=log_cfg.get("max_log_entries", 10000),
    )

    alert_cfg = cfg.get("alerts", {})
    alert_manager = AlertManager(
        email_enabled=alert_cfg.get("email_enabled", False),
        smtp_host=alert_cfg.get("smtp_host", ""),
        smtp_port=alert_cfg.get("smtp_port", 587),
        smtp_user=alert_cfg.get("smtp_user", ""),
        smtp_password=alert_cfg.get("smtp_password", ""),
        recipient_email=alert_cfg.get("recipient_email", ""),
        webhook_enabled=alert_cfg.get("webhook_enabled", False),
        webhook_url=alert_cfg.get("webhook_url", ""),
        alert_threshold=alert_cfg.get("alert_threshold", 1),
        alert_cooldown_seconds=alert_cfg.get("alert_cooldown_seconds", 300),
    )

    return camera, detector, classifier, logger, alert_manager


def run(config_path: str = "config.yaml") -> None:
    """Main loop: capture → detect → classify → log → alert → display."""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    camera, detector, classifier, logger, alert_manager = build_components(cfg)
    cl_cfg = cfg.get("classifier", {})
    confidence_threshold = cl_cfg.get("confidence_threshold", 0.7)

    print("Visual Camera Analyzer starting…")
    print("Ensure a 365–395 nm UV blacklight is illuminating the area.")
    print("Press 'q' to quit, 's' to save a snapshot.")

    with camera:
        while True:
            ok, frame = camera.read_frame()
            if not ok or frame is None:
                print("Failed to read frame from camera.", file=sys.stderr)
                break

            # ---- Detection ----
            detections, mask = detector.detect(frame)

            # ---- Classification ----
            for det in detections:
                x, y, w, h = det.bounding_box
                patch = frame[y : y + h, x : x + w]
                if patch.size == 0:
                    continue
                try:
                    label, confidence = classifier.predict(patch)
                    det.label = label
                    det.confidence = confidence
                except RuntimeError:
                    # Model not available — fall back to raw detector label
                    det.label = "fluorescent"
                    det.confidence = 0.0

            # Keep only confident urine detections for alerting
            confident_detections = [
                d
                for d in detections
                if d.confidence >= confidence_threshold or d.label == "fluorescent"
            ]

            # ---- Logging ----
            if confident_detections:
                annotated = detector.annotate_frame(frame, confident_detections)
                logger.log(confident_detections, annotated)
                alert_manager.check_and_alert(confident_detections)
            else:
                annotated = detector.annotate_frame(frame, [])

            # ---- Display ----
            cv2.imshow("Visual Camera Analyzer (UV Urine Detector)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                snapshot_path = os.path.join(
                    cfg.get("logging", {}).get("image_dir", "logs/images"),
                    "snapshot.jpg",
                )
                cv2.imwrite(snapshot_path, annotated)
                print(f"Snapshot saved: {snapshot_path}")

    cv2.destroyAllWindows()
    print(
        f"\nSession complete. "
        f"Total detections logged: {logger.total_detections()}, "
        f"Alerts sent: {alert_manager.total_alerts_sent}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual Camera Analyzer – UV Cat Urine Detector"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
