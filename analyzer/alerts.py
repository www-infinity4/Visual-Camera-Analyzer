"""
Alert System

Sends notifications when cat urine is detected.  Supports:
- Email alerts via SMTP (e.g. Gmail with an app password)
- HTTP webhook alerts for smart-home / IoT integrations (e.g. Home Assistant,
  IFTTT, Node-RED)

Alerts include a cooldown period to avoid notification flooding and a
configurable detection threshold.
"""

from __future__ import annotations

import json
import smtplib
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

try:
    import requests as _requests

    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REQUESTS_AVAILABLE = False

from analyzer.detector import Detection


class AlertManager:
    """
    Manages detection alerts via email and/or HTTP webhooks.

    Implements a cooldown period between alerts to prevent flooding.
    """

    def __init__(
        self,
        # Email settings
        email_enabled: bool = False,
        smtp_host: str = "smtp.example.com",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        recipient_email: str = "",
        # Webhook settings
        webhook_enabled: bool = False,
        webhook_url: str = "",
        # Threshold / cooldown
        alert_threshold: int = 1,
        alert_cooldown_seconds: float = 300.0,
    ):
        self.email_enabled = email_enabled
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.recipient_email = recipient_email

        self.webhook_enabled = webhook_enabled
        self.webhook_url = webhook_url

        self.alert_threshold = alert_threshold
        self.alert_cooldown_seconds = alert_cooldown_seconds

        self._last_alert_time: float = 0.0
        self._alert_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_alert(self, detections: List[Detection]) -> bool:
        """
        Evaluate detections and send an alert if conditions are met.

        Conditions:
        1. Number of urine detections >= alert_threshold
        2. Cooldown period since last alert has elapsed

        Args:
            detections: Detections from the current frame.

        Returns:
            True if an alert was sent, False otherwise.
        """
        urine_detections = [d for d in detections if d.label == "urine"]
        if len(urine_detections) < self.alert_threshold:
            return False

        now = time.monotonic()
        if now - self._last_alert_time < self.alert_cooldown_seconds:
            return False  # Still in cooldown

        self._last_alert_time = now
        self._alert_count += 1
        payload = self._build_payload(urine_detections)

        sent = False
        if self.email_enabled:
            self._send_email(payload)
            sent = True
        if self.webhook_enabled:
            self._send_webhook(payload)
            sent = True

        return sent

    @property
    def total_alerts_sent(self) -> int:
        """Total number of alerts sent in this session."""
        return self._alert_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_payload(self, detections: List[Detection]) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_type": "urine_detected",
            "detection_count": len(detections),
            "detections": [
                {
                    "label": d.label,
                    "confidence": round(d.confidence, 4),
                    "centroid": {"x": d.center_x, "y": d.center_y},
                    "area": round(d.area, 2),
                }
                for d in detections
            ],
        }

    def _send_email(self, payload: Dict[str, Any]) -> None:
        """Send an email alert using SMTP."""
        count = payload["detection_count"]
        subject = f"[Visual Camera Analyzer] Cat urine detected: {count} spot(s)"
        body = (
            f"Alert time: {payload['timestamp']}\n"
            f"Detections: {count}\n\n"
            + "\n".join(
                f"  - Spot at ({d['centroid']['x']}, {d['centroid']['y']}), "
                f"area={d['area']:.0f}px², confidence={d['confidence']:.0%}"
                for d in payload["detections"]
            )
        )

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.smtp_user
        msg["To"] = self.recipient_email

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.sendmail(self.smtp_user, [self.recipient_email], msg.as_string())

    def _send_webhook(self, payload: Dict[str, Any]) -> None:
        """POST a JSON payload to the configured webhook URL."""
        if not _REQUESTS_AVAILABLE:
            raise ImportError(
                "The 'requests' package is required for webhook alerts. "
                "Install it with: pip install requests"
            )
        response = _requests.post(
            self.webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
