"""QQ email notification sender for fall alerts."""

from __future__ import annotations

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send email alerts via QQ SMTP."""

    def __init__(
        self,
        smtp_host: str = "smtp.qq.com",
        smtp_port: int = 465,
        username: str = "",
        password: str = "",
        receiver: str = "",
    ):
        self._host = smtp_host
        self._port = smtp_port
        self._user = username
        self._pwd = password
        self._to = receiver or username

    def send_alert(self, subject: str, body: str) -> bool:
        try:
            msg = MIMEMultipart()
            msg["From"] = self._user
            msg["To"] = self._to
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))

            with smtplib.SMTP_SSL(self._host, self._port, timeout=10) as srv:
                srv.login(self._user, self._pwd)
                srv.sendmail(self._user, [self._to], msg.as_string())

            logger.info("[EmailNotifier] alert sent: %s", subject)
            return True
        except Exception as exc:
            logger.warning("[EmailNotifier] send failed: %s", exc)
            return False

    @property
    def enabled(self) -> bool:
        return bool(self._user and self._pwd)


def format_fall_email(name: str, camera_id: str, confidence: float) -> tuple[str, str]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[AI Monitor] 摔倒告警 - {name}"
    body = (
        f"摔倒告警\n"
        f"==================\n"
        f"身份: {name}\n"
        f"摄像头: {camera_id}\n"
        f"置信度: {confidence:.0%}\n"
        f"时间: {now}\n"
    )
    return subject, body
