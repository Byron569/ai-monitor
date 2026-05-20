"""
Webhook sender for WeCom (企业微信) and DingTalk (钉钉) group bots.

Provides standalone send functions as well as a rate-limited WebhookManager
that enforces per-event-type cooldowns.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standalone send functions
# ---------------------------------------------------------------------------


def send_wecom(text: str, webhook_url: str) -> bool:
    """Send a Markdown message to a WeCom group bot.

    Args:
        text: Markdown-formatted message content.
        webhook_url: WeCom bot webhook URL.

    Returns:
        True if the server responded with HTTP 200.
    """
    try:
        import requests

        payload = {"msgtype": "markdown", "markdown": {"content": text}}
        r = requests.post(webhook_url, json=payload, timeout=5)
        return r.status_code == 200
    except Exception as exc:
        logger.warning("[Webhook] send_wecom failed: %s", exc)
        return False


def send_dingtalk(text: str, webhook_url: str) -> bool:
    """Send a Markdown message to a DingTalk group bot.

    Args:
        text: Markdown-formatted message content.
        webhook_url: DingTalk bot webhook URL.

    Returns:
        True if the server responded with HTTP 200.
    """
    try:
        import requests

        payload = {
            "msgtype": "markdown",
            "markdown": {"title": "AI Monitor Alert", "text": text},
        }
        r = requests.post(webhook_url, json=payload, timeout=5)
        return r.status_code == 200
    except Exception as exc:
        logger.warning("[Webhook] send_dingtalk failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Alert formatters
# ---------------------------------------------------------------------------


def format_fall_alert(name: str, camera_id: str, confidence: float) -> str:
    """Format a fall-detection alert message in Markdown.

    Args:
        name: Recognised person name (or "Unknown").
        camera_id: Camera identifier.
        confidence: Fall detection confidence (0..1).

    Returns:
        Markdown string suitable for WeCom / DingTalk.
    """
    now = datetime.now().strftime("%H:%M:%S")
    return (
        f"## :rotating_light: 摔倒告警\n\n"
        f"**身份**: {name}\n"
        f"**摄像头**: {camera_id}\n"
        f"**置信度**: {confidence:.0%}\n"
        f"**时间**: {now}"
    )


def format_stranger_alert(name: str, camera_id: str) -> str:
    """Format a stranger-detection alert message."""
    now = datetime.now().strftime("%H:%M:%S")
    return (
        f"## :warning: 陌生人告警\n\n"
        f"**身份**: {name}\n"
        f"**摄像头**: {camera_id}\n"
        f"**时间**: {now}"
    )


def format_system_error(error_msg: str) -> str:
    """Format a system error alert message."""
    now = datetime.now().strftime("%H:%M:%S")
    return (
        f"## :exclamation: 系统错误\n\n"
        f"**错误**: {error_msg}\n"
        f"**时间**: {now}"
    )


# ---------------------------------------------------------------------------
# Default cooldowns (seconds)
# ---------------------------------------------------------------------------

DEFAULT_COOLDOWNS: Dict[str, int] = {
    "fall_detected": 60,
    "stranger_alert": 120,
    "system_error": 300,
}


class WebhookManager:
    """Rate-limited webhook manager.

    Sends events to configured WeCom / DingTalk webhooks while enforcing
    per-event-type cooldowns to avoid flooding the chat.

    Usage::

        wh = WebhookManager({"wecom": "https://qyapi.weixin.qq.com/...", "dingtalk": None})
        wh.send_event("fall_detected", name="Alice", camera_id="cam0", confidence=0.95)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Dict with optional keys:
                - ``wecom``: WeCom webhook URL (or None to disable).
                - ``dingtalk``: DingTalk webhook URL (or None to disable).
                - ``rate_limit``: Default rate limit in seconds (default 60).
                - ``cooldowns``: Per-event-type cooldowns dict (overrides DEFAULT_COOLDOWNS).
        """
        self._config = config or {}
        self._wecom_url: Optional[str] = self._config.get("wecom")
        self._dingtalk_url: Optional[str] = self._config.get("dingtalk")

        # Per-event-type cooldowns
        user_cooldowns = self._config.get("cooldowns", {})
        self._cooldowns: Dict[str, int] = {**DEFAULT_COOLDOWNS, **user_cooldowns}

        # Default fallback cooldown
        self._default_cooldown = self._config.get("rate_limit", 60)

        # Last-send time per event_type
        self._last_send: Dict[str, float] = {}

        # Warn if neither webhook is configured
        if not self._wecom_url and not self._dingtalk_url:
            logger.info("[WebhookManager] no webhooks configured — sends will be no-ops")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_event(self, event_type: str, **kwargs: Any) -> bool:
        """Send a rate-limited event to all configured webhooks.

        Args:
            event_type: Type of event (e.g. ``"fall_detected"``, ``"stranger_alert"``,
                        ``"system_error"``). The cooldown is looked up from
                        ``self._cooldowns`` or uses the default rate_limit.
            **kwargs: Extra context passed to the formatter (e.g. ``name``,
                      ``camera_id``, ``confidence``).

        Returns:
            True if at least one webhook was sent successfully.

        Note:
            Returns True immediately if the event is within cooldown to
            indicate it was *handled* (suppressed is the intended behaviour).
        """
        # --- Rate-limit check --------------------------------------------------
        now = time.time()
        cooldown = self._cooldowns.get(event_type, self._default_cooldown)
        last = self._last_send.get(event_type, 0.0)
        if now - last < cooldown:
            remaining = cooldown - (now - last)
            logger.debug(
                "[WebhookManager] %s suppressed (%.1fs remaining)",
                event_type,
                remaining,
            )
            return True  # Suppressed — not an error

        # --- Format message ----------------------------------------------------
        text = self._format(event_type, **kwargs)
        if text is None:
            logger.warning("[WebhookManager] unknown event_type: %s", event_type)
            return False

        # Record attempt time BEFORE sending (rate limiting tracks attempts)
        self._last_send[event_type] = now

        # --- Send to configured webhooks ---------------------------------------
        sent_ok = False
        if self._wecom_url:
            if send_wecom(text, self._wecom_url):
                sent_ok = True

        if self._dingtalk_url:
            if send_dingtalk(text, self._dingtalk_url):
                sent_ok = True

        if sent_ok:
            logger.info("[WebhookManager] %s sent", event_type)
        else:
            logger.warning("[WebhookManager] %s failed to send", event_type)

        return sent_ok

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format(event_type: str, **kwargs: Any) -> Optional[str]:
        """Format an event as Markdown based on its type."""
        name = kwargs.get("name", "Unknown")
        camera_id = kwargs.get("camera_id", "?")
        confidence = kwargs.get("confidence", 0.0)

        if event_type == "fall_detected":
            return format_fall_alert(name, camera_id, confidence)
        if event_type == "stranger_alert":
            return format_stranger_alert(name, camera_id)
        if event_type == "system_error":
            return format_system_error(kwargs.get("error_msg", "Unknown error"))
        return None

    @property
    def enabled(self) -> bool:
        """True if at least one webhook is configured."""
        return bool(self._wecom_url or self._dingtalk_url)
