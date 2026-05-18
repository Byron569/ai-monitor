"""
EventSystem — 统一事件系统。

事件类型:
  person_entered     — 新 track 进入监控
  person_left        — track 消失
  loitering_detected — 徘徊行为
  restricted_entered — 进入禁区
  stationary_detected — 长时间静止

用法:
    es = EventSystem()
    es.emit("person_entered", track_id=3, metadata={"name": "Bob"})
    events = es.flush()
"""

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _now_ts() -> str:
    return time.strftime("%H:%M:%S")


class EventSystem:
    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def emit(self, event_type: str, track_id: int, **metadata):
        event = {
            "event_type": event_type,
            "track_id": track_id,
            "timestamp": _now_ts(),
            "metadata": metadata,
        }
        self._events.append(event)
        logger.info(f"[EVENT] {event_type} track={track_id}")

    def flush(self) -> List[Dict[str, Any]]:
        events = self._events[:]
        self._events.clear()
        return events

    @property
    def pending_count(self) -> int:
        return len(self._events)
