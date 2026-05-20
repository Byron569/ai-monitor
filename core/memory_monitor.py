"""
MemoryMonitor -- adaptive memory-pressure monitor.

Monitors system memory usage so the pipeline can degrade gracefully
when the system is under memory pressure.

Usage:
    monitor = MemoryMonitor(max_percent=85, check_interval=5.0)
    decision = monitor.check(frame_count)

    if decision["should_degrade"]:
        actions = monitor.get_degradation_actions()
        apply_degradation(actions)
"""

from __future__ import annotations

import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional psutil import -- memory monitoring degrades gracefully when psutil
# is not installed (e.g., on minimal embedded images).
# ---------------------------------------------------------------------------

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed; memory monitoring is disabled.")


class MemoryMonitor:
    """Tracks system memory pressure and recommends degradation actions.

    Thresholds
    ----------
    ``max_percent`` : int
        Pressure above this value triggers degradation.
    ``recover_percent`` : int
        Pressure must fall to (max_percent - 10) for auto-recovery.
    ``check_interval`` : float
        Minimum seconds between checks; calling ``check()`` more frequently
        returns the last decision without querying the OS again.
    """

    def __init__(self, max_percent: int = 85, check_interval: float = 5.0) -> None:
        self._max = max_percent
        self._recover = max_percent - 10
        self._interval = check_interval
        self._last_check: float = 0.0
        self._degraded: bool = False
        self._last_pressure: float = 0.0
        self._last_action: str = "normal"

        # Validate thresholds
        if self._recover < 0:
            self._recover = 10  # sensible floor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, frame_count: int) -> Dict[str, object]:
        """Return a degradation decision dict.

        Returns
        -------
        dict with keys:
            should_degrade : bool
            pressure       : float   (0-100, 0 when psutil is unavailable)
            action         : str     ("normal", "degrade", "recover")
            frame_count    : int     (passthrough for logging)
        """
        now = time.monotonic()
        decision: Dict[str, object] = {
            "should_degrade": self._degraded,
            "pressure": self._last_pressure,
            "action": self._last_action,
            "frame_count": frame_count,
        }

        # Throttle -- only check every ``_interval`` seconds.
        if now - self._last_check < self._interval:
            return decision

        self._last_check = now

        # When psutil is unavailable we report 0 pressure and stay normal.
        if not _PSUTIL_AVAILABLE:
            self._last_pressure = 0.0
            self._last_action = "normal"
            self._degraded = False
            return decision

        pressure = psutil.virtual_memory().percent
        self._last_pressure = pressure

        if pressure > self._max:
            self._degraded = True
            self._last_action = "degrade"
            logger.info(
                "Memory pressure %.1f%% exceeds limit %d%% -- degrading",
                pressure,
                self._max,
            )
        elif self._degraded and pressure <= self._recover:
            self._degraded = False
            self._last_action = "recover"
            logger.info(
                "Memory pressure dropped to %.1f%% (recover threshold %d%%) -- "
                "recovering",
                pressure,
                self._recover,
            )

        decision["should_degrade"] = self._degraded
        decision["pressure"] = pressure
        decision["action"] = self._last_action
        return decision

    def get_degradation_actions(self) -> Dict[str, bool]:
        """Return recommended degradation actions when degraded.

        Returns
        -------
        dict with recommended flags:
            double_fall_interval : bool  (double the frame-skip interval)
            skip_behavior        : bool  (skip behavioral analysis)
            clear_cache          : bool  (clear internal caches)
        """
        degraded = self._degraded
        return {
            "double_fall_interval": degraded,
            "skip_behavior": degraded,
            "clear_cache": degraded,
        }

    # ------------------------------------------------------------------
    # Properties (useful for tests and introspection)
    # ------------------------------------------------------------------

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def pressure(self) -> float:
        return self._last_pressure

    @property
    def max_percent(self) -> int:
        return self._max

    @property
    def recover_percent(self) -> int:
        return self._recover

    @property
    def check_interval(self) -> float:
        return self._interval
