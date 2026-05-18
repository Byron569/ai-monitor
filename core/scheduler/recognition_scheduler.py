"""
RecognitionScheduler v2 — 按需识别调度器（边缘优化版）。

核心策略:
  1. 新 track 优先识别（稳定 3 帧以上）
  2. 未识别的人: cooldown 内不重试, 失败后 backoff 递增
  3. 已识别的人: recognized_cooldown 更长，不轻易重验证
  4. 队列堆积时只允许高质量人脸进入
  5. max_attempts 限制每 track 的总失败次数
"""

from typing import Dict, List, Optional


class _TrackState:
    __slots__ = (
        "last_attempt", "last_success", "pending",
        "identified", "fail_count", "identity",
    )

    def __init__(self):
        self.last_attempt = -999
        self.last_success = -999
        self.pending = False
        self.identified = False
        self.fail_count = 0
        self.identity = "Unknown"


class RecognitionScheduler:
    """
    识别调度决策器 v2。

    参数:
        cooldown:          未识别者重试冷却 (帧)
        recognized_cooldown: 已识别者重验证冷却 (帧)
        failed_backoff:    每次失败后的额外冷却 (帧)
        max_attempts:      每 track 最大失败次数 (0=不限)
    """

    def __init__(
        self,
        cooldown: int = 300,
        recognized_cooldown: int = 600,
        failed_backoff: int = 90,
        max_attempts: int = 20,
    ) -> None:
        self._cooldown = cooldown
        self._recognized_cooldown = recognized_cooldown
        self._failed_backoff = failed_backoff
        self._max_attempts = max_attempts
        self._states: Dict[int, _TrackState] = {}

    def get_next(
        self,
        persons: list,
        frame_id: int,
        queue_pressure: bool = False,
    ) -> Optional[object]:
        """
        返回下一个应被识别的人。

        queue_pressure=True 时只允许高质量候选（新 track / 从未失败过的 track）。
        """
        # 优先级1: 新 track（从未进入调度器，且已稳定3帧）
        for p in persons:
            if p.frame_seen < 3:
                continue
            if p.track_id not in self._states:
                return p

        # 优先级2: 未识别 + 冷却期已过
        for p in persons:
            if p.is_identified:
                continue
            st = self._states.get(p.track_id)
            if st is None:
                return p
            if st.pending:
                continue
            if self._max_attempts > 0 and st.fail_count >= self._max_attempts:
                continue
            # 计算有效冷却 = base_cooldown + fail_count * backoff
            effective_cd = self._cooldown + st.fail_count * self._failed_backoff
            if frame_id - st.last_attempt >= effective_cd:
                if queue_pressure and st.fail_count > 2:
                    continue
                return p

        # 优先级3: 已识别但长时间未验证
        for p in persons:
            if not p.is_identified:
                continue
            st = self._states.get(p.track_id)
            if st is None:
                continue
            if st.pending:
                continue
            if frame_id - st.last_success >= self._recognized_cooldown:
                if not queue_pressure:
                    return p

        return None

    def mark_submitted(self, track_id: int) -> None:
        if track_id not in self._states:
            self._states[track_id] = _TrackState()
        self._states[track_id].pending = True

    def mark_identified(self, track_id: int, frame_id: int, identity: str = "") -> None:
        if track_id not in self._states:
            self._states[track_id] = _TrackState()
        st = self._states[track_id]
        st.pending = False
        st.identified = True
        st.fail_count = 0
        st.last_success = frame_id
        st.last_attempt = frame_id
        if identity:
            st.identity = identity

    def mark_completed(self, track_id: int, frame_id: int) -> None:
        if track_id not in self._states:
            self._states[track_id] = _TrackState()
        st = self._states[track_id]
        st.pending = False
        st.last_attempt = frame_id
        st.fail_count += 1

    def clear(self, track_id: int) -> None:
        self._states.pop(track_id, None)

    def force_recognize(self, track_id: int) -> None:
        self._states.pop(track_id, None)

    def get_identity(self, track_id: int) -> str:
        st = self._states.get(track_id)
        return st.identity if st else "Unknown"

    def is_cooldown_active(self, track_id: int, frame_id: int) -> bool:
        st = self._states.get(track_id)
        if st is None:
            return False
        if st.identified:
            return frame_id - st.last_success < self._recognized_cooldown
        return frame_id - st.last_attempt < self._cooldown

    def cleanup(self, active_ids: set, frame_id: int) -> None:
        stale = [
            tid for tid in self._states
            if tid not in active_ids
            and frame_id - self._states[tid].last_attempt > 2000
        ]
        for tid in stale:
            del self._states[tid]

    @property
    def pending_count(self) -> int:
        return sum(1 for s in self._states.values() if s.pending)
