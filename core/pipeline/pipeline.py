"""
Pipeline v9 — 边缘最小化管线。

核心链路：Camera → MotionGate → FrameScheduler → SCRFD → Tracker → PersonManager
            → RecognitionScheduler → RecognitionWorker → Renderer

可选模块（默认关闭）：TrajectoryAnalyzer, BehaviorEngine, RegionManager, EventSystem, AlertManager
"""

import logging
import time
from typing import Dict

import cv2
import numpy as np

from core.camera import Camera
from core.detectors import SCRFDDetector
from core.frame_scheduler import FrameScheduler
from core.rendering import Renderer
from core.track_memory import TrackMemory
from core.person import PersonManager
from core.scheduler import RecognitionScheduler
from core.workers import RecognitionWorker
from utils import FPS, PerformanceMonitor
from utils.motion_gate import MotionGate

logger = logging.getLogger(__name__)

ARCFACE_SRC = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


def align_face(frame, landmarks, size=112):
    dst = ARCFACE_SRC * (size / 112.0)
    M = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst)[0]
    if M is None:
        return None
    return cv2.warpAffine(frame, M, (size, size), borderValue=0.0)


class Pipeline:
    """边缘最小化人脸识别 Pipeline v9。"""

    def __init__(
        self,
        camera: Camera,
        detector: SCRFDDetector,
        renderer: Renderer,
        fps: FPS,
        tracker,  # Any: MultiObjectTracker | LightweightIoUTracker
        person_manager: PersonManager,
        recog_scheduler: RecognitionScheduler,
        worker: RecognitionWorker,
        monitor: PerformanceMonitor,
        motion_gate: MotionGate,
        frame_scheduler: FrameScheduler,
        track_reassociation = None,
        quality_filter = None,
        trajectory_analyzer = None,
        behavior_engine = None,
        region_manager = None,
        event_system = None,
        alert_manager = None,
        window_name: str = "Vision AI",
        quit_key: str = "q",
        no_render: bool = False,
        max_frames: int = 0,
        queue_pressure_threshold: int = 3,
        tasks: list = None,
    ):
        self.camera = camera
        self.detector = detector
        self.renderer = renderer
        self.fps = fps
        self.tracker = tracker
        self.person_manager = person_manager
        self.recog_scheduler = recog_scheduler
        self.worker = worker
        self.monitor = monitor
        self.motion_gate = motion_gate
        self.frame_scheduler = frame_scheduler
        self.trajectory_analyzer = trajectory_analyzer
        self.behavior_engine = behavior_engine
        self.region_manager = region_manager
        self.event_system = event_system
        self.alert_manager = alert_manager
        self.window_name = window_name
        self.quit_key = quit_key
        self.no_render = no_render
        self.max_frames = max_frames
        self.quality_filter = quality_filter
        self._q_pressure = queue_pressure_threshold
        self._tasks = tasks or []

        self._frame_count = 0
        self._running = False
        self._landmarks_cache: Dict[int, np.ndarray] = {}
        self._track_memory: TrackMemory = tracker.get_memory()
        self._next_tid = 1
        self._overlap_start: Dict[tuple, int] = {}
        self._separation_time: Dict[tuple, int] = {}

        # 初始化阶段确定开关，避免每帧重复判断
        self._behavior_enabled = (
            self.trajectory_analyzer is not None
            or self.behavior_engine is not None
            or self.region_manager is not None
            or self.event_system is not None
        )

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True
        self.worker.start()

        print("=" * 60)
        print("  Pipeline v9 | Edge-Minimal Pipeline")
        print(f"  Detect: {self.frame_scheduler.detection_interval}f | Motion: {self.motion_gate.threshold}")
        print(f"  Behavior: {'ON' if self._behavior_enabled else 'OFF'}")
        print("=" * 60)

        with self.camera as cam:
            while self._running:
                # === 1. Camera ===
                self.monitor.tick("camera")
                ret, frame = cam.read()
                if not ret:
                    continue
                self.monitor.tock("camera", "camera")

                # === 2. Motion Gate ===
                motion_active, motion_score = self.motion_gate.check(frame)

                # === 3. Scheduler 决定是否需要 SCRFD 纠错 ===
                sched_allow, sched_reason = self.frame_scheduler.should_detect(
                    self._frame_count, motion_active, motion_score
                )

                # 额外规则：有 lost track → 必须运行 SCRFD
                lost_count = self._track_memory.lost_count
                need_correction = sched_allow or lost_count > 0

                self.monitor.tick("detect")
                detections = []
                correction_reason = sched_reason
                if need_correction:
                    detections = self.detector.detect(frame)
                    if lost_count > 0 and not sched_allow:
                        correction_reason = f"recovery ({lost_count} lost tracks)"
                self.monitor.tock("detect", "detect")

                # === 4. Tracker + TrackMemory (TrackMemory作为ID权威) ===
                self.monitor.tick("track")
                track_dets = [(d[0], d[1], d[2], d[3], d[4]) for d in detections]
                if track_dets:
                    self.tracker.update(track_dets, frame)
                
                # TrackMemory 贪心匹配（按x排序防止交叉运动抢ID）
                active_tracks = {}
                if detections:
                    det_bboxes = [(d[0], d[1], d[2], d[3]) for d in detections]
                    matches = self._track_memory.match_hungarian(det_bboxes)
                    for det_idx, tid in matches.items():
                        active_tracks[tid] = det_bboxes[det_idx]
                    for i, bbox in enumerate(det_bboxes):
                        if i not in matches:
                            tid = self._next_tid
                            self._next_tid += 1
                            active_tracks[tid] = bbox
                else:
                    active_tracks = {}  # 无检测时不注入缓存，让 PersonManager 自然老化
                self._track_memory.update(active_tracks)

                # 解锁丢失 track 的匹配锁
                for tid in list(self._track_memory._tracks.keys()):
                    ts = self._track_memory._tracks[tid]
                    if ts.status == "lost":
                        self._track_memory.clear_lock(tid)
                self.monitor.tock("track", "track")

                # === 6. PersonManager 同步 ===
                for tid, (x1, y1, x2, y2) in active_tracks.items():
                    self.person_manager.get_or_create(tid, (x1, y1, x2, y2))
                if detections:
                    self._update_landmarks_cache(detections, active_tracks)
                self.person_manager.cleanup()

                # === 7a. Identity Triggers (always active) ===
                # Trigger 1: box out of frame → force re-recognize
                h, w = frame.shape[:2]
                for tid, person in self.person_manager.get_active().items():
                    x1, y1, x2, y2 = person.bbox
                    out_left = x2 <= 0
                    out_right = x1 >= w
                    out_top = y2 <= 0
                    out_bottom = y1 >= h
                    if out_left or out_right or out_top or out_bottom:
                        self.recog_scheduler.force_recognize(tid)
                        self.person_manager.reset_identity(tid)
                        logger.info(f"[TRIGGER] track={tid} left frame, force recognize")

                # Trigger 2: overlap → reset identity; separation 0.5s → unlock + re-recognize
                active_list = list(self.person_manager.get_active().items())
                for i in range(len(active_list)):
                    for j in range(i + 1, len(active_list)):
                        tid_a, pa = active_list[i]
                        tid_b, pb = active_list[j]
                        overlap = self._calc_iou(pa.bbox, pb.bbox)
                        key = (min(tid_a, tid_b), max(tid_a, tid_b))
                        if overlap > 0.3:
                            if key not in self._overlap_start:
                                self._overlap_start[key] = self._frame_count
                                self.person_manager.reset_identity(tid_a)
                                self.person_manager.reset_identity(tid_b)
                                logger.info(f"[TRIGGER] tracks {tid_a},{tid_b} overlapping, reset identity")
                            self._separation_time.pop(key, None)
                        else:
                            prev = self._overlap_start.pop(key, None)
                            if prev is not None and prev > 0:
                                if key not in self._separation_time:
                                    self._separation_time[key] = self._frame_count
                                elif self._frame_count - self._separation_time[key] > 15:
                                    self.recog_scheduler.force_recognize(tid_a)
                                    self.recog_scheduler.force_recognize(tid_b)
                                    self._track_memory.clear_lock(tid_a)
                                    self._track_memory.clear_lock(tid_b)
                                    self._separation_time.pop(key, None)
                                    logger.info(f"[TRIGGER] tracks {tid_a},{tid_b} separated 0.5s, unlock+recognize")

                # === 7b. Behavior Analysis (全部由初始化开关控制) ===
                if self._behavior_enabled:
                    if self.trajectory_analyzer:
                        for tid, (x1, y1, x2, y2) in active_tracks.items():
                            self.trajectory_analyzer.analyze(tid, (x1, y1, x2, y2))
                        active_ids = set(self.person_manager.get_active().keys())
                        self.trajectory_analyzer.cleanup(active_ids)

                    if self.behavior_engine:
                        for tid in list(self.person_manager.get_active().keys()):
                            self.behavior_engine.update(tid, self._frame_count)
                        active_ids = set(self.person_manager.get_active().keys())
                        self.behavior_engine.cleanup(active_ids)

                    if self.region_manager and self.behavior_engine:
                        for tid, person in self.person_manager.get_active().items():
                            bs = self.behavior_engine.get(tid)
                            if bs is None:
                                continue
                            entered, ztype, zname = self.region_manager.check_entry(
                                tid, person.bbox
                            )
                            if entered and ztype == "restricted" and self.event_system:
                                self.event_system.emit(
                                    "restricted_entered", tid,
                                    zone=zname, behavior=bs.behavior.value,
                                )
                            if bs.behavior.value == "loitering" and self.event_system:
                                self.event_system.emit(
                                    "loitering_detected", tid,
                                    confidence=bs.confidence,
                                )

                    if self.event_system:
                        events = self.event_system.flush()
                        if events and self.alert_manager:
                            self.alert_manager.process(events)

                # === 7. Recognition (事件触发：仅新 track / not identified) ===
                self.monitor.tick("recognize")
                self._process_recognition(frame)
                self.monitor.tock("recognize", "recognize")

                # === 7.5. VisionTask Plugins (可插拔扩展任务) ===
                self._run_tasks(frame, active_tracks)

                # === 8. Render (skipped if no_render) ===
                if not self.no_render:
                    self.monitor.tick("render")
                    self._render(frame, motion_active, motion_score, correction_reason)
                    self.monitor.tock("render", "render")

                    cv2.imshow(self.window_name, frame)

                # quit check (always active, even in no_render)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(self.quit_key) or key == ord(self.quit_key.upper()):
                    self._running = False

                self._frame_count += 1

                # max_frames limit for benchmark mode
                if self.max_frames > 0 and self._frame_count >= self.max_frames:
                    self._running = False

        self.worker.stop()
        print(f"[Pipeline] 已退出 (共 {self._frame_count} 帧)")

    # ------------------------------------------------------------------
    # Landmarks cache
    # ------------------------------------------------------------------

    def _update_landmarks_cache(self, detections, active_tracks):
        for tid, (tx1, ty1, tx2, ty2) in active_tracks.items():
            best_iou, best_kps = 0, None
            for d in detections:
                dx1, dy1, dx2, dy2 = d[0], d[1], d[2], d[3]
                iou_val = self._calc_iou((tx1, ty1, tx2, ty2), (dx1, dy1, dx2, dy2))
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_kps = d[5]
            if best_kps is not None and best_iou > 0.3:
                self._landmarks_cache[tid] = np.array(best_kps)

    @staticmethod
    def _calc_iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def _find_or_create_tid(self, bbox: tuple, exclude: set = None) -> int:
        tid = self._track_memory.find_nearest(bbox, max_dist=150, exclude=exclude)
        if tid is not None:
            return tid
        tid = self._next_tid
        self._next_tid += 1
        return tid

    # ------------------------------------------------------------------
    # Recognition (事件触发)
    # ------------------------------------------------------------------

    def _process_recognition(self, frame: np.ndarray) -> None:
        active = list(self.person_manager.get_active().values())
        queue_pressure = self.worker.queue_size >= self._q_pressure
        target = self.recog_scheduler.get_next(active, self._frame_count, queue_pressure)

        if target is not None:
            tid = target.track_id
            x1, y1, x2, y2 = target.bbox

            # Check embedding cache first
            cached = self.person_manager.lookup_embedding(tid)
            if cached is not None:
                cached_name, cached_emb = cached
                if cached_name != "Unknown":
                    self.person_manager.identify(tid, cached_name, cached_emb)
                    self.recog_scheduler.mark_identified(tid, self._frame_count, cached_name)
                    self.monitor.inc("recog_cache_hit")
                    return

            if y2 <= y1 or x2 <= x1:
                return

            # Face quality filter
            quality_result = None
            if self.quality_filter:
                quality_result = self.quality_filter.evaluate(frame, target.bbox)
                if not quality_result.passed:
                    self.monitor.inc("recog_quality_reject")
                    return

            qs = quality_result.score if quality_result else 0.5

            # Crop and submit
            kps = self._landmarks_cache.get(tid)
            crop = None
            if kps is not None:
                aligned = align_face(frame, kps, size=112)
                if aligned is not None and aligned.size > 0:
                    crop = aligned

            if crop is None:
                crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                return

            force = (quality_result is not None and quality_result.score > 0.8)
            if self.worker.submit(tid, crop, quality_score=qs, force=force):
                self.recog_scheduler.mark_submitted(tid)
                self.monitor.inc("recog_enqueue")
                self.monitor.set_gauge("recog_queue_size", self.worker.queue_size)
            else:
                self.monitor.inc("recog_skip")

        # Harvest results
        results = self.worker.poll_results()
        for tid, name, sim, emb, qs in results:
            if name != "Unknown" and emb is not None:
                self.person_manager.identify(tid, name, emb)
                self.recog_scheduler.mark_identified(tid, self._frame_count, name)
                self.person_manager.cache_embedding(tid, name, emb)
            elif emb is not None:
                cached_name, _ = self.person_manager.find_cached_identity(emb)
                if cached_name is not None:
                    self.person_manager.identify(tid, cached_name, emb)
                    self.recog_scheduler.mark_identified(tid, self._frame_count, cached_name)
                    self.person_manager.cache_embedding(tid, cached_name, emb)
                else:
                    self.recog_scheduler.mark_completed(tid, self._frame_count)
            else:
                self.recog_scheduler.mark_completed(tid, self._frame_count)

        # Hard dedup: one registered name max
        seen: Dict[str, int] = {}
        for tid, person in self.person_manager.get_active().items():
            if person.is_identified and person.identity != "Unknown":
                if person.identity in seen:
                    self.person_manager.reset_identity(tid)
                    self.recog_scheduler.mark_completed(tid, self._frame_count)
                    st = self.recog_scheduler._states.get(tid)
                    if st:
                        st.last_attempt = self._frame_count + self.recog_scheduler._cooldown
                else:
                    seen[person.identity] = tid

        active_ids = set(self.person_manager.get_active().keys())
        self.recog_scheduler.cleanup(active_ids, self._frame_count)

    # ------------------------------------------------------------------
    # VisionTask Plugins
    # ------------------------------------------------------------------

    def _build_context(self):
        return {
            "frame_count": self._frame_count,
            "person_manager": self.person_manager,
            "event_system": self.event_system,
        }

    def _run_tasks(self, frame, active_tracks) -> None:
        if not self._tasks:
            return

        tracks = []
        for tid, bbox in active_tracks.items():
            person = self.person_manager.get(tid)
            ident = person.identity if person else "Unknown"
            tracks.append((tid, bbox, ident))

        ctx = self._build_context()

        for task in self._tasks:
            if not task.enabled:
                continue
            try:
                if not task.should_run(self._frame_count, tracks, ctx):
                    continue
            except Exception:
                continue

            self.monitor.tick(f"task.{task.name}")
            try:
                events = task.run(frame, tracks, ctx)
            except Exception:
                events = []
            self.monitor.tock(f"task.{task.name}", f"task.{task.name}")

            if events and self.event_system:
                for evt in events:
                    self.event_system.emit(
                        evt.event_type, evt.track_id,
                        confidence=evt.confidence,
                        **(evt.payload),
                    )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(self, frame, motion_active, motion_score, reason):
        # 硬去重：注册用户只保留第一个框
        seen_names = set()
        for tid, person in list(self.person_manager.get_active().items()):
            if person.is_identified and person.identity != "Unknown":
                if person.identity in seen_names:
                    self.person_manager.reset_identity(tid)
                    self.recog_scheduler.force_recognize(tid)
                    self.recog_scheduler.mark_completed(tid, self._frame_count)
                    st = self.recog_scheduler._states.get(tid)
                    if st:
                        st.last_attempt = self._frame_count + 600
                    continue
                seen_names.add(person.identity)

        for tid, person in self.person_manager.get_active().items():
            if person.frame_seen < 3:
                continue
            name = person.identity
            sim = 0.85 if name != "Unknown" else 0.0
            self.renderer.draw_face_identity(
                frame, bbox=person.bbox, name=name, similarity=sim, confidence=0.9,
            )
            cv2.putText(
                frame, f"ID:{tid}", (person.bbox[0], person.bbox[3] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
            )

        # 绘制摔倒检测结果 (人体框 + 骨架 + 状态)
        fall_count = 0
        for task in self._tasks:
            if not task.enabled:
                continue
            results = getattr(task, "last_results", {})
            for tid, result in results.items():
                body_bbox = result.get("bbox")
                if body_bbox:
                    self.renderer.draw_body_bbox(frame, body_bbox)
                kps = result.get("keypoints")
                if kps:
                    self.renderer.draw_skeleton(frame, kps)
                fs = result.get("fall_state", "")
                if fs and fs != "Normal":
                    fall_count += 1
                self.renderer.draw_fall_status(
                    frame,
                    bbox=body_bbox or [0, 0, 0, 0],
                    fall_state=fs,
                    confidence=result.get("confidence", 0.0),
                )

        self.fps.update()
        self.renderer.draw_fps(frame, self.fps.get_fps())

        motion_str = f"{motion_score:.1f}({'Y' if motion_active else 'N'})"
        info_lines = [
            f"Persons: {self.person_manager.stable_count} | Lost: {self._track_memory.lost_count}",
            f"Fall: {fall_count} alert(s) | Motion: {motion_str} | {reason}",
            f"Frame: #{self._frame_count} | Q: {self.worker.queue_size} | Cache: {self.person_manager.cache_size}",
            self.monitor.report(),
        ]

        # 每60帧输出识别统计
        if self.monitor.should_report(self._frame_count, 60):
            logger.info(self.monitor.recog_report())

        self.renderer.draw_system_info(frame, info_lines)

    def stop(self) -> None:
        self._running = False
