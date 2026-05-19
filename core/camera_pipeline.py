"""
CameraPipeline — per-camera independent processing thread.

Reuses core pipeline logic: detection → tracking → recognition → render.
Each camera has its own tracker, TrackMemory, PersonManager, frame counter.
Shared: RecognitionWorker, FaceDatabase, GlobalInferenceScheduler.
"""

import logging
import threading
import time

import cv2
import numpy as np

from core.pipeline.recognition_mixin import process_recognition_harvest

logger = logging.getLogger(__name__)


class CameraPipeline(threading.Thread):

    def __init__(
        self,
        camera_id,
        source,
        detector,
        tracker,
        person_manager,
        recog_scheduler,
        worker,
        quality_filter=None,
        motion_gate=None,
        frame_scheduler=None,
        renderer=None,
        render=False,
        inference_scheduler=None,
        database=None,
        metrics=None,
    ):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self._source = source
        self.detector = detector
        self.tracker = tracker
        self.person_manager = person_manager
        self.recog_scheduler = recog_scheduler
        self.worker = worker
        self.quality_filter = quality_filter
        self.motion_gate = motion_gate
        self._render = render
        self._inference_scheduler = inference_scheduler
        self.database = database
        self.metrics = metrics
        self.renderer = renderer

        self.frame_scheduler = frame_scheduler
        if self.frame_scheduler is None:
            from core.frame_scheduler import FrameScheduler
            self.frame_scheduler = FrameScheduler(detection_interval=4)

        self._running = False
        self._frame_count = 0
        self._track_memory = tracker.get_memory()
        self._next_tid = 1
        self._landmarks_cache = {}
        self._recog_enqueued = 0
        self._recog_skipped = 0
        self._recog_cache_hit = 0
        self._recog_rejected = 0

        self._cap = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 10.0

    # --- camera ---

    def _open_camera(self):
        try:
            self._cap = cv2.VideoCapture(self._source)
            if not self._cap.isOpened():
                self._cap = None
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
        except Exception:
            self._cap = None
            return False

    def _release_camera(self):
        if self._cap:
            try: self._cap.release()
            except Exception: pass
        self._cap = None

    def _try_reconnect(self):
        delay = self._reconnect_delay
        while self._running and delay <= self._max_reconnect_delay:
            time.sleep(delay)
            self._release_camera()
            if self._open_camera():
                if self.metrics: self.metrics.record_reconnect()
                self._reconnect_delay = 1.0
                return True
            delay = min(delay * 2, self._max_reconnect_delay)
        return False

    # --- run ---

    def run(self):
        self._running = True
        if not self._open_camera():
            if not self._try_reconnect():
                return

        while self._running:
            if self._cap is None:
                if not self._try_reconnect(): break
                continue

            try:
                ret, frame = self._cap.read()
                if not ret:
                    if self.metrics: self.metrics.record_drop()
                    self._release_camera()
                    if not self._try_reconnect(): break
                    continue
            except Exception:
                if self.metrics: self.metrics.record_drop()
                self._release_camera()
                if not self._try_reconnect(): break
                continue

            self._frame_count += 1
            if self.metrics: self.metrics.record_frame()

            # motion
            motion_active, motion_score = True, 0
            if self.motion_gate:
                motion_active, motion_score = self.motion_gate.check(frame)

            # detection
            can_detect = self.frame_scheduler.should_detect(self._frame_count, motion_active, motion_score)[0]
            if self._inference_scheduler and can_detect:
                can_detect = self._inference_scheduler.acquire(self.camera_id)

            detections = []
            if can_detect:
                t0 = time.perf_counter()
                try: detections = self.detector.detect(frame)
                except Exception: pass
                if self.metrics: self.metrics.record_detect((time.perf_counter() - t0) * 1000)
                if self._inference_scheduler:
                    self._inference_scheduler.release(self.camera_id)

            # tracking + track memory
            t0 = time.perf_counter()
            track_dets = [(d[0], d[1], d[2], d[3], d[4]) for d in detections]
            if track_dets: self.tracker.update(track_dets, frame)
            active_tracks = {}
            if detections:
                det_bboxes = [(d[0], d[1], d[2], d[3]) for d in detections]
                matches = self._track_memory.match_hungarian(det_bboxes)
                for det_idx, tid in matches.items():
                    active_tracks[tid] = det_bboxes[det_idx]
                for i, bbox in enumerate(det_bboxes):
                    if i not in matches:
                        tid = self._next_tid; self._next_tid += 1
                        active_tracks[tid] = bbox
            self._track_memory.update(active_tracks)
            for tid in list(self._track_memory._tracks.keys()):
                if self._track_memory._tracks[tid].status == "lost":
                    self._track_memory.clear_lock(tid)
            if self.metrics: self.metrics.record_track((time.perf_counter() - t0) * 1000)

            # person manager
            for tid, (x1, y1, x2, y2) in active_tracks.items():
                self.person_manager.get_or_create(tid, (x1, y1, x2, y2))
            self.person_manager.cleanup()

            # recognition
            t0 = time.perf_counter()
            self._process_recognition(frame)
            if self.metrics: self.metrics.record_recognize((time.perf_counter() - t0) * 1000)

            # periodic status (every 60 frames)
            if self._frame_count % 60 == 0:
                s = self.metrics.summary() if self.metrics else {}
                persons = self.person_manager.stable_count
                lost = self._track_memory.lost_count
                qs = self.worker.queue_size
                logger.info(
                    f"[{self.camera_id}] fps={s.get('fps',0):.0f} "
                    f"det={s.get('detect_ms',0):.0f}ms trk={s.get('track_ms',0):.1f}ms "
                    f"rec={s.get('recog_ms',0):.1f}ms | persons={persons} lost={lost} "
                    f"| enq={self._recog_enqueued} skip={self._recog_skipped} "
                    f"rej={self._recog_rejected} cache={self._recog_cache_hit} q={qs}"
                )

            # render
            if self._render and self.renderer:
                for tid, person in self.person_manager.get_active().items():
                    if person.frame_seen < 3: continue
                    self.renderer.draw_face_identity(frame, bbox=person.bbox, name=person.identity,
                                                      similarity=0.85 if person.is_identified else 0.0, confidence=0.9)
                cv2.imshow(str(self.camera_id), frame)
                cv2.waitKey(1)

        self._release_camera()

    # --- recognition ---

    def _process_recognition(self, frame):
        active = list(self.person_manager.get_active().values())
        qp = self.worker.queue_size >= 3
        target = self.recog_scheduler.get_next(active, self._frame_count, qp)
        if target is not None:
            tid = target.track_id
            cached = self.person_manager.lookup_embedding(tid)
            if cached is not None and cached[0] != "Unknown":
                self.person_manager.identify(tid, cached[0], cached[1])
                self.recog_scheduler.mark_identified(tid, self._frame_count, cached[0])
                self._recog_cache_hit += 1
                return
            x1, y1, x2, y2 = target.bbox
            if y2 <= y1 or x2 <= x1: return
            if self.quality_filter:
                qr = self.quality_filter.evaluate(frame, target.bbox)
                if not qr.passed:
                    self._recog_rejected += 1
                    return
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: return
            if self.worker.submit(tid, crop, quality_score=0.5):
                self.recog_scheduler.mark_submitted(tid)
                self._recog_enqueued += 1
            else:
                self._recog_skipped += 1

        process_recognition_harvest(
            self.worker, self.person_manager, self.recog_scheduler,
            self._frame_count,
        )
        ids = set(self.person_manager.get_active().keys())
        self.recog_scheduler.cleanup(ids, self._frame_count)

    def stop(self):
        self._running = False
