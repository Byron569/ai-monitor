"""
FallDetectionWorker — 异步摔倒检测工作线程 (边缘优化版)。

架构: 主线程 submit(frame) 非阻塞 → Worker 线程推理+追踪+evaluate_fall
       → 主线程 poll_results() 收割结果

参照 RecognitionWorker 模式: daemon thread + bounded queue + 异常隔离
"""

import logging
import time
import threading
from queue import Queue, Full, Empty

import numpy as np

logger = logging.getLogger(__name__)


class FallDetectionWorker(threading.Thread):
    """异步摔倒检测 Worker 线程。

    主线程接口:
      - submit(frame, frame_id) → bool (非阻塞)
      - poll_results() → dict (非阻塞)
      - last_results: dict (直接读缓存)
    """

    def __init__(
        self,
        inference_backend,
        max_queue_size: int = 2,
        interval: int = 15,
        confidence_threshold: float = 0.5,
    ):
        super().__init__(daemon=True)
        self._backend = inference_backend
        self._input_queue: Queue = Queue(maxsize=max_queue_size)
        self._output_queue: Queue = Queue()
        self._running = True
        self._interval = interval
        self._conf_threshold = confidence_threshold
        self._frame_count = 0

        # 内部追踪 (Worker 拥有, 不受 PersonManager 影响)
        self._fall_tracks: dict = {}
        self._next_fall_tid: int = 1000
        self._track_max_lost: int = 30

        # 最新结果缓存 (主线程直接读)
        self.last_results: dict = {}

        self._processed_count = 0
        self._skip_count = 0

    # ==================================================================
    # 主线程接口 (非阻塞)
    # ==================================================================

    def submit(self, frame, frame_id: int, person_tracks: list) -> bool:
        """提交帧到推理队列 (非阻塞)。队列满返回 False。"""
        if not self._running:
            return False
        try:
            self._input_queue.put_nowait((frame.copy(), frame_id, person_tracks))
            return True
        except Full:
            self._skip_count += 1
            return False

    def poll_results(self) -> bool:
        """收割最新结果 (非阻塞)。返回 True 表示有新结果。"""
        has_new = False
        while True:
            try:
                self._output_queue.get_nowait()
                self._processed_count += 1
                has_new = True
            except Empty:
                break
        return has_new

    # ==================================================================
    # Worker 线程
    # ==================================================================

    def run(self):
        while self._running:
            try:
                frame, frame_id, person_tracks = self._input_queue.get(timeout=0.5)
            except Empty:
                continue
            except Exception:
                continue

            try:
                self._process(frame, frame_id, person_tracks)
            except Exception:
                logger.debug("FallDetectionWorker inference error", exc_info=True)

    def _process(self, frame, frame_id, person_tracks):
        if frame_id % self._interval != 0:
            return

        # 1. 推理
        detections = self._backend.infer(frame)
        if not detections:
            return

        # 2. 构建检测列表
        det_list = []
        for det in detections:
            kp_list = det.keypoints
            if len(kp_list) < 17:
                continue
            kp_array = np.array([[kp.x, kp.y] for kp in kp_list], dtype=np.float32)
            confs = np.array([kp.confidence for kp in kp_list], dtype=np.float32)
            det_list.append({"bbox": det.bbox, "keypoints": kp_array.tolist(), "confs": confs})

        if not det_list:
            return

        # 心跳
        if frame_id % 30 == 0:
            logger.info(f"[FallWorker] frame=#{frame_id} detected={len(det_list)} "
                        f"tracks={len(self._fall_tracks)}")

        # 3. body-to-body IoU 匹配
        matched_ftids = set()
        unmatched_dets = list(range(len(det_list)))

        if self._fall_tracks:
            ftids = list(self._fall_tracks.keys())
            iou_matrix = np.zeros((len(ftids), len(det_list)))
            for i, ftid in enumerate(ftids):
                tb = self._fall_tracks[ftid]["bbox"]
                for j, d in enumerate(det_list):
                    iou_matrix[i, j] = self._iou(tb, d["bbox"])

            while True:
                best = None
                max_iou = 0.3
                for i in range(len(ftids)):
                    for j in unmatched_dets:
                        if iou_matrix[i, j] > max_iou:
                            max_iou = iou_matrix[i, j]
                            best = (i, j, ftids[i])
                if best is None:
                    break
                i, j, ftid = best
                det = det_list[j]
                self._fall_tracks[ftid]["bbox"] = det["bbox"]
                self._fall_tracks[ftid]["keypoints"] = det["keypoints"]
                self._fall_tracks[ftid]["confs"] = det["confs"]
                self._fall_tracks[ftid]["last_seen"] = frame_id
                matched_ftids.add(ftid)
                ftids.pop(i)
                unmatched_dets.remove(j)

        # 新 track (上限 10 人, 防止虚假检测膨胀)
        for j in unmatched_dets:
            if len(self._fall_tracks) >= 10:
                break
            det = det_list[j]
            ftid = self._next_fall_tid
            self._next_fall_tid += 1
            self._fall_tracks[ftid] = {
                "bbox": det["bbox"],
                "keypoints": det["keypoints"],
                "confs": det["confs"],
                "history": [],
                "fall_state": self._new_fall_state(),
                "last_seen": frame_id,
                "person_tid": None,
            }

        # 清理丢失 track
        stale = [ftid for ftid, t in self._fall_tracks.items()
                 if frame_id - t["last_seen"] > self._track_max_lost]
        for ftid in stale:
            del self._fall_tracks[ftid]

        # 4. 每个 track 运行 evaluate_fall
        from plugins.fall_engine.features import yolo_to_5keypoints, get_torso_inclination
        from plugins.fall_engine.fall_logic import evaluate_fall

        new_results = {}
        for ftid, ft in self._fall_tracks.items():
            kp_array = np.array(ft["keypoints"], dtype=np.float32)
            confs = np.array(ft.get("confs", [0.5] * 17), dtype=np.float32)

            kp_5 = yolo_to_5keypoints(kp_array, confs)
            if kp_5 is None:
                continue

            # 追加 history
            ft["history"].append(kp_5)
            if len(ft["history"]) > 20:
                ft["history"] = ft["history"][-20:]

            angle_kps = {
                "shoulder": ((kp_array[5][0] + kp_array[6][0]) / 2,
                             (kp_array[5][1] + kp_array[6][1]) / 2),
                "hip": ((kp_array[11][0] + kp_array[12][0]) / 2,
                        (kp_array[11][1] + kp_array[12][1]) / 2),
                "knee": ((kp_array[13][0] + kp_array[14][0]) / 2,
                         (kp_array[13][1] + kp_array[14][1]) / 2),
            }
            torso_inclination = get_torso_inclination(kp_5)

            x1, y1, x2, y2 = ft["bbox"]
            w, h = x2 - x1, y2 - y1
            aspect_ratio = h / w if w > 0 else 1.0

            person_data = {
                "pid": ftid, "kp_5": kp_5, "aspect_ratio": aspect_ratio,
                "bbox": ft["bbox"], "fall_state": ft["fall_state"],
                "history": ft["history"], "keypoints": kp_array,
                "confs": confs, "angle_keypoints": angle_kps,
                "torso_inclination": torso_inclination,
            }

            try:
                result = evaluate_fall(person_data, time.time())
            except Exception:
                continue

            state = result.get("state", "Normal")
            person_tid = self._match_person(ft["bbox"], person_tracks)
            ft["person_tid"] = person_tid

            new_results[ftid] = {
                "keypoints": ft["keypoints"],
                "fall_state": state,
                "confidence": result.get("confidence", 0.0),
                "bbox": ft["bbox"],
                "person_tid": person_tid,
            }

            if result.get("fall_detected"):
                if ft["fall_state"].get("_fall_logged") != True:
                    ft["fall_state"]["_fall_logged"] = True
                    logger.info(f"[FallWorker] FALL DETECTED ftid={ftid} person={person_tid} "
                                f"conf={result.get('confidence', 0):.2f}")
            else:
                ft["fall_state"]["_fall_logged"] = False

        self.last_results = new_results

        # 通知主线程有新结果
        try:
            self._output_queue.put_nowait(True)
        except Full:
            pass

    # ==================================================================
    # 辅助方法
    # ==================================================================

    @staticmethod
    def _new_fall_state() -> dict:
        return {
            "is_potential_fall": False, "fall_start_time": None,
            "fall_detected": False, "trigger_history": [],
            "consecutive_triggers": 0, "trigger_gap_count": 0,
        }

    @staticmethod
    def _iou(boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-8)

    @staticmethod
    def _match_person(body_bbox, person_tracks):
        bx, by = (body_bbox[0] + body_bbox[2]) / 2, (body_bbox[1] + body_bbox[3]) / 2
        best_tid, best_dist = None, float("inf")
        for entry in person_tracks:
            if len(entry) < 2:
                continue
            tid, tbbox = entry[0], entry[1]
            tx, ty = (tbbox[0] + tbbox[2]) / 2, (tbbox[1] + tbbox[3]) / 2
            dist = ((bx - tx) ** 2 + (by - ty) ** 2) ** 0.5
            if dist < best_dist and dist < 150:
                best_dist, best_tid = dist, tid
        return best_tid

    # ==================================================================
    # 生命周期
    # ==================================================================

    def stop(self):
        self._running = False
        try:
            self.join(timeout=3.0)
        except Exception:
            pass

    @property
    def queue_size(self):
        return self._input_queue.qsize()

    @property
    def processed_count(self):
        return self._processed_count

    @property
    def skip_count(self):
        return self._skip_count
