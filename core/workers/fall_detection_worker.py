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

from utils.geometry import compute_iou
from plugins.fall_engine.features import yolo_to_5keypoints, get_torso_inclination
from plugins.fall_engine.fall_logic import evaluate_fall

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
        ghost_timeout: float = 3.0,
        ghost_timeout_fallen: float = 30.0,
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

        # Ghost 目标机制：丢失 track 不立即删除，而是标记为 ghost 保留 fall_state
        self._ghost_timeout = ghost_timeout
        self._ghost_timeout_fallen = ghost_timeout_fallen

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

        # 心跳
        if frame_id % 30 == 0:
            logger.info(f"[FallWorker] frame=#{frame_id} "
                        f"tracks={len(self._fall_tracks)}")

        # 1. 推理
        detections = self._backend.infer(frame)

        # 2. 构建检测列表 + IoU 去重 (避免同一人被多次检测)
        det_list = []
        if detections:
            for det in detections:
                kp_list = det.keypoints
                if len(kp_list) < 17:
                    continue
                kp_array = np.array([[kp.x, kp.y] for kp in kp_list], dtype=np.float32)
                confs = np.array([kp.confidence for kp in kp_list], dtype=np.float32)
                det_list.append({"bbox": det.bbox, "keypoints": kp_array.tolist(), "confs": confs})

            # IoU dedup: if two detections overlap > 90%, keep the higher-confidence one
            deduped = []
            for d in sorted(det_list, key=lambda x: float(np.mean(x["confs"])), reverse=True):
                dup = False
                for m in deduped:
                    if compute_iou(d["bbox"], m["bbox"]) > 0.9:
                        dup = True
                        break
                if not dup:
                    deduped.append(d)
            det_list = deduped

        # 3. IoU 匹配 (含 ghost 继承)
        matched_ftids = set()
        unmatched_dets = list(range(len(det_list)))

        if det_list and self._fall_tracks:
            ftids = list(self._fall_tracks.keys())
            iou_matrix = np.zeros((len(ftids), len(det_list)))
            for i, ftid in enumerate(ftids):
                tb = self._fall_tracks[ftid]["bbox"]
                for j, d in enumerate(det_list):
                    iou_matrix[i, j] = compute_iou(tb, d["bbox"])

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
                self._fall_tracks[ftid]["is_ghost"] = False
                matched_ftids.add(ftid)
                ftids.pop(i)
                unmatched_dets.remove(j)

        # 4. Ghost 继承：新检测尝试从附近 ghost 继承 fall_state (ROI 替代方案)
        for j in unmatched_dets:
            if len(self._fall_tracks) >= 10:
                break
            det = det_list[j]
            ftid = self._next_fall_tid
            self._next_fall_tid += 1
            inherited_fs = self._find_ghost_fall_state(det["bbox"])
            fall_state = inherited_fs if inherited_fs is not None else self._new_fall_state()
            self._fall_tracks[ftid] = {
                "bbox": det["bbox"],
                "keypoints": det["keypoints"],
                "confs": det["confs"],
                "history": [],
                "fall_state": fall_state,
                "last_seen": frame_id,
                "person_tid": None,
                "is_ghost": False,
            }

        # 5. 将未匹配的现有 track 标记为 ghost
        for ftid, ft in self._fall_tracks.items():
            if ftid not in matched_ftids:
                fs = ft.get("fall_state", {})
                if fs.get("is_potential_fall") or fs.get("fall_detected"):
                    ft["is_ghost"] = True
                    if "ghost_start_time" not in ft:
                        ft["ghost_start_time"] = time.time()

        # 6. 清理过期 ghost
        self._cleanup_ghosts()

        # 7. 每个非 ghost track 运行 evaluate_fall
        self._evaluate_all_tracks(frame_id, person_tracks)

        # 通知主线程有新结果
        try:
            self._output_queue.put_nowait(True)
        except Full:
            pass

    def _evaluate_all_tracks(self, frame_id, person_tracks):
        """对所有非 ghost track 进行跌倒评估。"""
        new_results = {}
        for ftid, ft in self._fall_tracks.items():
            if ft.get("is_ghost", False):
                continue

            kp_array = np.array(ft["keypoints"], dtype=np.float32)
            confs = np.array(ft.get("confs", [0.5] * 17), dtype=np.float32)

            kp_5 = yolo_to_5keypoints(kp_array, confs)
            if kp_5 is None:
                continue

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

    # ------------------------------------------------------------------
    # Ghost 目标机制 (ROI 替代方案: 零额外推理)
    # ------------------------------------------------------------------

    def _find_ghost_fall_state(self, new_bbox):
        """从附近 ghost 继承 fall_state。成功继承后删除原 ghost。"""
        for ftid, ft in list(self._fall_tracks.items()):
            if not ft.get("is_ghost", False):
                continue
            fs = ft.get("fall_state", {})
            if not fs.get("is_potential_fall") and not fs.get("fall_detected"):
                continue
            if compute_iou(new_bbox, ft["bbox"]) > 0.2:
                inherited = dict(fs)
                del self._fall_tracks[ftid]
                return inherited
        return None

    def _cleanup_ghosts(self):
        """清理过期的 ghost track。"""
        now = time.time()
        stale = []
        for ftid, ft in self._fall_tracks.items():
            if not ft.get("is_ghost", False):
                continue
            gs = ft.get("ghost_start_time", now)
            timeout = self._ghost_timeout_fallen if ft["fall_state"].get(
                "fall_detected") else self._ghost_timeout
            if now - gs > timeout:
                stale.append(ftid)
        for ftid in stale:
            del self._fall_tracks[ftid]
            self.last_results.pop(ftid, None)

    def _process_ghosts_for_test(self, frame_id, person_tracks):
        """测试辅助：当无检测时，将有跌倒嫌疑的 track 转为 ghost。"""
        for ft in self._fall_tracks.values():
            fs = ft.get("fall_state", {})
            if fs.get("is_potential_fall") or fs.get("fall_detected"):
                ft["is_ghost"] = True
                if "ghost_start_time" not in ft:
                    ft["ghost_start_time"] = time.time()
        self._cleanup_ghosts()

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
