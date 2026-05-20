"""
FallDetectionProcess — multiprocess fall detection worker.

Runs YOLO-Pose inference in a dedicated child process with IoU matching,
ghost tracking, and ``evaluate_fall()`` from the existing fall-engine
logic.  Subclass of ``BaseInferenceProcess``.

Lifecycle (child process):
    init_model() -> _warmup() -> cmd_loop (read cmd_queue, dispatch) -> _child_cleanup()

Meta fields expected for each ``submit`` command:
    frame_id    — frame sequence number (int)
    tracks      — list of active person tracks [(tid, bbox, identity), ...]
    frame_h     — frame height (int, default 480)
    frame_w     — frame width (int, default 640)

Result dict returned per frame:
    {
        "fall_events": dict[ftid, result_dict],
        "frame_id": int,
        "skipped": bool,   # True when frame was subsampled
    }
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from core.workers.base_process import BaseInferenceProcess
from plugins.fall_engine.features import yolo_to_5keypoints, get_torso_inclination
from plugins.fall_engine.fall_logic import evaluate_fall
from utils.geometry import compute_iou

logger = logging.getLogger(__name__)


class FallDetectionProcess(BaseInferenceProcess):
    """Multiprocess fall detection worker.

    Runs YOLO-Pose inference in a dedicated child process, manages internal
    fall tracks with IoU matching and ghost tracking, and evaluates fall
    state via ``evaluate_fall()``.

    Parameters
    ----------
    model_path:
        Path to the YOLO-Pose model file.  Passed to ``create_backend()``.
    device:
        Inference device (``"cpu"`` or ``"cuda"``).
    backend:
        Backend name (``"ultralytics"``, ``"onnx"``, ``"ncnn"``).
    conf:
        Detection confidence threshold.
    input_size:
        Model input resolution.
    interval:
        Process every Nth frame (subsampling).
    ghost_timeout:
        Seconds before a ghost track (non-fallen) is cleaned up.
    ghost_timeout_fallen:
        Seconds before a ghost track (fallen) is cleaned up.
    name:
        Process name (used for logging and SHM naming).
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        backend: str = "ultralytics",
        conf: float = 0.5,
        input_size: int = 640,
        interval: int = 15,
        ghost_timeout: float = 3.0,
        ghost_timeout_fallen: float = 30.0,
        name: str = "fall_detection",
    ) -> None:
        super().__init__(name=name)
        self._model_path = model_path
        self._device = device
        self._backend_name = backend
        self._conf = conf
        self._input_size = input_size
        self._interval = interval
        self._ghost_timeout = ghost_timeout
        self._ghost_timeout_fallen = ghost_timeout_fallen

        # State lives in the child process — initialised in init_model().
        self._fall_tracks: dict[int, dict] = {}
        self._next_fall_tid: int = 1000
        self._track_max_lost: int = 30

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def init_model(self) -> None:
        """Load YOLO-Pose backend in the child process via create_backend()."""
        from plugins.fall_engine.backends import create_backend

        self._backend = create_backend(
            backend=self._backend_name,
            model_path=self._model_path,
            device=self._device,
            conf=self._conf,
            input_size=self._input_size,
        )

        # Tracking state (reset on restart)
        self._fall_tracks = {}
        self._next_fall_tid = 1000
        logger.info("[FallDetectionProcess] init_model complete")

    def process_frame(self, slot_idx: int, meta: dict) -> dict[str, Any]:
        """Process a frame: run YOLO-Pose, match tracks, evaluate fall.

        Args:
            slot_idx: Ring-buffer slot index.
            meta: Must contain ``"frame_id"`` and optionally ``"tracks"``,
                  ``"frame_h"``, ``"frame_w"``.

        Returns:
            Dict with keys ``fall_events``, ``frame_id``, ``skipped``.
        """
        frame_id = meta.get("frame_id", 0)
        person_tracks = meta.get("tracks", [])
        frame_h = meta.get("frame_h", 480)
        frame_w = meta.get("frame_w", 640)

        # -- Subsampling --
        if frame_id % self._interval != 0:
            return {"fall_events": {}, "frame_id": frame_id, "skipped": True}

        # Heartbeat
        if frame_id % 30 == 0:
            logger.info(
                "[FallProcess] frame=#%d tracks=%d",
                frame_id, len(self._fall_tracks),
            )

        # -- Read frame from SharedMemory --
        offset = slot_idx * self.SLOT_SIZE
        nbytes = frame_h * frame_w * 3
        frame = np.frombuffer(
            self._input_shm.buf,
            dtype=np.uint8,
            count=nbytes,
            offset=offset,
        ).reshape(frame_h, frame_w, 3).copy()

        # 1. Run YOLO-Pose inference ---------------------------------------
        detections = self._backend.infer(frame)

        # 2. Build detection list ------------------------------------------
        det_list: list[dict] = []
        if detections:
            for det in detections:
                kp_list = det.keypoints
                if len(kp_list) < 17:
                    continue
                kp_array = np.array(
                    [[kp.x, kp.y] for kp in kp_list], dtype=np.float32
                )
                confs = np.array(
                    [kp.confidence for kp in kp_list], dtype=np.float32
                )
                det_list.append({
                    "bbox": det.bbox,
                    "keypoints": kp_array.tolist(),
                    "confs": confs,
                })

        # 3. IoU matching (same logic as FallDetectionWorker) --------------
        matched_ftids: set[int] = set()
        unmatched_dets = list(range(len(det_list)))

        if det_list and self._fall_tracks:
            ftids = list(self._fall_tracks.keys())
            iou_matrix = np.zeros((len(ftids), len(det_list)))
            for i, ftid in enumerate(ftids):
                tb = self._fall_tracks[ftid]["bbox"]
                for j, d in enumerate(det_list):
                    iou_matrix[i, j] = compute_iou(tb, d["bbox"])

            while True:
                best: tuple | None = None
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

        # 4. Ghost inheritance: new detections inherit fall_state ----------
        for j in unmatched_dets:
            if len(self._fall_tracks) >= 10:
                break
            det = det_list[j]
            ftid = self._next_fall_tid
            self._next_fall_tid += 1
            inherited_fs = self._find_ghost_fall_state(det["bbox"])
            fall_state = (
                inherited_fs
                if inherited_fs is not None
                else self._new_fall_state()
            )
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

        # 5. Mark unmatched existing tracks as ghost -----------------------
        for ftid, ft in self._fall_tracks.items():
            if ftid not in matched_ftids:
                fs = ft.get("fall_state", {})
                if fs.get("is_potential_fall") or fs.get("fall_detected"):
                    ft["is_ghost"] = True
                    if "ghost_start_time" not in ft:
                        ft["ghost_start_time"] = time.time()

        # 6. Clean up expired ghosts ---------------------------------------
        self._cleanup_ghosts()

        # 7. Evaluate fall for all non-ghost tracks ------------------------
        new_results: dict[int, dict] = {}
        for ftid, ft in self._fall_tracks.items():
            if ft.get("is_ghost", False):
                continue

            kp_array = np.array(ft["keypoints"], dtype=np.float32)
            confs_arr = np.array(
                ft.get("confs", [0.5] * 17), dtype=np.float32
            )

            kp_5 = yolo_to_5keypoints(kp_array, confs_arr)
            if kp_5 is None:
                continue

            ft["history"].append(kp_5)
            if len(ft["history"]) > 20:
                ft["history"] = ft["history"][-20:]

            angle_kps = {
                "shoulder": (
                    (kp_array[5][0] + kp_array[6][0]) / 2,
                    (kp_array[5][1] + kp_array[6][1]) / 2,
                ),
                "hip": (
                    (kp_array[11][0] + kp_array[12][0]) / 2,
                    (kp_array[11][1] + kp_array[12][1]) / 2,
                ),
                "knee": (
                    (kp_array[13][0] + kp_array[14][0]) / 2,
                    (kp_array[13][1] + kp_array[14][1]) / 2,
                ),
            }
            torso_inclination = get_torso_inclination(kp_5)

            x1, y1, x2, y2 = ft["bbox"]
            w_val, h_val = x2 - x1, y2 - y1
            aspect_ratio = h_val / w_val if w_val > 0 else 1.0

            person_data = {
                "pid": ftid,
                "kp_5": kp_5,
                "aspect_ratio": aspect_ratio,
                "bbox": ft["bbox"],
                "fall_state": ft["fall_state"],
                "history": ft["history"],
                "keypoints": kp_array,
                "confs": confs_arr,
                "angle_keypoints": angle_kps,
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
                if ft["fall_state"].get("_fall_logged") is not True:
                    ft["fall_state"]["_fall_logged"] = True
                    logger.info(
                        "[FallProcess] FALL DETECTED ftid=%d person=%s "
                        "conf=%.2f",
                        ftid, person_tid, result.get("confidence", 0.0),
                    )
            else:
                ft["fall_state"]["_fall_logged"] = False

        return {
            "fall_events": new_results,
            "frame_id": frame_id,
            "skipped": False,
        }

    # ------------------------------------------------------------------
    # Ghost management (same logic as FallDetectionWorker)
    # ------------------------------------------------------------------

    def _find_ghost_fall_state(self, new_bbox) -> dict | None:
        """Find a nearby ghost track whose ``fall_state`` should be inherited.

        Returns the inherited ``fall_state`` dict, or ``None``.
        The source ghost track is deleted after inheritance.
        """
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
        """Remove ghost tracks that have exceeded their timeout."""
        now = time.time()
        stale: list[int] = []
        for ftid, ft in self._fall_tracks.items():
            if not ft.get("is_ghost", False):
                continue
            gs = ft.get("ghost_start_time", now)
            timeout = (
                self._ghost_timeout_fallen
                if ft["fall_state"].get("fall_detected")
                else self._ghost_timeout
            )
            if now - gs > timeout:
                stale.append(ftid)
        for ftid in stale:
            del self._fall_tracks[ftid]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _new_fall_state() -> dict:
        return {
            "is_potential_fall": False,
            "fall_start_time": None,
            "fall_detected": False,
            "trigger_history": [],
            "consecutive_triggers": 0,
            "trigger_gap_count": 0,
        }

    @staticmethod
    def _match_person(body_bbox, person_tracks):
        """Find the nearest person track within 150 px distance."""
        bx = (body_bbox[0] + body_bbox[2]) / 2
        by = (body_bbox[1] + body_bbox[3]) / 2
        best_tid: int | None = None
        best_dist = float("inf")
        for entry in person_tracks:
            if len(entry) < 2:
                continue
            tid, tbbox = entry[0], entry[1]
            tx = (tbbox[0] + tbbox[2]) / 2
            ty = (tbbox[1] + tbbox[3]) / 2
            dist = ((bx - tx) ** 2 + (by - ty) ** 2) ** 0.5
            if dist < best_dist and dist < 150:
                best_dist, best_tid = dist, tid
        return best_tid
