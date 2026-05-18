"""
FallDetectionTask — 摔倒检测插件 (轻量包装)。

委托 FallDetectionWorker 做异步推理+追踪+判断。
主线程: submit(frame) → poll_results() → VisionEvent
"""

import logging
import time
from typing import List

from core.interfaces import VisionTask, VisionEvent
from plugins.fall_engine import set_config
from plugins.fall_engine.backends import create_backend
from core.workers.fall_detection_worker import FallDetectionWorker

logger = logging.getLogger(__name__)


class FallDetectionTask(VisionTask):
    """摔倒检测任务 (异步 Worker 架构)。

    配置路径: tasks.fall_detection
    """

    def __init__(self, config: dict = None):
        super().__init__()
        self.name = "fall_detection"
        cfg = config or {}

        self.enabled = cfg.get("enabled", False)
        self.interval = cfg.get("interval", 15)
        self._debug = cfg.get("debug", False)

        # 注入摔倒判断参数
        set_config(cfg)

        # 状态变化追踪 (仅状态变化时发事件)
        self._last_emitted: dict = {}  # {ftid: state}

        # 创建推理后端 + Worker
        self._worker = None
        if self.enabled:
            try:
                backend = create_backend(
                    backend=cfg.get("backend", "onnx"),
                    model_path=cfg.get("model_path", "models/yolov8n-pose.onnx"),
                    device=cfg.get("device", "cpu"),
                    conf=cfg.get("confidence_threshold", 0.5),
                    input_size=cfg.get("input_size", 640),
                    enable_tracking=False,
                )
                self._worker = FallDetectionWorker(
                    inference_backend=backend,
                    max_queue_size=2,
                    interval=self.interval,
                    confidence_threshold=cfg.get("confidence_threshold", 0.5),
                )
                self._worker.start()
                logger.info(f"[FallDetection] Worker 启动: backend={cfg.get('backend', 'onnx')} "
                            f"device={cfg.get('device', 'cpu')} interval={self.interval}")
            except Exception:
                logger.exception("[FallDetection] Worker 启动失败, 摔倒检测不可用")
                self.enabled = False

    @property
    def last_results(self):
        return self._worker.last_results if self._worker else {}

    def should_run(self, frame_id: int, tracks: list, context: dict) -> bool:
        if not self.enabled or self._worker is None:
            return False
        return True  # Worker 自己控制帧间隔

    def run(self, frame, tracks: list, context: dict) -> List[VisionEvent]:
        events = []
        frame_id = context.get("frame_count", 0)

        # 提交推理 (非阻塞)
        self._worker.submit(frame, frame_id, tracks)

        # 收割结果 (非阻塞)
        self._worker.poll_results()

        # 从 last_results 产出 VisionEvent (仅状态变化时)
        for ftid, result in self._worker.last_results.items():
            state = result.get("fall_state", "Normal")
            prev = self._last_emitted.get(ftid)
            if state == prev:
                continue  # 状态未变, 不重复发事件
            self._last_emitted[ftid] = state

            if state == "FALL" and result.get("confidence", 0) >= 0.3:
                person_tid = result.get("person_tid")
                logger.info(f"[FallDetection] FALL DETECTED ftid={ftid} person={person_tid} "
                            f"conf={result['confidence']:.2f}")
                events.append(VisionEvent(
                    event_type="fall_detected",
                    track_id=person_tid or ftid,
                    confidence=result["confidence"],
                    payload={"fall_state": "FALL", "bbox": result["bbox"], "keypoints": result["keypoints"]},
                ))
            elif state == "Potential Fall":
                events.append(VisionEvent(
                    event_type="fall_potential",
                    track_id=ftid,
                    confidence=result.get("confidence", 0.0),
                    payload={"fall_state": "potential_fall", "bbox": result["bbox"]},
                ))
            elif state == "Normal" and prev in ("Potential Fall", "FALL"):
                events.append(VisionEvent(
                    event_type="fall_recovered",
                    track_id=ftid,
                    confidence=0.0,
                    payload={"fall_state": "recovered", "bbox": result["bbox"]},
                ))

        return events

    def close(self):
        if self._worker:
            self._worker.stop()
