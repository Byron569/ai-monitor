"""
MultiCameraManager — 多摄像头管理器。

管理多个 CameraPipeline 线程，提供:
  - 统一 start/stop 生命周期
  - shared RecognitionWorker
  - shared FaceDatabase
  - GlobalInferenceScheduler
  - GlobalMetrics
"""

import logging
import threading
import time
from typing import Dict, List

from core.camera_pipeline import CameraPipeline
from core.person import PersonManager
from core.metrics.camera_metrics import CameraMetrics
from core.metrics.global_metrics import GlobalMetrics
from core.scheduler.global_inference_scheduler import GlobalInferenceScheduler

logger = logging.getLogger(__name__)


class MultiCameraManager:
    """
    多摄像头管理器。

    用法:
        manager = MultiCameraManager(
            shared_worker=worker,
            shared_database=database,
            inference_scheduler=GlobalInferenceScheduler(max_concurrent=2),
        )
        manager.add_camera("cam0", source=0, detector=det, tracker=trk, ...)
        manager.add_camera("cam1", source=1, ...)
        manager.start_all()
        manager.wait()  # or manager.join()
        manager.stop_all()
    """

    def __init__(
        self,
        shared_worker=None,
        shared_database=None,
        inference_scheduler=None,
        max_concurrent_detect: int = 2,
    ):
        self._pipelines: Dict[str, CameraPipeline] = {}
        self._shared_worker = shared_worker
        self._shared_database = shared_database
        self._inference_scheduler = inference_scheduler or GlobalInferenceScheduler(max_concurrent_detect)
        self._global_metrics = GlobalMetrics()
        self._running = False

    def add_camera(
        self,
        camera_id: str,
        source,
        detector,
        tracker,
        recog_scheduler,
        quality_filter=None,
        motion_gate=None,
        frame_scheduler=None,
        renderer=None,
        render: bool = False,
    ) -> CameraPipeline:
        """添加一个摄像头 Pipeline (不启动)。"""
        metrics = CameraMetrics(camera_id)
        self._global_metrics.register(camera_id, metrics)

        embed_cache_cfg = {"embedding_cache_ttl": 30, "max_cache_size": 128}
        person_manager = PersonManager(max_age=1.0, **embed_cache_cfg)

        pipeline = CameraPipeline(
            camera_id=camera_id,
            source=source,
            detector=detector,
            tracker=tracker,
            person_manager=person_manager,
            recog_scheduler=recog_scheduler,
            worker=self._shared_worker,
            quality_filter=quality_filter,
            motion_gate=motion_gate,
            frame_scheduler=frame_scheduler,
            renderer=renderer,
            render=render,
            inference_scheduler=self._inference_scheduler,
            database=self._shared_database,
            metrics=metrics,
        )
        self._pipelines[camera_id] = pipeline
        return pipeline

    def start_all(self):
        """启动所有摄像头。"""
        self._running = True
        for cid, pipeline in self._pipelines.items():
            pipeline.start()
            logger.info(f"[Manager] {cid} started")

    def stop_all(self):
        """停止所有摄像头。"""
        self._running = False
        for cid, pipeline in self._pipelines.items():
            pipeline.stop()
        for cid, pipeline in self._pipelines.items():
            pipeline.join(timeout=3.0)
        logger.info("[Manager] all cameras stopped")

    def wait(self, max_frames: int = 0):
        """阻塞等待。max_frames=0 时无限等待。"""
        try:
            while self._running:
                time.sleep(1.0)
                if max_frames > 0:
                    total = sum(p._frame_count for p in self._pipelines.values())
                    if total >= max_frames:
                        break
        except KeyboardInterrupt:
            self.stop_all()

    def join(self):
        """等待所有线程结束。"""
        for pipeline in self._pipelines.values():
            pipeline.join()

    @property
    def pipelines(self) -> Dict[str, CameraPipeline]:
        return dict(self._pipelines)

    @property
    def global_metrics(self) -> GlobalMetrics:
        return self._global_metrics

    @property
    def inference_scheduler(self) -> GlobalInferenceScheduler:
        return self._inference_scheduler
