"""
Vision AI — SCRFD + ArcFace + LightweightTracker 边缘人脸识别系统 v9。

Pipeline: Camera → MotionGate → FrameScheduler → SCRFD → Tracker → PersonManager
             → RecognitionScheduler → RecognitionWorker → Renderer

用法:
  python main.py
  python main.py --config configs/edge_minimal.yaml
  python main.py --profile edge_minimal
  python main.py --device cpu
  python main.py --no-render
"""

import argparse
import logging
import os
import sys

os.environ["ORT_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

from core.camera import Camera
from core.detectors import SCRFDDetector
from core.frame_scheduler import FrameScheduler
from core.recognition import FaceRecognizer
from core.rendering import Renderer
from core.pipeline import Pipeline
from core.person import PersonManager
from core.scheduler import RecognitionScheduler
from core.workers import RecognitionWorker
from database import FaceDatabase
from utils import FPS, load_config, get_project_root, PerformanceMonitor
from utils.motion_gate import MotionGate


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_camera(config, args):
    cfg = config.get("camera", {})
    cam_arg = args.camera
    if isinstance(cam_arg, list):
        cam_arg = cam_arg[-1] if cam_arg else None
    if cam_arg is not None:
        try:
            index = int(cam_arg)
        except (ValueError, TypeError):
            index = cfg.get("index", 0)
    else:
        index = cfg.get("index", 0)
    return Camera(
        index=index,
        width=cfg.get("width", 640),
        height=cfg.get("height", 480),
    )


def build_detector(config, args, use_gpu):
    cfg = config.get("detector", {})
    return SCRFDDetector(
        model_name=cfg.get("model_name", "buffalo_s"),
        input_size=cfg.get("input_size", 640),
        conf_threshold=cfg.get("conf_threshold", 0.5),
        nms_threshold=cfg.get("nms_threshold", 0.4),
        device="cuda" if use_gpu else "cpu",
    )


def build_tracker(config, args):
    cfg = config.get("tracking", {})
    tracker_type = cfg.get("type", cfg.get("tracker", "iou")).lower()

    if tracker_type in ("iou", "lightweight_iou", "lightweight"):
        from core.tracking import LightweightIoUTracker
        return LightweightIoUTracker(
            iou_threshold=cfg.get("iou_threshold", 0.3),
            max_lost=cfg.get("max_lost", 15),
            min_hits=cfg.get("min_hits", 2),
        )

    if tracker_type in ("bytetrack", "byte_track"):
        from core.tracking import MultiObjectTracker
        if MultiObjectTracker is None:
            raise RuntimeError(
                "ByteTrack requested but boxmot is not installed. "
                "Install requirements/desktop.txt or set tracking.type=iou."
            )
        return MultiObjectTracker(
            track_thresh=cfg.get("track_thresh", 0.5),
            track_buffer=cfg.get("max_disappeared", 30),
            match_thresh=cfg.get("match_thresh", 0.8),
            frame_rate=cfg.get("frame_rate", 30),
        )

    raise ValueError(f"Unknown tracking.type: {tracker_type}")


def build_recognizer(config, args):
    cfg = config.get("recognition", {})
    model_name = args.model or cfg.get("model_name", "buffalo_s")
    return FaceRecognizer(
        model_name=model_name,
        device=args.device or cfg.get("device"),
    )


def build_database(config, args):
    cfg = config.get("database", {})
    project_root = get_project_root()
    db_path = args.db or cfg.get("path", "face_db/identities.pkl")
    db_path = os.path.join(project_root, db_path)
    database = FaceDatabase()
    database.load(db_path)
    return database


def build_scheduler(config, args):
    cfg = config.get("recognition", {})
    cooldown = args.cooldown or cfg.get("recognition_cooldown", 300)
    return RecognitionScheduler(
        cooldown=cooldown,
        recognized_cooldown=cfg.get("recognized_cooldown", 600),
        failed_backoff=cfg.get("failed_backoff", 90),
        max_attempts=cfg.get("max_attempts", 20),
    )


def build_worker(config, args, recognizer, database):
    cfg = config.get("recognition", {})
    return RecognitionWorker(
        recognizer=recognizer,
        database=database,
        max_queue_size=cfg.get("max_queue_size", 4),
    )


def build_quality_filter(config):
    cfg = config.get("recognition", {})
    from core.face_quality import FaceQualityFilter
    return FaceQualityFilter(
        min_face_size=cfg.get("min_face_size", 48),
        blur_threshold=cfg.get("blur_threshold", 80),
        min_quality_score=cfg.get("min_quality_score", 0.55),
    )


def build_tasks(config):
    """根据配置加载启用的 VisionTask 插件。"""
    tasks = []
    tasks_cfg = config.get("tasks", {})

    # Fall Detection
    fd_cfg = tasks_cfg.get("fall_detection", {})
    if fd_cfg.get("enabled", False):
        from plugins.fall_detection import FallDetectionTask
        tasks.append(FallDetectionTask(fd_cfg))

    return tasks


def build_motion_gate(config):
    cfg = config.get("motion", {})
    return MotionGate(
        threshold=cfg.get("threshold", 2.0),
        history=cfg.get("history", 0),
    )


def build_frame_scheduler(config, args):
    detector_cfg = config.get("detector", {})
    motion_cfg = config.get("motion", {})
    detect_interval = args.detect_interval or detector_cfg.get("detection_interval", 2)
    return FrameScheduler(
        detection_interval=detect_interval,
        force_interval=motion_cfg.get("force_interval", 15),
    )


def build_optional_modules(config):
    """
    根据 runtime 配置决定是否加载可选重模块。
    返回 dict: {module_name: instance_or_None}
    """
    runtime_cfg = config.get("runtime", {})
    modules = {}

    # TrackReassociation — dead code in pipeline, never called
    if runtime_cfg.get("enable_reassociation", False):
        from core.track_reassociation import TrackReassociation
        modules["track_reassociation"] = TrackReassociation(
            iou_threshold=0.3, distance_threshold=100,
        )
    else:
        modules["track_reassociation"] = None

    # Behavior layer
    if runtime_cfg.get("enable_behavior", False):
        from core.trajectory_analyzer import TrajectoryAnalyzer
        from core.behavior_engine import BehaviorEngine
        behavior_cfg = config.get("behavior", {})
        trajectory_analyzer = TrajectoryAnalyzer(
            stationary_threshold=behavior_cfg.get("stationary_threshold", 60),
        )
        modules["trajectory_analyzer"] = trajectory_analyzer
        modules["behavior_engine"] = BehaviorEngine(
            trajectory_analyzer=trajectory_analyzer,
            region_manager=None,
            stationary_threshold=behavior_cfg.get("stationary_threshold", 60),
            loitering_threshold=behavior_cfg.get("loitering_threshold", 300),
        )
    else:
        modules["trajectory_analyzer"] = None
        modules["behavior_engine"] = None

    # Region
    if runtime_cfg.get("enable_region", False):
        from core.region_manager import RegionManager
        region_manager = RegionManager()
        regions_cfg = config.get("regions", {})
        for zone_type, zones in regions_cfg.items():
            if isinstance(zones, list):
                for zone in zones:
                    region_manager.add_zone(zone_type, zone.get("name", zone_type), zone.get("points", []))
        modules["region_manager"] = region_manager
    else:
        modules["region_manager"] = None

    # Event system
    if runtime_cfg.get("enable_event_system", False):
        from core.event_system import EventSystem
        modules["event_system"] = EventSystem()
    else:
        modules["event_system"] = None

    # Alert
    if runtime_cfg.get("enable_alert", False):
        from core.alert_manager import AlertManager
        behavior_cfg = config.get("behavior", {})
        modules["alert_manager"] = AlertManager(
            cooldown_seconds=behavior_cfg.get("alert_cooldown", 30),
        )
    else:
        modules["alert_manager"] = None

    return modules


def _deep_merge(base: dict, overlay: dict) -> None:
    """递归将 overlay 合并到 base (修改 base)。"""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vision AI — Edge Face Recognition")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None,
                        choices=["edge_minimal", "balanced", "desktop"])
    parser.add_argument("--camera", type=str, action="append", default=None,
                        help="摄像头: --camera 0 --camera 1 或 --camera rtsp://...")
    parser.add_argument("--multi-camera", type=str, default=None,
                        help="多摄像头 YAML 配置文件路径")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--cooldown", type=int, default=None)
    parser.add_argument("--detect-interval", type=int, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--benchmark", action="store_true",
                        help="benchmark模式: 等效 --no-render --max-frames 300")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多运行帧数 (0=无限, 用于 benchmark)")
    args = parser.parse_args()

    # --- Config loading: default → profile → explicit config → CLI args ---
    config = load_config(None)  # default.yaml as base

    if args.profile:
        profile_path = f"configs/{args.profile}.yaml"
        if os.path.exists(profile_path):
            profile_cfg = load_config(profile_path)
            _deep_merge(config, profile_cfg)

    if args.config:
        explicit_cfg = load_config(args.config)
        _deep_merge(config, explicit_cfg)

    # CLI overrides
    if args.benchmark:
        args.no_render = True
        if args.max_frames == 0:
            args.max_frames = 300

    # --- Core config sections ---
    runtime_cfg = config.get("runtime", {})
    pipeline_cfg = config.get("pipeline", {})
    detector_cfg = config.get("detector", {})

    device = args.device or detector_cfg.get("device") or "cuda"
    use_gpu = runtime_cfg.get("use_gpu", True) and device != "cpu"

    # --- Build core modules ---
    camera = build_camera(config, args)
    detector = build_detector(config, args, use_gpu)
    tracker = build_tracker(config, args)
    recognizer = build_recognizer(config, args)
    database = build_database(config, args)
    recog_scheduler = build_scheduler(config, args)
    worker = build_worker(config, args, recognizer, database)
    recog_cfg = config.get("recognition", {})
    person_manager = PersonManager(
        max_age=1.0,
        embedding_cache_ttl=recog_cfg.get("embedding_cache_ttl", 30),
        max_cache_size=recog_cfg.get("max_cache_size", 128),
    )
    motion_gate = build_motion_gate(config)
    frame_scheduler = build_frame_scheduler(config, args)
    renderer = Renderer(font_scale=0.6, thickness=2)
    fps = FPS(window_size=pipeline_cfg.get("fps_window_size", 30))
    monitor = PerformanceMonitor()

    # --- Build optional modules ---
    opt = build_optional_modules(config)
    quality_filter = build_quality_filter(config)

    # --- Build vision tasks (plugins) ---
    tasks = build_tasks(config)

    # --- Banner ---
    mode = runtime_cfg.get("mode", "edge_minimal")
    print("=" * 60)
    print(f"  Vision AI v9 | Profile: {mode} | {'CUDA' if use_gpu else 'CPU'}")
    print(f"  Detector: SCRFD ({detector_cfg.get('model_name','buffalo_s')}) @ {detector.input_size}px")
    print(f"  Motion: {motion_gate.threshold} | Detect: {frame_scheduler.detection_interval}f")
    print(f"  Behavior: {'ON' if opt['behavior_engine'] else 'OFF'}")
    print("=" * 60)

    # --- Multi-Camera Mode ---
    if args.multi_camera or (args.camera and len(args.camera) > 1):
        _run_multi_camera(config, args, use_gpu)
        return

    # --- Pipeline ---
    pipeline = Pipeline(
        camera=camera,
        detector=detector,
        renderer=renderer,
        fps=fps,
        tracker=tracker,
        person_manager=person_manager,
        recog_scheduler=recog_scheduler,
        worker=worker,
        monitor=monitor,
        motion_gate=motion_gate,
        frame_scheduler=frame_scheduler,
        quality_filter=quality_filter,
        track_reassociation=opt.get("track_reassociation"),
        trajectory_analyzer=opt.get("trajectory_analyzer"),
        behavior_engine=opt.get("behavior_engine"),
        region_manager=opt.get("region_manager"),
        event_system=opt.get("event_system"),
        alert_manager=opt.get("alert_manager"),
        window_name=pipeline_cfg.get("window_name", "Vision AI"),
        quit_key=pipeline_cfg.get("quit_key", "q"),
        no_render=args.no_render or pipeline_cfg.get("no_render", False),
        max_frames=args.max_frames or pipeline_cfg.get("max_frames", 0),
        queue_pressure_threshold=recog_cfg.get("queue_pressure_threshold", 3),
        tasks=tasks,
    )

    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n[System] 用户中断")

    print("[System] 程序已退出")


# ---------------------------------------------------------------------------
# Multi-Camera Runner
# ---------------------------------------------------------------------------

def _run_multi_camera(config, args, use_gpu):
    """多摄像头模式。"""
    from core.multi_camera_manager import MultiCameraManager
    from core.scheduler.global_inference_scheduler import GlobalInferenceScheduler

    recognizer = build_recognizer(config, args)
    database = build_database(config, args)
    worker = build_worker(config, args, recognizer, database)
    quality_filter = build_quality_filter(config)
    motion_gate = build_motion_gate(config)

    cameras_cfg = []
    if args.multi_camera:
        import yaml
        with open(args.multi_camera) as f:
            mc = yaml.safe_load(f)
        cameras_cfg = mc.get("cameras", [])
    elif args.camera and len(args.camera) > 1:
        for i, src in enumerate(args.camera):
            cameras_cfg.append({"id": f"cam{i}", "source": src, "profile": args.profile or "edge_minimal"})

    if not cameras_cfg:
        print("[System] no cameras configured")
        return

    sched_cfg = config.get("scheduler", {})
    max_concurrent = sched_cfg.get("max_concurrent_detect", 2)
    inference_scheduler = GlobalInferenceScheduler(max_concurrent)

    manager = MultiCameraManager(
        shared_worker=worker,
        shared_database=database,
        inference_scheduler=inference_scheduler,
    )

    recog_cfg = config.get("recognition", {})
    for cam_cfg in cameras_cfg:
        cid = cam_cfg["id"]
        src = cam_cfg["source"]
        try: src = int(src)
        except (ValueError, TypeError): pass

        profile = cam_cfg.get("profile", "edge_minimal")
        cam_config = dict(config)
        if profile != config.get("runtime", {}).get("mode"):
            profile_path = f"configs/{profile}.yaml"
            if os.path.exists(profile_path):
                pc = load_config(profile_path)
                _deep_merge(cam_config, pc)

        cam_detector = build_detector(cam_config, args, use_gpu)
        cam_tracker = build_tracker(cam_config, args)
        cam_scheduler = RecognitionScheduler(
            cooldown=recog_cfg.get("recognition_cooldown", 300),
            recognized_cooldown=recog_cfg.get("recognized_cooldown", 600),
            failed_backoff=recog_cfg.get("failed_backoff", 90),
        )
        cam_frame_sched = FrameScheduler(
            detection_interval=cam_config.get("detector", {}).get("detection_interval", 4),
        )

        manager.add_camera(
            camera_id=cid, source=src,
            detector=cam_detector, tracker=cam_tracker,
            recog_scheduler=cam_scheduler,
            quality_filter=quality_filter,
            motion_gate=motion_gate,
            frame_scheduler=cam_frame_sched,
            renderer=Renderer(font_scale=0.5, thickness=1),
            render=not args.no_render and not args.benchmark,
        )

    worker.start()
    manager.start_all()
    try:
        manager.wait(max_frames=args.max_frames)
    except KeyboardInterrupt:
        print("\n[System] 用户中断")
    manager.stop_all()
    worker.stop()

    # per-camera summary
    for cid, p in manager.pipelines.items():
        if p.metrics:
            s = p.metrics.summary()
            logger.info(f"[{cid}] fps={s['fps']} det={s['detect_ms']}ms frames={s['frames']} drop={s['dropped']}")
    print("[System] 程序已退出")


if __name__ == "__main__":
    main()
