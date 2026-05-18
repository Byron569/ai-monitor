"""test_multi_camera.py — Multi Camera 架构验收 (no real camera)"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

PASS, FAIL = 0, 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; print(f"  [FAIL] {name}  -- {detail}")


def test_manager_create():
    print("\n--- MultiCameraManager ---")
    from core.multi_camera_manager import MultiCameraManager
    m = MultiCameraManager()
    check("create OK", m is not None)
    check("0 pipelines", len(m.pipelines) == 0)


def test_add_cameras():
    print("\n--- add cameras ---")
    from core.multi_camera_manager import MultiCameraManager
    from core.tracking.iou_tracker import LightweightIoUTracker
    from core.scheduler.recognition_scheduler import RecognitionScheduler

    class FakeDetector:
        def detect(self, frame): return []
    class FakeWorker:
        running = True; queue_size = 0
        def start(self): pass
        def stop(self): pass
        def submit(self, *a, **kw): return False
        def poll_results(self): return []

    m = MultiCameraManager(shared_worker=FakeWorker())
    m.add_camera("cam0", 0, FakeDetector(), LightweightIoUTracker(min_hits=1), RecognitionScheduler(300))
    m.add_camera("cam1", 1, FakeDetector(), LightweightIoUTracker(min_hits=1), RecognitionScheduler(300))
    check("2 pipelines", len(m.pipelines) == 2)


def test_tracker_isolation():
    print("\n--- tracker isolation ---")
    from core.tracking.iou_tracker import LightweightIoUTracker
    t1 = LightweightIoUTracker(min_hits=1)
    t2 = LightweightIoUTracker(min_hits=1)
    t1.update([(100, 100, 200, 200, 0.9)])
    check("t1 active 1", t1.active_count == 1)
    check("t2 active 0", t2.active_count == 0)


def test_inference_scheduler():
    print("\n--- GlobalInferenceScheduler ---")
    from core.scheduler.global_inference_scheduler import GlobalInferenceScheduler
    gs = GlobalInferenceScheduler(2)
    check("cam0 acquire", gs.acquire("cam0"))
    check("cam1 acquire", gs.acquire("cam1"))
    check("cam2 full", not gs.acquire("cam2"))
    check("active 2", gs.active_count == 2)
    gs.release("cam0")
    check("cam2 ok after release", gs.acquire("cam2"))


def test_camera_metrics():
    print("\n--- CameraMetrics ---")
    from core.metrics.camera_metrics import CameraMetrics
    m = CameraMetrics("test")
    m.record_frame()
    m.record_detect(15.0)
    m.record_detect(25.0)
    m.record_drop()
    s = m.summary()
    check("fps > 0", s["fps"] > 0)
    check("detect ~20ms", 19 < s["detect_ms"] < 21)
    check("dropped 1", s["dropped"] == 1)


def test_global_metrics():
    print("\n--- GlobalMetrics ---")
    from core.metrics.camera_metrics import CameraMetrics
    from core.metrics.global_metrics import GlobalMetrics
    gm = GlobalMetrics()
    m1 = CameraMetrics("cam0"); m1.record_frame()
    m2 = CameraMetrics("cam1"); m2.record_frame()
    gm.register("cam0", m1); gm.register("cam1", m2)
    check("2 cams", gm.total_cameras == 2)
    check("summary ok", "cam0" in gm.summary())


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [MULTI CAMERA TEST]")
    print("=" * 50)
    test_manager_create()
    test_add_cameras()
    test_tracker_isolation()
    test_inference_scheduler()
    test_camera_metrics()
    test_global_metrics()
    print(f"\n  ok={PASS} fail={FAIL}")
    return PASS, FAIL
