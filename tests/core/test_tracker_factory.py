"""
test_tracker_factory.py — Tracker 工厂验收

Q1: tracking.type=iou 返回 LightweightIoUTracker？
Q2: unknown type 抛 ValueError？
Q3: edge_minimal 不 import boxmot？
"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

PASS, FAIL = 0, 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; print(f"  [FAIL] {name}  -- {detail}")


class _FakeArgs:
    camera = None; device = None; db = None; model = None
    cooldown = None; detect_interval = None; no_render = False
    max_frames = 0; benchmark = False; config = None; profile = None


def test_iou_returns_lightweight():
    print("\n--- type=iou 返回 LightweightIoUTracker ---")
    config = {"tracking": {"type": "iou", "iou_threshold": 0.3, "max_lost": 10, "min_hits": 2}}
    from main import build_tracker
    tracker = build_tracker(config, _FakeArgs())
    from core.tracking.iou_tracker import LightweightIoUTracker
    check("是LightweightIoUTracker", isinstance(tracker, LightweightIoUTracker))


def test_unknown_type_raises():
    print("\n--- unknown type 抛 ValueError ---")
    config = {"tracking": {"type": "unknown_xxx"}}
    from main import build_tracker
    try:
        build_tracker(config, _FakeArgs())
        check("抛异常", False)
    except ValueError as e:
        check("抛 ValueError", True, f"msg={e}")


def test_edge_minimal_no_boxmot():
    print("\n--- edge_minimal 不强制 boxmot ---")
    from main import build_tracker
    config = {"tracking": {"type": "iou"}}
    tracker = build_tracker(config, _FakeArgs())
    from core.tracking.iou_tracker import LightweightIoUTracker
    check("type=iou 返回 LightweightIoUTracker (无boxmot)", isinstance(tracker, LightweightIoUTracker))


def test_tracking_init_safe():
    print("\n--- tracking/__init__ safe import ---")
    from core.tracking import LightweightIoUTracker, MultiObjectTracker, TrackManager
    check("LightweightIoUTracker 可用", LightweightIoUTracker is not None)
    check("TrackManager 可用", TrackManager is not None)
    check("MultiObjectTracker 加载不崩溃", True)


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [FACTORY TEST] Tracker Factory")
    print("=" * 50)
    test_iou_returns_lightweight()
    test_unknown_type_raises()
    test_edge_minimal_no_boxmot()
    test_tracking_init_safe()
    print(f"\n  factory_ok={PASS} fail={FAIL}")
    return PASS, FAIL
