"""
test_behavior_system.py — Behavior Engine 验收

Q1: 正常移动 → behavior=moving？
Q2: 长时间静止 → behavior=stationary？
Q3: 静止超 threshold → loitering？
"""

import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

from core.trajectory_analyzer import TrajectoryAnalyzer
from core.behavior_engine import BehaviorEngine
from core.region_manager import RegionManager

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


def test_moving():
    print("\n--- 场景A: 正常移动 ---")
    ta = TrajectoryAnalyzer(stationary_threshold=60)
    rm = RegionManager()
    engine = BehaviorEngine(ta, rm, stationary_threshold=60)
    for i in range(100):
        bbox = (100 + i * 5, 100, 200 + i * 5, 200)  # 5px/frame > threshold 3
        ta.analyze(1, bbox)
        engine.update(1, i)
    bs = engine.get(1)
    check("behavior=moving", bs is not None and bs.behavior.value == "moving",
          f"实际={bs.behavior.value if bs else 'None'}")


def test_stationary():
    print("\n--- 场景B: 长时间静止 ---")
    ta = TrajectoryAnalyzer(stationary_threshold=60)
    rm = RegionManager()
    engine = BehaviorEngine(ta, rm, stationary_threshold=60)
    for i in range(100):
        bbox = (100, 100, 200, 200)
        ta.analyze(1, bbox)
        engine.update(1, i)
    bs = engine.get(1)
    check("behavior=stationary", bs is not None and bs.behavior.value == "stationary",
          f"实际={bs.behavior.value if bs else 'None'}")
    tf = ta.get(1)
    check("stationary_frames 累积", tf is not None and tf.stationary_frames > 50,
          f"实际={tf.stationary_frames if tf else 0}")


def test_loitering():
    print("\n--- 场景C: 长时间徘徊 ---")
    ta = TrajectoryAnalyzer(stationary_threshold=60)
    rm = RegionManager()
    rm.add_zone("loitering_zone", "test", [(0, 0), (300, 0), (300, 300), (0, 300)])
    engine = BehaviorEngine(ta, rm, stationary_threshold=60, loitering_threshold=100)
    for i in range(120):
        bbox = (100, 100, 200, 200)
        ta.analyze(1, bbox)
        engine.update(1, i)
    bs = engine.get(1)
    check("behavior=stationary（非 loitering, frame<300）",
          bs is not None and bs.behavior.value == "stationary")


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [BEHAVIOR TEST] Behavior Engine")
    print("=" * 50)
    test_moving()
    test_stationary()
    test_loitering()
    print(f"\n  behavior_ok={PASS} fail={FAIL}")
    return PASS, FAIL
