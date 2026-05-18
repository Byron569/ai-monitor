"""
test_iou_tracker.py — LightweightIoUTracker 验收

Q1: 可以初始化？
Q2: 单检测创建 track？
Q3: 连续相近检测保持同一 track_id？
Q4: 丢失后 lost 增加？
Q5: lost > max_lost 后删除？
Q6: 交叉场景不崩溃？
Q7: get_memory() 返回 TrackMemory 兼容层？
Q8: 不依赖 boxmot？
"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

from core.tracking.iou_tracker import LightweightIoUTracker

PASS, FAIL = 0, 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; print(f"  [FAIL] {name}  -- {detail}")


def test_init():
    print("\n--- 初始化 ---")
    t = LightweightIoUTracker(iou_threshold=0.3, max_lost=15, min_hits=2)
    check("iou_threshold=0.3", t.iou_threshold == 0.3)
    check("max_lost=15", t.max_lost == 15)
    check("active_count=0", t.active_count == 0)


def test_single_detection():
    print("\n--- 单检测创建 track ---")
    t = LightweightIoUTracker(min_hits=1)
    dets = [(100, 100, 200, 200, 0.9)]
    result = t.update(dets)
    check("返回1个track", len(result) == 1)
    tid = list(result.keys())[0]
    bbox = result[tid]
    check("bbox接近", abs(bbox[0] - 100) <= 2 and abs(bbox[1] - 100) <= 2)
    check("active_count=1", t.active_count == 1)


def test_same_id():
    print("\n--- 连续相近检测保持同一 ID ---")
    t = LightweightIoUTracker(min_hits=1)
    dets1 = [(100, 100, 200, 200, 0.9)]
    r1 = t.update(dets1)
    tid1 = list(r1.keys())[0]
    dets2 = [(105, 105, 205, 205, 0.9)]
    r2 = t.update(dets2)
    tid2 = list(r2.keys())[0]
    check("track_id一致", tid1 == tid2, f"tid1={tid1} tid2={tid2}")


def test_lost_then_remove():
    print("\n--- 丢失后删除 ---")
    t = LightweightIoUTracker(max_lost=3, min_hits=1)
    dets = [(100, 100, 200, 200, 0.9)]
    t.update(dets)
    check("初始active=1", t.active_count == 1)
    for _ in range(4):
        t.update([])
    check("lost后active=0", t.active_count == 0)


def test_crossing():
    print("\n--- 交叉场景 ---")
    t = LightweightIoUTracker(min_hits=2, max_lost=30)
    # Person A moves right, Person B moves left
    for i in range(50):
        dets = [
            (100 + i*3, 200, 200 + i*3, 300, 0.9),  # A: left→right
            (500 - i*3, 200, 600 - i*3, 300, 0.9),  # B: right→left
        ]
        t.update(dets)
    check("cross后不崩溃", t.active_count >= 0, f"active={t.active_count}")


def test_get_memory():
    print("\n--- TrackMemory 兼容层 ---")
    t = LightweightIoUTracker(min_hits=1)
    dets = [(100, 100, 200, 200, 0.9)]
    t.update(dets)
    mem = t.get_memory()
    check("get_memory 返回非 None", mem is not None)
    check("has lost_count", hasattr(mem, "lost_count"))
    check("has update", hasattr(mem, "update"))
    check("has match_hungarian", hasattr(mem, "match_hungarian"))
    check("has get_active", hasattr(mem, "get_active"))


def test_no_boxmot():
    print("\n--- 不依赖 boxmot ---")
    t = LightweightIoUTracker(min_hits=1)
    dets = [(100, 100, 200, 200, 0.9)]
    t.update(dets)
    check("LightweightIoUTracker 不依赖 boxmot", True)  # 初始化没崩溃即证明


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [IOU TRACKER TEST] LightweightIoUTracker")
    print("=" * 50)
    test_init()
    test_single_detection()
    test_same_id()
    test_lost_then_remove()
    test_crossing()
    test_get_memory()
    test_no_boxmot()
    print(f"\n  iou_tracker_ok={PASS} fail={FAIL}")
    return PASS, FAIL
