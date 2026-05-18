"""
test_vision_task_interface.py — VisionTask 接口验收

Q1: VisionTask 抽象类可继承？
Q2: should_run 返回 bool？
Q3: run 返回 VisionEvent 列表？
Q4: VisionEvent 字段完整？
"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

from core.interfaces import VisionTask, VisionEvent

PASS, FAIL = 0, 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; print(f"  [FAIL] {name}  -- {detail}")

def test_vision_event():
    print("\n--- VisionEvent 构造 ---")
    evt = VisionEvent(event_type="fall_detected", track_id=3, confidence=0.87,
                       payload={"bbox": (0,0,100,100)})
    check("event_type=fall_detected", evt.event_type == "fall_detected")
    check("track_id=3", evt.track_id == 3)
    check("confidence=0.87", abs(evt.confidence - 0.87) < 0.001)
    check("timestamp > 0", evt.timestamp > 0)
    check("payload包含bbox", "bbox" in evt.payload)
    check("__repr__包含type", "fall_detected" in repr(evt))

def test_concrete_task():
    print("\n--- 具体任务实现 ---")
    class MockTask(VisionTask):
        def __init__(self):
            super().__init__()
            self.name = "mock"
            self.enabled = True
            self.interval = 3
        def should_run(self, frame_id, tracks, context):
            return frame_id % self.interval == 0 and len(tracks) > 0
        def run(self, frame, tracks, context):
            return [VisionEvent("mock_event", track_id=1, confidence=0.9)]

    task = MockTask()
    check("name=mock", task.name == "mock")
    check("enabled=True", task.enabled)
    check("should_run(frame=6, tracks=[1])", task.should_run(6, [(1,)] , {}))
    check("should_run(frame=7) false", not task.should_run(7, [(1,)], {}))

    events = task.run(None, [(1,)], {})
    check("返回1个event", len(events) == 1)
    check("event_type=mock_event", events[0].event_type == "mock_event")

def test_disabled_task():
    print("\n--- 禁用任务不运行 ---")
    class DisabledTask(VisionTask):
        def __init__(self):
            super().__init__()
            self.enabled = False
            self.interval = 1
        def should_run(self, frame_id, tracks, context):
            return self.enabled
        def run(self, frame, tracks, context):
            return [VisionEvent("never")]

    task = DisabledTask()
    check("should_run=False", not task.should_run(0, [(1,)], {}))

def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [INTERFACE TEST] VisionTask + VisionEvent")
    print("=" * 50)
    test_vision_event()
    test_concrete_task()
    test_disabled_task()
    print(f"\n  interface_ok={PASS} fail={FAIL}")
    return PASS, FAIL
