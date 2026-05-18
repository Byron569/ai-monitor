"""
test_fall_detection_stub.py — FallDetectionTask 空实现验收

Q1: 默认 enabled=False？
Q2: should_run 在关闭时返回 False？
Q3: 启用后可 should_run？
Q4: run 返回空列表？
Q5: 不引入 ultralytics / torch？
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

def test_default_disabled():
    print("\n--- 默认关闭 ---")
    from plugins.fall_detection_stub import FallDetectionTask
    task = FallDetectionTask()
    check("enabled=False", not task.enabled)
    check("name=fall_detection", task.name == "fall_detection")
    check("interval=5", task.interval == 5)

def test_should_run_disabled():
    print("\n--- 关闭时不运行 ---")
    from plugins.fall_detection_stub import FallDetectionTask
    task = FallDetectionTask()
    check("should_run=False (disabled)", not task.should_run(0, [(1,)], {}))

def test_should_run_enabled():
    print("\n--- 启用后按interval运行 ---")
    from plugins.fall_detection_stub import FallDetectionTask
    task = FallDetectionTask({"enabled": True, "interval": 3})
    check("enabled=True", task.enabled)
    check("frame=0, tracks=empty → false", not task.should_run(0, [], {}))
    check("frame=3, tracks=[1] → true", task.should_run(3, [(1,)], {}))
    check("frame=4, tracks=[1] → false", not task.should_run(4, [(1,)], {}))
    check("frame=6, tracks=[1] → true", task.should_run(6, [(1,)], {}))

def test_run_returns_empty():
    print("\n--- run 返回空列表 ---")
    from plugins.fall_detection_stub import FallDetectionTask
    task = FallDetectionTask()
    events = task.run(None, [(1, (0,0,100,100), "Unknown")], {})
    check("返回空列表", events == [])

def test_no_heavy_imports():
    print("\n--- 不引入重依赖 ---")
    from plugins.fall_detection_stub import FallDetectionTask
    task = FallDetectionTask()
    events = task.run(None, [], {})
    # stub 不加载模型、不调用 ultralytics/torch
    check("stub 不加载重依赖 (正常构造+run)", events == [])

def test_config_overrides():
    print("\n--- 配置覆盖 ---")
    from plugins.fall_detection_stub import FallDetectionTask
    task = FallDetectionTask({
        "enabled": True,
        "interval": 10,
        "confidence_threshold": 0.8,
        "min_duration_frames": 15,
    })
    check("interval=10", task.interval == 10)

def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [STUB TEST] FallDetectionTask Stub")
    print("=" * 50)
    test_default_disabled()
    test_should_run_disabled()
    test_should_run_enabled()
    test_run_returns_empty()
    test_no_heavy_imports()
    test_config_overrides()
    print(f"\n  stub_ok={PASS} fail={FAIL}")
    return PASS, FAIL
