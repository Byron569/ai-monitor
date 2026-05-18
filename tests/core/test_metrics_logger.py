"""
test_metrics_logger.py — MetricsLogger 指标记录器验收

Q1: counter inc / get 正确？
Q2: gauge set / get 正确？
Q3: elapsed 时间递增？
Q4: summary 格式化输出？
"""

import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

from core.metrics_logger import MetricsLogger

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


def test_counter():
    print("\n--- counter 计数器 ---")
    m = MetricsLogger()
    check("未定义的counter返回0", m.get("frames") == 0)
    m.inc("frames")
    check("inc()默认+1", m.get("frames") == 1)
    m.inc("frames", 10)
    check("inc(frames,10)", m.get("frames") == 11)
    m.inc("frames", -1)
    check("inc(frames,-1)", m.get("frames") == 10)
    m.inc("faces")
    check("独立counter不影响", m.get("faces") == 1 and m.get("frames") == 10)


def test_gauge():
    print("\n--- gauge 仪表 ---")
    m = MetricsLogger()
    m.set("fps", 30.5)
    check("set + get", m.get("fps") == 30.5)
    m.set("fps", 25.0)
    check("覆盖更新", m.get("fps") == 25.0)
    check("未定义的gauge返回0", m.get("unknown_gauge") == 0)
    check("带default返回值", m.get("unknown", -1) == -1)


def test_elapsed():
    print("\n--- elapsed 运行时间 ---")
    m = MetricsLogger()
    t0 = m.elapsed()
    check("初始elapsed>=0", t0 >= 0)
    time.sleep(0.01)
    t1 = m.elapsed()
    check("elapsed递增", t1 > t0, f"t0={t0:.4f} t1={t1:.4f}")


def test_summary():
    print("\n--- summary 格式化 ---")
    m = MetricsLogger()
    m.inc("detections", 150)
    m.set("avg_latency", 12.3)
    s = m.summary()
    check("包含[METRICS]", "[METRICS]" in s)
    check("包含runtime", "runtime=" in s)
    check("包含detections=150", "detections=150" in s)
    check("包含avg_latency=12.30", "avg_latency=12.30" in s)


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [METRICS TEST] MetricsLogger")
    print("=" * 50)
    test_counter()
    test_gauge()
    test_elapsed()
    test_summary()
    print(f"\n  metrics_ok={PASS} fail={FAIL}")
    return PASS, FAIL
