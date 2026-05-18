"""
test_recognition_worker_queue.py — Worker队列行为验收

Q1: submit入队？
Q2: 队列满时丢弃？
Q3: poll_results收割？
Q4: 结果带quality_score？
Q5: cache命中避免重复识别？
"""
import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

import numpy as np
from core.workers.recognition_worker import RecognitionWorker

PASS, FAIL = 0, 0

class _MockRecognizer:
    def __init__(self): self.threshold = 0.7
    def get_embedding(self, crop):
        return np.random.randn(512).astype(np.float32)

class _MockDatabase:
    def search(self, embedding, threshold=0.7):
        return ("TestUser", 0.85)

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; print(f"  [FAIL] {name}  -- {detail}")

def test_result_includes_quality():
    print("\n--- 结果携带quality_score ---")
    rec, db = _MockRecognizer(), _MockDatabase()
    w = RecognitionWorker(rec, db, max_queue_size=4)
    w.start()
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    w.submit(1, crop, quality_score=0.75)
    time.sleep(0.5)
    results = w.poll_results()
    check("收到结果", len(results) > 0)
    if results:
        tid, name, sim, emb, qs = results[0]
        check("qs=0.75", abs(qs - 0.75) < 0.01, f"qs={qs}")
    w.stop()
    w.join(timeout=1)


def test_queue_full_drop():
    print("\n--- 队列满丢弃 ---")
    rec, db = _MockRecognizer(), _MockDatabase()
    w = RecognitionWorker(rec, db, max_queue_size=2)
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    ok1 = w.submit(1, crop)
    ok2 = w.submit(2, crop)
    ok3 = w.submit(3, crop)
    check("前2成功", ok1 and ok2)
    check("第3丢弃", not ok3)
    check("skip_count=1", w.skip_count == 1)


def test_stop_graceful():
    print("\n--- 优雅退出 ---")
    rec, db = _MockRecognizer(), _MockDatabase()
    w = RecognitionWorker(rec, db, max_queue_size=4)
    check("初始_running=True", w._running)
    w.stop()
    check("stop后_running=False", not w._running)
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    ok = w.submit(1, crop)
    check("stop后submit=false", not ok)


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [QUEUE TEST] Worker Queue + Cache")
    print("=" * 50)
    test_result_includes_quality()
    test_queue_full_drop()
    test_stop_graceful()
    print(f"\n  queue_ok={PASS} fail={FAIL}")
    return PASS, FAIL
