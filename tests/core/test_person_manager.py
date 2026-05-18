"""
test_person_manager.py — PersonManager 人员生命周期验收

Q1: get_or_create 新建/复用？
Q2: identify 设置身份+embedding？
Q3: remove 缓存 embedding？
Q4: find_cached_identity 余弦相似度匹配？
Q5: Re-ID cache TTL 过期清理？
Q6: cleanup 清理过期 Person？
Q7: reset_identity 重置为 Unknown？
Q8: get_unidentified 正确过滤？
"""

import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

import numpy as np
from core.person.person_manager import PersonManager

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


def _make_emb(seed=0):
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_create_update():
    print("\n--- 创建与更新 ---")
    pm = PersonManager(max_age=5.0)
    p = pm.get_or_create(1, (10, 20, 100, 200), 0.9)
    check("创建成功 track_id=1", p.track_id == 1)
    check("初始 frame_seen=1", p.frame_seen == 1)
    p2 = pm.get_or_create(1, (15, 25, 105, 205), 0.95)
    check("复用已有person", p2 is p)
    check("frame_seen=2", p.frame_seen == 2)
    check("count=1", pm.count == 1)


def test_identify():
    print("\n--- 身份识别 ---")
    pm = PersonManager(max_age=5.0)
    p = pm.get_or_create(1, (10, 20, 100, 200))
    emb = _make_emb(42)
    pm.identify(1, "Byron", emb)
    check("身份=Byron", pm.get_identity(1) == "Byron")
    check("is_identified=True", pm.is_identified(1))
    check("embedding 已保存", p.embedding is not None)


def test_remove_and_cache():
    print("\n--- 移除时缓存 embedding ---")
    pm = PersonManager(max_age=5.0)
    emb = _make_emb(42)
    pm.get_or_create(1, (10, 20, 100, 200))
    pm.identify(1, "Byron", emb)
    check("Re-ID cache 初始为空", len(pm._reid_cache) == 0)
    pm.remove(1)
    check("移除后 cache 有 1 条", len(pm._reid_cache) == 1)
    check("person 已删除", pm.get(1) is None)


def test_find_cached_identity_match():
    print("\n--- Re-ID 余弦匹配成功 ---")
    pm = PersonManager(max_age=5.0)
    emb = _make_emb(42)
    pm.get_or_create(1, (10, 20, 100, 200))
    pm.identify(1, "Byron", emb)
    pm.remove(1)
    name, sim = pm.find_cached_identity(emb)
    check("匹配成功 name=Byron", name == "Byron")
    check("相似度>=阈值0.5", sim >= 0.5, f"sim={sim:.3f}")


def test_find_cached_identity_no_match():
    print("\n--- Re-ID 无匹配 ---")
    pm = PersonManager(max_age=5.0)
    emb1 = _make_emb(42)
    pm.get_or_create(1, (10, 20, 100, 200))
    pm.identify(1, "Byron", emb1)
    pm.remove(1)
    emb2 = -_make_emb(99)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
    name, sim = pm.find_cached_identity(emb2)
    check("不相似返回None", name is None)


def test_reid_cache_ttl_expiry():
    print("\n--- Re-ID cache TTL 过期 ---")
    pm = PersonManager(max_age=5.0)
    pm.REID_CACHE_TTL = 0.01
    pm.REID_SIM_THRESHOLD = 0.5
    emb = _make_emb(42)
    pm.get_or_create(1, (10, 20, 100, 200))
    pm.identify(1, "Byron", emb)
    pm.remove(1)
    check("移除后 cache 有 1 条", len(pm._reid_cache) == 1)
    time.sleep(0.03)
    name, sim = pm.find_cached_identity(emb)
    check("TTL 过期后无匹配", name is None)
    check("cache 已清空", len(pm._reid_cache) == 0)


def test_cleanup_expired_persons():
    print("\n--- cleanup 清理过期 Person ---")
    pm = PersonManager(max_age=0.01)
    pm.get_or_create(1, (10, 20, 100, 200))
    check("初始 count=1", pm.count == 1)
    time.sleep(0.03)
    pm.cleanup()
    check("过期后 count=0", pm.count == 0)


def test_reset_identity():
    print("\n--- 重置身份 ---")
    pm = PersonManager(max_age=5.0)
    emb = _make_emb(42)
    pm.get_or_create(1, (10, 20, 100, 200))
    pm.identify(1, "Byron", emb)
    check("初始=Byron", pm.get_identity(1) == "Byron")
    pm.reset_identity(1)
    check("重置后=Unknown", pm.get_identity(1) == "Unknown")
    check("embedding 已清除", pm.get(1).embedding is None)


def test_get_unidentified():
    print("\n--- get_unidentified 过滤 ---")
    pm = PersonManager(max_age=5.0)
    pm.get_or_create(1, (10, 20, 100, 200), 0.8)
    pm.get_or_create(2, (50, 60, 150, 250), 0.9)
    pm.identify(1, "Byron", _make_emb(42))
    unidentified = pm.get_unidentified()
    check("未识别1人", len(unidentified) == 1)
    check("track_id=2", unidentified[0].track_id == 2)


def test_stable_count():
    print("\n--- stable_count 统计 ---")
    pm = PersonManager(max_age=5.0)
    p1 = pm.get_or_create(1, (10, 20, 100, 200))
    p2 = pm.get_or_create(2, (50, 60, 150, 250))
    check("初始 stable=0 (seen<3)", pm.stable_count == 0)
    for _ in range(3):
        pm.get_or_create(1, (10, 20, 100, 200))
    check("p1 stable=1", pm.stable_count == 1)


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [PERSON MANAGER TEST] PersonManager")
    print("=" * 50)
    test_create_update()
    test_identify()
    test_remove_and_cache()
    test_find_cached_identity_match()
    test_find_cached_identity_no_match()
    test_reid_cache_ttl_expiry()
    test_cleanup_expired_persons()
    test_reset_identity()
    test_get_unidentified()
    test_stable_count()
    print(f"\n  person_manager_ok={PASS} fail={FAIL}")
    return PASS, FAIL
