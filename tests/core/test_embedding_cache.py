"""
test_embedding_cache.py — Embedding Cache 验收

Q1: cache_embedding存入？
Q2: lookup_embedding命中？
Q3: TTL过期后miss？
Q4: 超容量淘汰？
"""
import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

import numpy as np
from core.person.person_manager import PersonManager

PASS, FAIL = 0, 0

def _make_emb(seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(512).astype(np.float32)

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; print(f"  [FAIL] {name}  -- {detail}")

def test_cache_store_lookup():
    print("\n--- 存入命中 ---")
    pm = PersonManager(max_age=5.0, embedding_cache_ttl=30, max_cache_size=128)
    emb = _make_emb(42)
    pm.cache_embedding(1, "Byron", emb)
    result = pm.lookup_embedding(1)
    check("命中", result is not None)
    check("identity=Byron", result is not None and result[0] == "Byron")
    check("cache_size=1", pm.cache_size == 1)


def test_cache_ttl_expiry():
    print("\n--- TTL过期 ---")
    pm = PersonManager(max_age=5.0, embedding_cache_ttl=0.02, max_cache_size=128)
    emb = _make_emb(42)
    pm.cache_embedding(1, "Byron", emb)
    check("初始命中", pm.lookup_embedding(1) is not None)
    time.sleep(0.03)
    result = pm.lookup_embedding(1)
    check("过期miss", result is None)


def test_cache_max_size():
    print("\n--- 容量淘汰 ---")
    pm = PersonManager(max_age=5.0, embedding_cache_ttl=30, max_cache_size=3)
    for i in range(5):
        pm.cache_embedding(i, f"User{i}", _make_emb(i))
    check("不超max_cache_size=3", pm.cache_size <= 3)


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [CACHE TEST] EmbeddingCache")
    print("=" * 50)
    test_cache_store_lookup()
    test_cache_ttl_expiry()
    test_cache_max_size()
    print(f"\n  cache_ok={PASS} fail={FAIL}")
    return PASS, FAIL
