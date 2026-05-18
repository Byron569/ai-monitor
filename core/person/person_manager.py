"""
PersonManager — 以"人"为中心的状态管理器。

v8 优化:
  - Embedding Cache: track_id→embedding, TTL限制, 大小限制
  - Re-ID Cache: 丢失 track 的 embedding 保留供后续匹配
"""

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np


class PersonManager:
    """
    多人体状态管理器。

    职责：
      1. 维护活跃 Person 字典 (track_id → Person)
      2. 身份缓存 (track_id → identity + embedding)
      3. 过期 Person 自动清理
      4. 人脸特征重识别 (Embedding Re-ID)
      5. 运行时 Embedding Cache (TTL + 大小限制)
    """

    REID_CACHE_TTL = 10.0
    REID_SIM_THRESHOLD = 0.5

    def __init__(
        self,
        max_age: float = 5.0,
        embedding_cache_ttl: float = 30.0,
        max_cache_size: int = 128,
    ) -> None:
        from core.person.person import Person
        self._Person = Person
        self._persons: Dict[int, Person] = {}
        self.max_age = max_age

        # Re-ID cache: {key: (name, embedding, timestamp)}
        self._reid_cache: Dict[str, Tuple[str, np.ndarray, float]] = {}

        # Embedding Cache: track_id → (embedding, identity, timestamp)
        self._emb_cache: OrderedDict = OrderedDict()
        self._emb_cache_ttl = embedding_cache_ttl
        self._emb_cache_max = max_cache_size

    def get_or_create(
        self, track_id: int, bbox: tuple, confidence: float = 0.0
    ):
        if track_id in self._persons:
            p = self._persons[track_id]
            p.update_bbox(bbox, confidence)
            return p
        p = self._Person(track_id, bbox, confidence)
        self._persons[track_id] = p
        return p

    # ------------------------------------------------------------------
    # Embedding Cache (runtime)
    # ------------------------------------------------------------------

    def cache_embedding(self, track_id: int, identity: str, embedding: np.ndarray) -> None:
        """缓存 embedding 用于后续快速匹配。"""
        if embedding is None:
            return
        self._emb_cache[track_id] = (embedding.copy(), identity, time.time())
        self._emb_cache.move_to_end(track_id)
        # 淘汰最旧
        while len(self._emb_cache) > self._emb_cache_max:
            self._emb_cache.popitem(last=False)
        self._evict_expired_embeddings()

    def lookup_embedding(self, track_id: int) -> Optional[Tuple[str, np.ndarray]]:
        """查找缓存的 embedding。返回 (identity, embedding) 或 None。"""
        if track_id not in self._emb_cache:
            return None
        emb, identity, ts = self._emb_cache[track_id]
        if time.time() - ts > self._emb_cache_ttl:
            del self._emb_cache[track_id]
            return None
        self._emb_cache.move_to_end(track_id)
        return (identity, emb)

    def _evict_expired_embeddings(self) -> None:
        now = time.time()
        expired = [tid for tid, (_, _, ts) in self._emb_cache.items()
                    if now - ts > self._emb_cache_ttl]
        for tid in expired:
            del self._emb_cache[tid]

    @property
    def cache_size(self) -> int:
        return len(self._emb_cache)

    # ------------------------------------------------------------------
    # Re-ID Cache
    # ------------------------------------------------------------------

    def find_cached_identity(self, query_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        if query_embedding is None or not self._reid_cache:
            return None, 0.0
        now = time.time()
        expired_keys = [k for k, v in self._reid_cache.items()
                        if now - v[2] > self.REID_CACHE_TTL]
        for k in expired_keys:
            del self._reid_cache[k]
        if not self._reid_cache:
            return None, 0.0
        best_name = None
        best_score = 0.0
        for name, cached_emb, ts in self._reid_cache.values():
            score = float(np.dot(query_embedding.ravel(), cached_emb.ravel()))
            if score > best_score:
                best_score = score
                best_name = name
        if best_score >= self.REID_SIM_THRESHOLD:
            return best_name, best_score
        return None, 0.0

    def _cache_embedding(self, name: str, embedding: np.ndarray) -> None:
        if embedding is None:
            return
        key = f"{name}_{time.time()}"
        self._reid_cache[key] = (name, embedding.copy(), time.time())

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get(self, track_id: int):
        return self._persons.get(track_id)

    def identify(self, track_id: int, name: str, embedding: np.ndarray) -> None:
        p = self._persons.get(track_id)
        if p:
            p.mark_identified(name, embedding)

    def is_identified(self, track_id: int) -> bool:
        p = self._persons.get(track_id)
        return p is not None and p.is_identified

    def get_identity(self, track_id: int) -> str:
        p = self._persons.get(track_id)
        return p.identity if p else "Unknown"

    def remove(self, track_id: int) -> None:
        p = self._persons.get(track_id)
        if p and p.is_identified and p.embedding is not None:
            self._cache_embedding(p.identity, p.embedding)
        self._persons.pop(track_id, None)
        self._emb_cache.pop(track_id, None)

    def cleanup(self) -> None:
        now = time.time()
        expired = [tid for tid, p in self._persons.items()
                   if now - p.last_seen > self.max_age]
        for tid in expired:
            self.remove(tid)

    def get_unidentified(self) -> List:
        candidates = [p for p in self._persons.values()
                      if not p.is_identified]
        candidates.sort(key=lambda p: p.confidence, reverse=True)
        return candidates

    def get_identified(self) -> List:
        return [p for p in self._persons.values() if p.is_identified]

    def reset_identity(self, track_id: int) -> None:
        p = self._persons.get(track_id)
        if p:
            p.identity = "Unknown"
            p.embedding = None

    def get_active(self) -> Dict[int, object]:
        return dict(self._persons)

    @property
    def count(self) -> int:
        return len(self._persons)

    @property
    def stable_count(self) -> int:
        return len([p for p in self._persons.values() if p.frame_seen >= 3])
