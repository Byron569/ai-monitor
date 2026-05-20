"""ONNX Inference Session Manager — pooled, LRU-evicted session cache.

Provides a single point of access for ONNX Runtime sessions across the
application.  Sessions are cached by ``(model_path, device, use_int8)`` and
evicted via LRU when the pool size is exceeded.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SessionKey = Tuple[str, str, bool]
"""Cache key tuple: ``(model_path, device, use_int8)``."""


def _build_session_options(
    device: str,
    enable_mem_pattern: bool,
    enable_cpu_arena: bool,
) -> ort.SessionOptions:
    """Build an ``ort.SessionOptions`` following the project convention."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = enable_mem_pattern
    opts.enable_cpu_mem_arena = enable_cpu_arena

    if device.lower() == "cpu":
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1

    return opts


def _select_providers(device: str) -> List[str]:
    """Return the provider list for *device*."""
    if device.lower() == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _make_dummy_input(
    session: ort.InferenceSession,
) -> Dict[str, np.ndarray]:
    """Create a dict of dummy numpy arrays matching the model's input spec.

    Dynamic dimensions (those with value ``-1`` in the ONNX model) are set to
    ``1`` so that the warm-up run can proceed without a real input.
    """
    feed: Dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        shape = list(inp.shape)
        # Replace any dynamic dimensions (-1) with 1
        shape = [1 if dim is None or dim < 0 else dim for dim in shape]

        dtype = np.float32
        if inp.type and "float16" in inp.type:
            dtype = np.float16
        elif inp.type and "int32" in inp.type:
            dtype = np.int32
        elif inp.type and "int64" in inp.type:
            dtype = np.int64
        elif inp.type and "uint8" in inp.type:
            dtype = np.uint8

        feed[inp.name] = np.random.randn(*shape).astype(dtype)
    return feed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class InferenceSessionManager:
    """Pooled ONNX session cache with LRU eviction.

    Parameters
    ----------
    pool_size:
        Maximum number of sessions to keep in the cache.  When the limit is
        exceeded the least-recently-used session is evicted and closed.
    enable_mem_pattern:
        Passed through to ``ort.SessionOptions`` for all devices.
    enable_cpu_arena:
        Passed through to ``ort.SessionOptions`` for all devices.
    """

    def __init__(
        self,
        pool_size: int = 4,
        enable_mem_pattern: bool = True,
        enable_cpu_arena: bool = True,
    ) -> None:
        self._pool_size = pool_size
        self._enable_mem_pattern = enable_mem_pattern
        self._enable_cpu_arena = enable_cpu_arena

        # Guarded by ``_lock``.
        self._cache: OrderedDict[_SessionKey, ort.InferenceSession] = OrderedDict()
        self._lock = threading.Lock()

    # -- Public helpers -----------------------------------------------------

    def get_session(
        self,
        model_path: str,
        device: str = "cpu",
        use_int8: bool = False,
    ) -> ort.InferenceSession:
        """Return a cached ONNX session, creating one if necessary.

        If the cache has reached ``pool_size`` the least-recently-used entry
        is evicted first.
        """
        key = (model_path, device, use_int8)

        with self._lock:
            if key in self._cache:
                # Move to end (most-recently-used).
                session = self._cache.pop(key)
                self._cache[key] = session
                logger.debug(
                    "[SessionManager] cache hit  key=%s  cache_size=%d",
                    key,
                    len(self._cache),
                )
                return session

        # ---- Cache miss: create a new session (outside the lock) ----------
        opts = _build_session_options(
            device=device,
            enable_mem_pattern=self._enable_mem_pattern,
            enable_cpu_arena=self._enable_cpu_arena,
        )
        providers = _select_providers(device)
        session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers,
        )

        # ---- Insert into cache (under lock, re-check for TOCTOU) ----------
        with self._lock:
            # Re-check: another thread may have inserted this key while we
            # were creating our session (TOCTOU race prevention).
            if key in self._cache:
                # Another thread already cached this variant.  Close our
                # duplicate and return the already-cached session.
                try:
                    del session
                    import gc; gc.collect()  # Force reclamation of ORT C++ objects
                except Exception:
                    logger.exception(
                        "[SessionManager] error closing duplicate session key=%s", key
                    )
                # Move cached entry to MRU position.
                session = self._cache.pop(key)
                self._cache[key] = session
                return session

            # Evict LRU (first item) if at capacity.
            if len(self._cache) >= self._pool_size:
                _evict_lru(self._cache)

            self._cache[key] = session
            logger.info(
                "[SessionManager] created session key=%s  cache_size=%d",
                key,
                len(self._cache),
            )

        return session

    def release_session(self, model_path: str) -> None:
        """Remove all cached sessions for *model_path* from cache and close them.

        Iterates through all cached variants (any device or int8 flag) that
        share the same *model_path* and releases each one.
        """
        with self._lock:
            keys_to_remove = [key for key in self._cache if key[0] == model_path]
            sessions = []
            for key in keys_to_remove:
                sessions.append((key, self._cache.pop(key)))
        for key, session in sessions:
            try:
                del session
                import gc; gc.collect()  # Force reclamation of ORT C++ objects
            except Exception:
                logger.exception(
                    "[SessionManager] error closing session key=%s", key
                )

    def warmup_all(
        self,
        model_paths: List[str],
        device: str = "cpu",
        use_int8: bool = False,
    ) -> None:
        """Create (if needed) and warm-up sessions for every *model_paths* entry.

        Each model is run once with a random dummy input to prime any runtime
        memory allocations / JIT compilation.

        Parameters
        ----------
        model_paths:
            List of model file paths to warm up.
        device:
            Target device for the sessions (passed through to ``get_session``).
        use_int8:
            Whether to use int8 quantised sessions (passed through to
            ``get_session``).
        """
        for mp in model_paths:
            session = self.get_session(mp, device=device, use_int8=use_int8)
            try:
                feed = _make_dummy_input(session)
                _ = session.run(None, feed)
                logger.info("[SessionManager] warmup done  path=%s", mp)
            except Exception:
                logger.exception(
                    "[SessionManager] warmup failed  path=%s  (ignored)", mp
                )

    def close(self) -> None:
        """Release all cached sessions and clear the cache."""
        with self._lock:
            sessions = list(self._cache.values())
            self._cache.clear()
        for s in sessions:
            try:
                del s
                import gc; gc.collect()  # Force reclamation of ORT C++ objects
            except Exception:
                logger.exception("[SessionManager] error closing session during close()")


# ---------------------------------------------------------------------------
# Internal helpers (private)
# ---------------------------------------------------------------------------


def _evict_lru(cache: OrderedDict) -> None:
    """Evict the least-recently-used entry from an OrderedDict cache."""
    try:
        _key, session = cache.popitem(last=False)  # FIFO order → LRU
        del session
        import gc; gc.collect()  # Force reclamation of ORT C++ objects
        logger.debug("[SessionManager] evicted LRU  key=%s", _key)
    except KeyError:
        pass  # empty cache — nothing to evict
