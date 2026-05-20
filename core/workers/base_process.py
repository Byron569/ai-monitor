"""
BaseInferenceProcess — multiprocess worker base class with SharedMemory IPC.

Provides:
  - Queue-based command/result protocol between main and child process
  - SharedMemory ring buffer (N_SLOTS slots) for efficient frame data transfer
  - Subclass hooks: init_model(), process_frame(), _warmup(), _child_cleanup()
  - Crash recovery: is_alive() check with automatic restart

Lifecycle in child process (run()):
  init_model() -> _warmup() -> cmd_loop (read cmd_queue, dispatch) -> _child_cleanup()

Communication protocol:
  Commands (main -> child):  {"cmd": "submit", "slot_idx": 0, "meta": {...}}
                             {"cmd": "stop"}
  Results (child -> main):   {"slot_idx": 0, "tid": 1, "identity": "Byron", ...}
"""

import logging
import multiprocessing
import os
import queue
import uuid
from multiprocessing import shared_memory
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseInferenceProcess(multiprocessing.Process):
    """Multiprocess worker base class with SharedMemory IPC and crash recovery.

    Subclass responsibilities:
        init_model()       — load models (called once in child process)
        process_frame()    — process a frame from the input ring buffer slot
        _warmup()          — optional warmup after model load
        _child_cleanup()   — optional cleanup before child exits
    """

    # Number of ring-buffer slots for incoming frame data.
    N_SLOTS: int = 4

    # Bytes per slot (default ~900 KB for a 640x480 RGB frame).
    SLOT_SIZE: int = 640 * 480 * 3

    # Output shared-memory buffer size (default 512 KB).
    OUTPUT_SIZE: int = 512 * 1024

    def __init__(self, name: str = "base_process") -> None:
        super().__init__(name=name)
        self.daemon = True
        self._user_name = name

        # -- Communication channels (created in parent, inherited/attached in child)
        self._create_ipc()

        # numpy views of shared memory (set in run() on the child side)
        self._input_buf_np: Optional[np.ndarray] = None
        self._output_buf_np: Optional[np.ndarray] = None

        # -- crash-recovery internal state
        self._crash_check_interval: float = 30.0
        self._last_alive_check: float = 0.0
        self._stopped_intentionally: bool = False

    # ------------------------------------------------------------------
    # IPC creation / teardown
    # ------------------------------------------------------------------

    def _create_ipc(self) -> None:
        """Create fresh multiprocessing Queues and SharedMemory blocks."""
        tag = f"{self._user_name}_{uuid.uuid4().hex[:8]}"

        self._cmd_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._result_queue: multiprocessing.Queue = multiprocessing.Queue()

        # Input ring buffer
        self._input_shm_name: str = f"bis_i_{tag}"
        self._input_shm: shared_memory.SharedMemory = shared_memory.SharedMemory(
            name=self._input_shm_name,
            create=True,
            size=self.N_SLOTS * self.SLOT_SIZE,
        )

        # Output buffer
        self._output_shm_name: str = f"bis_o_{tag}"
        self._output_shm: shared_memory.SharedMemory = shared_memory.SharedMemory(
            name=self._output_shm_name,
            create=True,
            size=self.OUTPUT_SIZE,
        )

    def _cleanup_shm(self) -> None:
        """Close and unlink both SharedMemory blocks (call from whichever
        process owns the handle)."""
        for attr in ("_input_shm", "_output_shm"):
            shm = getattr(self, attr, None)
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
                setattr(self, attr, None)

    def __del__(self) -> None:
        """Destructor: release SharedMemory if not already cleaned up."""
        try:
            self._cleanup_shm()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Subclass hooks (override in subclasses)
    # ------------------------------------------------------------------

    def init_model(self) -> None:
        """Load models or other heavy resources once in the child process."""
        pass

    def process_frame(self, slot_idx: int, meta: dict) -> dict:
        """Process a frame from the input ring buffer slot.

        Args:
            slot_idx: Index into the input ring buffer (0 .. N_SLOTS-1).
            meta: Arbitrary metadata dict from the command sender,
                  e.g. ``{"tid": 1, "timestamp": 1234.5}``.

        Returns:
            dict: Result that will be pushed onto the result queue for
                  the main process to collect.
        """
        return {"slot_idx": slot_idx}

    def _warmup(self) -> None:
        """Optional warmup step.  Runs in the child after init_model()."""
        pass

    def _child_cleanup(self) -> None:
        """Cleanup hook called in the child before the process exits."""
        pass

    # ------------------------------------------------------------------
    # Child-process entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Entry point for the child process.

        Order: attach SHM -> init_model -> _warmup -> command loop -> cleanup.
        """
        logger.info("[%s] Process started (PID=%s)", self.name, os.getpid())

        # Attach to the shared memory created by the parent.
        try:
            child_in = shared_memory.SharedMemory(name=self._input_shm_name)
            child_out = shared_memory.SharedMemory(name=self._output_shm_name)
        except Exception as exc:
            logger.error(
                "[%s] Failed to attach shared memory: %s", self.name, exc
            )
            return

        self._input_shm = child_in
        self._output_shm = child_out
        self._input_buf_np = np.ndarray(
            (self.N_SLOTS * self.SLOT_SIZE,),
            dtype=np.uint8,
            buffer=self._input_shm.buf,
        )
        self._output_buf_np = np.ndarray(
            (self.OUTPUT_SIZE,),
            dtype=np.uint8,
            buffer=self._output_shm.buf,
        )

        try:
            self.init_model()
            self._warmup()
            self._cmd_loop()
        except Exception:
            logger.exception("[%s] Unhandled error in run()", self.name)
        finally:
            self._child_cleanup()
            child_in.close()
            child_out.close()
            logger.info("[%s] Process exiting", self.name)

    # ------------------------------------------------------------------
    # Command loop (runs in child)
    # ------------------------------------------------------------------

    def _cmd_loop(self) -> None:
        """Read commands from ``_cmd_queue`` and dispatch them.

        Known commands:
            submit  -> calls process_frame and pushes the result
            stop    -> breaks the loop
        """
        while True:
            cmd = self._cmd_queue.get()
            if not isinstance(cmd, dict):
                logger.warning("[%s] Ignoring non-dict command: %s", self.name, cmd)
                continue

            cmd_type = cmd.get("cmd", "")

            if cmd_type == "stop":
                logger.info("[%s] Received stop command", self.name)
                break

            if cmd_type == "submit":
                slot_idx = cmd.get("slot_idx", 0)
                meta = cmd.get("meta", {})
                try:
                    result = self.process_frame(slot_idx, meta)
                    if not isinstance(result, dict):
                        result = {"result": result, "slot_idx": slot_idx}
                    self._result_queue.put(result)
                except Exception:
                    logger.exception(
                        "[%s] process_frame(slot=%s) failed", self.name, slot_idx
                    )
                    self._result_queue.put({
                        "error": True,
                        "slot_idx": slot_idx,
                        "message": "process_frame raised an exception",
                    })
            else:
                logger.warning("[%s] Unknown command: %s", self.name, cmd_type)

    # ------------------------------------------------------------------
    # Main-process API
    # ------------------------------------------------------------------

    def send_cmd(self, cmd: dict) -> None:
        """Send a command dict to the child process.

        Example::

            worker.send_cmd({"cmd": "submit", "slot_idx": 0, "meta": {"tid": 1}})
        """
        self._cmd_queue.put(cmd)

    def submit_frame(
        self,
        frame: np.ndarray,
        slot_idx: int,
        meta: Optional[dict] = None,
    ) -> None:
        """Write a frame into the input ring buffer and send a submit command.

        This is a convenience wrapper.  It copies *frame* into the shared
        memory slot at *slot_idx* and enqueues a ``submit`` command with
        the given metadata.

        Args:
            frame: uint8 array to write into the slot.
            slot_idx: Target ring-buffer slot index (0 .. N_SLOTS-1).
            meta: Optional metadata dict (e.g. ``{"tid": 1}``).
        """
        if frame.nbytes > self.SLOT_SIZE:
            raise ValueError(
                f"Frame size ({frame.nbytes}) exceeds SLOT_SIZE ({self.SLOT_SIZE})"
            )
        if not (0 <= slot_idx < self.N_SLOTS):
            raise ValueError(f"slot_idx {slot_idx} out of range [0, {self.N_SLOTS})")

        offset = slot_idx * self.SLOT_SIZE
        self._input_shm.buf[offset: offset + frame.nbytes] = frame.tobytes()
        self.send_cmd({
            "cmd": "submit",
            "slot_idx": slot_idx,
            "meta": meta or {},
        })

    def poll_results(self) -> list[dict]:
        """Non-blocking read of all currently available results.

        Returns:
            List of result dicts produced by the child's ``process_frame``.
        """
        results: list[dict] = []
        while True:
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def wait_for_result(self, timeout: Optional[float] = None) -> Optional[dict]:
        """Block until one result is available.

        Args:
            timeout: Maximum seconds to wait (``None`` = infinite).

        Returns:
            A result dict, or ``None`` on timeout / queue error.
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except Exception:
            return None

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop the child process.

        Sends a ``stop`` command, joins with *timeout*, then force-terminates
        if still alive.  Shared memory is cleaned up afterwards.
        """
        self._stopped_intentionally = True
        if self.is_alive():
            try:
                self.send_cmd({"cmd": "stop"})
                self.join(timeout=timeout)
            except Exception as exc:
                logger.warning(
                    "[%s] Error during graceful stop: %s", self.name, exc
                )

            if self.is_alive():
                logger.warning("[%s] Force terminate", self.name)
                self.terminate()
                self.join(timeout=2.0)

        self._cleanup_shm()

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    def check_alive(self) -> bool:
        """Check whether the child process is alive and restart if dead.

        Returns:
            ``True`` if the child is alive (or was successfully restarted).
        """
        if self._stopped_intentionally:
            return False

        if self.is_alive():
            return True

        logger.warning(
            "[%s] Process dead (PID=%s). Restarting...",
            self.name,
            self.pid,
        )
        self._restart()
        return self.is_alive()

    def _restart(self) -> None:
        """Reset internal process state and spin up a fresh child.

        Because ``multiprocessing.Process`` cannot call ``start()`` twice,
        we manually reset the guarded attributes so ``start()`` may be
        invoked again.
        """
        try:
            # -- Release the old popen handle
            if self._popen is not None:
                try:
                    self._popen.poll()       # collect exit code
                    self._popen.close()
                except Exception:
                    pass
                self._popen = None
                if hasattr(self, "_sentinel"):
                    del self._sentinel
                multiprocessing.process._children.discard(self)

            self._closed = False

            # -- Fresh IPC: create new SHM before cleaning up old SHM
            _old_shms = []
            for attr in ("_input_shm", "_output_shm"):
                shm = getattr(self, attr, None)
                if shm is not None:
                    _old_shms.append(shm)
                    setattr(self, attr, None)

            self._create_ipc()

            for shm in _old_shms:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass

            # ``start()`` deletes ``_target`` / ``_args`` / ``_kwargs`` on the
            # first call, so they must exist before we call ``start()`` again.
            self._target = None
            self._args = ()
            self._kwargs = {}

            self.start()
            logger.info("[%s] Restarted (new PID=%s)", self.name, self.pid)
        except Exception:
            logger.exception("[%s] Failed to restart process", self.name)
