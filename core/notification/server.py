"""
NotificationServer — FastAPI + WebSocket notification server.

Runs in a daemon thread, provides REST endpoints and a WebSocket for
the web dashboard, and maintains a rolling buffer of events.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

_fastapi_available: bool = False
_uvicorn_available: bool = False

try:
    from fastapi import FastAPI, WebSocket  # noqa: F401
    from fastapi.responses import Response  # noqa: F401

    _fastapi_available = True
except ImportError:
    FastAPI = None  # type: ignore
    WebSocket = None  # type: ignore
    Response = None  # type: ignore
    logger.warning(
        "[NotificationServer] fastapi not installed — server disabled. "
        "Install: pip install fastapi"
    )

try:
    import uvicorn  # noqa: F401

    _uvicorn_available = True
except ImportError:
    uvicorn = None  # type: ignore
    logger.warning(
        "[NotificationServer] uvicorn not installed — server disabled. "
        "Install: pip install uvicorn[standard]"
    )

SERVER_ENABLED = _fastapi_available and _uvicorn_available


class NotificationServer:
    """FastAPI + WebSocket notification server.

    Runs in a daemon thread so it does not block the main pipeline loop.
    Exposes:

    * ``GET /api/status`` — current FPS, persons, fall alerts, memory
    * ``GET /api/persons`` — list of active persons
    * ``GET /api/events`` — rolling event buffer (optional ``since`` query)
    * ``GET /api/snapshot`` — latest frame JPEG
    * ``WS /ws`` — real-time push to dashboard clients
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, max_events: int = 1000):
        self.host = host
        self.port = port
        self._max_events = max_events

        # Rolling data
        self._events: deque = deque(maxlen=max_events)
        self._clients: Set[Any] = set()  # WebSocket client connections
        self._status: dict = {"fps": 0, "persons": 0, "fall_alerts": 0, "mem": "0%"}
        self._persons: list = []
        self._snapshot: bytes = b""

        # Threading
        self._server: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # FastAPI app
        if SERVER_ENABLED:
            self._app = FastAPI(title="Vision AI Notification Server")
            self._setup_routes()
        else:
            self._app = None

    # ------------------------------------------------------------------
    # Route setup
    # ------------------------------------------------------------------

    def _setup_routes(self) -> None:
        if not SERVER_ENABLED:
            return

        from fastapi import WebSocket, WebSocketDisconnect
        from fastapi.responses import Response

        server = self  # closure capture

        @self._app.get("/api/status")
        async def status():
            return server._status

        @self._app.get("/api/persons")
        async def persons():
            return server._persons

        @self._app.get("/api/events")
        async def events(since: int = 0):
            if since > 0:
                return [e for e in server._events if e.get("ts", 0) > since]
            return list(server._events)

        @self._app.get("/api/snapshot")
        async def snapshot():
            return Response(content=server._snapshot, media_type="image/jpeg")

        @self._app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            server._clients.add(websocket)
            try:
                while True:
                    await websocket.receive_text()  # keep-alive pong
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                server._clients.discard(websocket)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start uvicorn in a daemon thread."""
        if not SERVER_ENABLED:
            logger.info("[NotificationServer] disabled (fastapi/uvicorn not available)")
            return
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_server,
            name="NotificationServer",
            daemon=True,
        )
        self._thread.start()
        logger.info("[NotificationServer] started on %s:%s", self.host, self.port)

    def _run_server(self) -> None:
        """Run uvicorn (blocking, runs in daemon thread)."""
        try:
            uvicorn.run(
                self._app,
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )
        except Exception as exc:
            logger.error("[NotificationServer] uvicorn error: %s", exc)
            self._running = False

    def stop(self) -> None:
        """Signal shutdown."""
        self._running = False
        logger.info("[NotificationServer] stopped")

    # ------------------------------------------------------------------
    # Data update methods (called from pipeline thread)
    # ------------------------------------------------------------------

    def push_event(self, event: dict) -> None:
        """Add an event to the rolling buffer and broadcast via WebSocket."""
        import json

        self._events.append(event)
        self._broadcast({"type": "event", "data": event})

    def update_status(
        self, fps: float, persons: int, fall_alerts: int, mem_percent: str
    ) -> None:
        """Update the /api/status data and broadcast."""
        self._status = {
            "fps": round(fps, 1),
            "persons": persons,
            "fall_alerts": fall_alerts,
            "mem": mem_percent,
        }
        self._broadcast({"type": "status", "data": self._status})

    def update_persons(self, persons_list: list) -> None:
        """Update the person list and broadcast."""
        self._persons = persons_list
        self._broadcast({"type": "persons", "data": persons_list})

    def update_snapshot(self, jpeg_bytes: bytes) -> None:
        """Update the latest snapshot JPEG."""
        self._snapshot = jpeg_bytes

    # ------------------------------------------------------------------
    # WebSocket broadcast
    # ------------------------------------------------------------------

    def _broadcast(self, message: dict) -> None:
        """Send a JSON message to all connected WebSocket clients."""
        import json

        if not self._clients:
            return
        payload = json.dumps(message, ensure_ascii=False)
        dead: list = []
        for ws in self._clients:
            try:
                import asyncio

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(ws.send_text(payload))
                loop.close()
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)
