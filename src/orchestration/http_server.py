"""
HTTP server for multi-UI support.

Provides read-only endpoints for external monitoring tools and web dashboards
to observe orchestration runs in progress.
"""

import json
import logging
from typing import Optional, TYPE_CHECKING
import threading
import asyncio
from aiohttp import web

if TYPE_CHECKING:
    from .orchestrator import Orchestrator


logger = logging.getLogger(__name__)


class OrchestratorHTTPServer:
    """
    HTTP server providing read-only access to orchestration state and events.

    This enables web dashboards and monitoring tools to observe runs without
    direct access to the orchestration layer.
    """

    def __init__(self, orchestrator: "Orchestrator", port: int = 8080):
        """
        Initialize HTTP server.

        Args:
            orchestrator: Reference to the orchestrator instance
            port: Port to listen on
        """
        self.orchestrator = orchestrator
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started_event: Optional[threading.Event] = None

        # Setup routes
        self.app.router.add_get("/state", self.handle_get_state)
        self.app.router.add_get("/events", self.handle_get_events)
        self.app.router.add_get("/health", self.handle_health)

    async def start(self) -> None:
        """Start the HTTP server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await site.start()
        logger.info(f"HTTP server started on port {self.port}")

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self.runner:
            await self.runner.cleanup()
            logger.info("HTTP server stopped")

    # Threaded helpers to match spec "separate thread" wording
    def start_threaded(self) -> None:
        """Start the HTTP server in a separate thread with its own event loop."""
        if self._thread and self._thread.is_alive():
            return

        self._started_event = threading.Event()

        def _run():
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                async def _async_start():
                    await self.start()
                self._loop.run_until_complete(_async_start())
                if self._started_event:
                    self._started_event.set()
                self._loop.run_forever()
            except Exception as e:
                logger.exception(f"HTTP server thread error: {e}")
            finally:
                try:
                    if self._loop and not self._loop.is_closed():
                        self._loop.run_until_complete(self.stop())
                        self._loop.stop()
                        self._loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=_run, name="OrchestratorHTTPServer", daemon=True)
        self._thread.start()
        if self._started_event:
            self._started_event.wait(timeout=5.0)

    def stop_threaded(self) -> None:
        """Stop the threaded HTTP server and join the thread."""
        try:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    async def handle_get_state(self, request: web.Request) -> web.Response:
        """
        Handle GET /state - returns current orchestration state.

        Returns:
            JSON response with current state or 404 if no active run
        """
        try:
            state = self.orchestrator.get_current_state()
            if not state:
                return web.json_response(
                    {"error": "No active orchestration run"}, status=404
                )

            return web.json_response(state.to_dict())

        except Exception as e:
            logger.exception("Error handling /state request")
            return web.json_response(
                {"error": f"Internal server error: {str(e)}"}, status=500
            )

    async def handle_get_events(self, request: web.Request) -> web.Response:
        """
        Handle GET /events - stream events from a given offset.

        Query parameters:
            since: Byte offset to start from (default: 0)
            limit: Maximum number of events to return (default: 1000, max: 10000)

        Returns:
            JSON response with events array and next offset
        """
        try:
            # Parse query parameters; support both 'since' (spec) and 'offset' (client)
            since_q = request.query.get("since")
            offset_q = request.query.get("offset")
            since_offset = int(since_q if since_q is not None else (offset_q or "0"))
            limit = min(int(request.query.get("limit", "1000")), 10000)
            # Optional timestamp filter (ISO 8601)
            ts_q = request.query.get("since_ts")
            ts = None
            if ts_q:
                try:
                    from datetime import datetime
                    ts = datetime.fromisoformat(ts_q)
                except Exception:
                    ts = None

            # Get events from orchestrator
            events, next_offset = await self.orchestrator.get_events_since(
                offset=since_offset, limit=limit, timestamp=ts
            )

            return web.json_response(
                {
                    "events": events,
                    "next_offset": next_offset,
                    "has_more": len(events) == limit,
                }
            )

        except ValueError as e:
            return web.json_response(
                {"error": f"Invalid query parameters: {str(e)}"}, status=400
            )
        except (OSError, json.JSONDecodeError) as e:
            logger.exception("Error handling /events request")
            return web.json_response(
                {"error": f"Internal server error: {str(e)}"}, status=500
            )

    async def handle_health(self, request: web.Request) -> web.Response:
        """
        Handle GET /health - simple health check endpoint.

        Returns:
            JSON response with server status
        """
        return web.json_response(
            {
                "status": "ok",
                "active_run": self.orchestrator.get_current_state() is not None,
            }
        )
