"""
HTTP server for multi-UI support.

Provides read-only endpoints for external monitoring tools and web dashboards
to observe orchestration runs in progress.
"""

import json
import logging
from typing import Optional, TYPE_CHECKING
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

        except (AttributeError, TypeError, json.JSONEncodeError) as e:
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
            # Parse query parameters
            since_offset = int(request.query.get("since", "0"))
            limit = min(int(request.query.get("limit", "1000")), 10000)

            # Get events from orchestrator
            events, next_offset = await self.orchestrator.get_events_since(
                offset=since_offset, limit=limit
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
