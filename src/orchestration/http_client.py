"""
HTTP client for connecting to orchestrator server.

Provides read-only access to orchestrator state and events.
"""

import aiohttp
import asyncio
from typing import Dict, Any, List, Tuple, Optional, Set
import logging


logger = logging.getLogger(__name__)


class OrchestratorClient:
    """HTTP client for orchestrator server."""

    def __init__(self, base_url: str):
        """
        Initialize orchestrator client.

        Args:
            base_url: Base URL of orchestrator server (e.g. http://localhost:8080)
        """
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure we have an active session."""
        if not self._session:
            self._session = aiohttp.ClientSession()
        return self._session

    async def get_state(self) -> Optional[Dict[str, Any]]:
        """
        Get current orchestrator state.

        Returns:
            State dictionary or None if error
        """
        try:
            session = await self._ensure_session()
            async with session.get(f"{self.base_url}/state") as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"Failed to get state: {resp.status}")
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            logger.error(f"Error getting state: {e}")
            return None

    async def get_events(
        self,
        offset: int = 0,
        limit: int = 1000,
        run_id: Optional[str] = None,
        event_types: Optional[Set[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get events from orchestrator.

        Args:
            offset: Starting offset
            limit: Maximum events to return
            run_id: Filter by run ID
            event_types: Filter by event types

        Returns:
            Tuple of (events list, new offset)
        """
        try:
            # Build query parameters
            # Spec uses 'since' for the starting offset
            params = {"since": offset, "limit": limit}
            if run_id:
                params["run_id"] = run_id
            if event_types:
                params["event_types"] = ",".join(event_types)

            session = await self._ensure_session()
            async with session.get(f"{self.base_url}/events", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("events", []), data.get("next_offset", offset)
                else:
                    logger.error(f"Failed to get events: {resp.status}")
                    return [], offset
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            logger.error(f"Error getting events: {e}")
            return [], offset

    async def close(self):
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None
