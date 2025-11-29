"""Async helpers for Docker manager."""

import asyncio
from typing import Any, AsyncIterator, Iterable


async def async_iter(blocking_iter: Iterable[Any]) -> AsyncIterator[Any]:
    """Convert a blocking iterator to an async iterator."""
    loop = asyncio.get_event_loop()
    sentinel = object()

    def iterate():
        for item in blocking_iter:
            yield item
        yield sentinel

    iterator = iterate()
    while True:
        item = await loop.run_in_executor(None, lambda: next(iterator))
        if item is sentinel:
            break
        yield item


__all__ = ["async_iter"]
