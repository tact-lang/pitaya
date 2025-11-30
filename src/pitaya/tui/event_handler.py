"""
Event handler for processing Pitaya events.

Transforms raw events from the event stream into TUI state updates.
"""

from .event_processing import EventProcessor, AsyncEventStream, EventFileWatcher

__all__ = ["EventProcessor", "AsyncEventStream", "EventFileWatcher"]
