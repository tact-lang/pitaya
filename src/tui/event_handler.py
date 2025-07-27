"""
Event handler for processing orchestrator events.

Transforms raw events from the event stream into TUI state updates.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING
import asyncio
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

if TYPE_CHECKING:
    from watchdog.observers import Observer
else:
    try:
        from watchdog.observers import Observer
    except ImportError:
        Observer = Any

from .models import (
    TUIState,
    RunDisplay,
    StrategyDisplay,
    InstanceDisplay,
    InstanceStatus,
)

logger = logging.getLogger(__name__)


class EventProcessor:
    """Processes orchestrator events and updates TUI state."""

    def __init__(self, state: TUIState):
        """
        Initialize event processor.

        Args:
            state: TUI state to update
        """
        self.state = state
        self._event_handlers = self._setup_handlers()

    def _setup_handlers(self) -> Dict[str, Callable]:
        """Setup event type to handler mapping."""
        return {
            # Run-level events
            "run.started": self._handle_run_started,
            "run.completed": self._handle_run_completed,
            "run.failed": self._handle_run_failed,
            # Strategy-level events
            "strategy.started": self._handle_strategy_started,
            "strategy.completed": self._handle_strategy_completed,
            "strategy.failed": self._handle_strategy_failed,
            # Instance-level events
            "instance.queued": self._handle_instance_queued,
            "instance.started": self._handle_instance_started,
            "instance.completed": self._handle_instance_completed,
            "instance.failed": self._handle_instance_failed,
            "instance.progress": self._handle_instance_progress,
            # Instance phase events
            "instance.workspace_preparing": self._handle_instance_workspace_preparing,
            "instance.container_creating": self._handle_instance_container_creating,
            "instance.claude_starting": self._handle_instance_claude_starting,
            "instance.result_collection_started": self._handle_instance_result_collection,
            # Claude-specific events
            "instance.claude_system": self._handle_claude_system,
            "instance.claude_assistant": self._handle_claude_assistant,
            "instance.claude_tool_use": self._handle_claude_tool_use,
            "instance.claude_tool_result": self._handle_claude_tool_result,
            "instance.claude_result": self._handle_claude_result,
        }

    def process_event(self, event: Dict[str, Any]) -> None:
        """
        Process a single event.

        Args:
            event: Event dictionary with type, timestamp, data
        """
        event_type = event.get("type")
        if not event_type:
            return

        # Update event tracking
        self.state.events_processed += 1

        # Debug logging - more detailed for instance events
        if event_type.startswith("instance."):
            logger.info(
                f"Processing {event_type} for instance {event.get('instance_id', 'None')}"
            )

        # Get handler for event type
        handler = self._event_handlers.get(event_type)
        if handler:
            try:
                handler(event)
                logger.debug(f"Successfully processed event: {event_type}")
            except (AttributeError, TypeError, ValueError, KeyError) as e:
                logger.error(f"Error processing event {event_type}: {e}")
                self.state.add_error(f"Event processing error: {e}")
        else:
            logger.debug(f"No handler for event type: {event_type}")

    # Run-level event handlers

    def _handle_run_started(self, event: Dict[str, Any]) -> None:
        """Handle run.started event."""
        data = event.get("data", {})

        self.state.current_run = RunDisplay(
            run_id=data.get("run_id", "unknown"),
            prompt=data.get("prompt", ""),
            repo_path=data.get("repo_path", ""),
            base_branch=data.get("base_branch", "main"),
            started_at=self._parse_timestamp(event.get("timestamp")),
        )

        self.state.connected_to_orchestrator = True

    def _handle_run_completed(self, event: Dict[str, Any]) -> None:
        """Handle run.completed event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        run = self.state.current_run

        run.completed_at = self._parse_timestamp(event.get("timestamp"))
        run.total_cost = data.get("total_cost", 0.0)
        run.total_tokens = data.get("total_tokens", 0)

    def _handle_run_failed(self, event: Dict[str, Any]) -> None:
        """Handle run.failed event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        self.state.add_error(f"Run failed: {data.get('error', 'Unknown error')}")

    # Strategy-level event handlers

    def _handle_strategy_started(self, event: Dict[str, Any]) -> None:
        """Handle strategy.started event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        strategy_id = data.get("strategy_id")
        if not strategy_id:
            return

        strategy = StrategyDisplay(
            strategy_id=strategy_id,
            strategy_name=data.get("strategy_name", "unknown"),
            config=data.get("config", {}),
            started_at=self._parse_timestamp(event.get("timestamp")),
        )

        self.state.current_run.strategies[strategy_id] = strategy
        logger.info(
            f"Created strategy {strategy_id} ({data.get('strategy_name', 'unknown')})"
        )

    def _handle_strategy_completed(self, event: Dict[str, Any]) -> None:
        """Handle strategy.completed event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        strategy_id = data.get("strategy_id")
        if not strategy_id or strategy_id not in self.state.current_run.strategies:
            return

        strategy = self.state.current_run.strategies[strategy_id]
        strategy.completed_at = self._parse_timestamp(event.get("timestamp"))
        strategy.is_complete = True

    def _handle_strategy_failed(self, event: Dict[str, Any]) -> None:
        """Handle strategy.failed event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        strategy_id = data.get("strategy_id")
        if strategy_id and strategy_id in self.state.current_run.strategies:
            strategy = self.state.current_run.strategies[strategy_id]
            strategy.is_complete = True
            self.state.add_error(
                f"Strategy {strategy.strategy_name} failed: {data.get('error', 'Unknown')}"
            )

    # Instance-level event handlers

    def _handle_instance_queued(self, event: Dict[str, Any]) -> None:
        """Handle instance.queued event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        # Try to get instance_id from both places (top level and in data)
        instance_id = event.get("instance_id") or data.get("instance_id")
        if not instance_id:
            logger.warning(f"No instance_id in event: {event}")
            return

        instance = InstanceDisplay(
            instance_id=instance_id,
            strategy_name=data.get("strategy", "unknown"),
            status=InstanceStatus.QUEUED,
            branch_name=data.get("branch_name"),
            last_updated=self._parse_timestamp(event.get("timestamp"))
            or datetime.now(),
        )

        self.state.current_run.instances[instance_id] = instance
        self.state.current_run.total_instances += 1

        # Add instance to strategy tracking
        strategy_name = data.get("strategy", "unknown")
        for strategy in self.state.current_run.strategies.values():
            if strategy.strategy_name == strategy_name:
                strategy.instance_ids.append(instance_id)
                strategy.total_instances += 1
                logger.info(
                    f"Added instance {instance_id} to strategy {strategy.strategy_id}"
                )
                break

        logger.info(
            f"Created instance {instance_id} from queued event, total instances: {len(self.state.current_run.instances)}"
        )

    def _handle_instance_started(self, event: Dict[str, Any]) -> None:
        """Handle instance.started event."""
        if not self.state.current_run:
            logger.warning("Got instance.started but no current_run")
            return

        instance_id = event.get("instance_id")
        if not instance_id:
            logger.warning("Got instance.started without instance_id")
            return

        # Check if instance was already created by queued event
        if instance_id not in self.state.current_run.instances:
            logger.info(f"Creating instance {instance_id} from started event")
            # Create the instance if it doesn't exist
            data = event.get("data", {})
            instance = InstanceDisplay(
                instance_id=instance_id,
                strategy_name=data.get("strategy", "unknown"),
                status=InstanceStatus.RUNNING,
                started_at=self._parse_timestamp(event.get("timestamp")),
                prompt=data.get("prompt"),
                model=data.get("model", "sonnet"),
                current_activity="Starting...",
                last_updated=datetime.now(),
            )
            self.state.current_run.instances[instance_id] = instance
            self.state.current_run.total_instances += 1
            self.state.current_run.active_instances += 1
            logger.info(
                f"Instance {instance_id} started. Active: {self.state.current_run.active_instances}"
            )
        else:
            # Update existing instance
            data = event.get("data", {})
            instance = self.state.current_run.instances[instance_id]

            # Only increment active if status is changing from QUEUED to RUNNING
            was_queued = instance.status == InstanceStatus.QUEUED

            instance.status = InstanceStatus.RUNNING
            instance.started_at = self._parse_timestamp(event.get("timestamp"))
            instance.prompt = data.get("prompt")
            instance.model = data.get("model", "sonnet")
            instance.last_updated = datetime.now()

            if was_queued:
                self.state.current_run.active_instances += 1
                logger.info(
                    f"Instance {instance_id} started (was queued). Active: {self.state.current_run.active_instances}"
                )

    def _handle_instance_completed(self, event: Dict[str, Any]) -> None:
        """Handle instance.completed event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id:
            logger.warning("Got instance.completed without instance_id")
            return

        if instance_id not in self.state.current_run.instances:
            logger.warning(
                f"Got instance.completed for unknown instance: {instance_id}"
            )
            return

        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]

        # Only update to completed if not already completed (avoid race conditions)
        if instance.status != InstanceStatus.COMPLETED:
            instance.status = InstanceStatus.COMPLETED
            instance.completed_at = self._parse_timestamp(event.get("timestamp"))
            instance.branch_name = data.get("branch_name")
            instance.duration_seconds = data.get("duration_seconds", 0.0)
            instance.current_activity = "Completed"  # Clear the activity
            instance.last_updated = datetime.now()
            logger.info(
                f"Instance {instance_id} completed after activity: {instance.current_activity}"
            )

        # Update metrics from data
        metrics = data.get("metrics", {})
        instance.cost = metrics.get("total_cost", 0.0)
        instance.total_tokens = metrics.get("total_tokens", 0)
        instance.input_tokens = metrics.get("input_tokens", 0)
        instance.output_tokens = metrics.get("output_tokens", 0)

        # Update run totals
        self.state.current_run.completed_instances += 1
        self.state.current_run.active_instances = max(
            0, self.state.current_run.active_instances - 1
        )
        self.state.current_run.total_cost += instance.cost
        self.state.current_run.total_tokens += instance.total_tokens

        logger.info(
            f"Instance {instance_id} completed. Run totals - Completed: {self.state.current_run.completed_instances}, Active: {self.state.current_run.active_instances}"
        )

        # Update strategy totals
        for strategy in self.state.current_run.strategies.values():
            if instance_id in strategy.instance_ids:
                strategy.completed_instances += 1
                break

    def _handle_instance_failed(self, event: Dict[str, Any]) -> None:
        """Handle instance.failed event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]

        instance.status = InstanceStatus.FAILED
        instance.completed_at = self._parse_timestamp(event.get("timestamp"))
        instance.error = data.get("error")
        instance.error_type = data.get("error_type")
        instance.last_updated = datetime.now()

        self.state.current_run.failed_instances += 1
        self.state.current_run.active_instances = max(
            0, self.state.current_run.active_instances - 1
        )

        # Update strategy totals
        for strategy in self.state.current_run.strategies.values():
            if instance_id in strategy.instance_ids:
                strategy.failed_instances += 1
                break

    def _handle_instance_progress(self, event: Dict[str, Any]) -> None:
        """Handle instance.progress event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]

        instance.current_activity = data.get("activity")
        instance.last_updated = datetime.now()

    # Claude-specific event handlers

    def _handle_claude_system(self, event: Dict[str, Any]) -> None:
        """Handle Claude system message (connection established)."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Claude connected"
        instance.last_updated = datetime.now()

    def _handle_claude_assistant(self, event: Dict[str, Any]) -> None:
        """Handle Claude assistant message."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Claude is thinking..."
        instance.last_updated = datetime.now()

    def _handle_claude_tool_use(self, event: Dict[str, Any]) -> None:
        """Handle Claude tool use."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]

        tool_name = data.get("tool", "unknown")
        instance.last_tool_use = tool_name

        # Make tool descriptions more user-friendly
        tool_descriptions = {
            "str_replace_editor": "Editing files",
            "bash": "Running commands",
            "str_replace_based_edit_tool": "Editing code",
            "read_file": "Reading files",
            "write_file": "Writing files",
            "list_files": "Listing files",
            "search_files": "Searching files",
            "find_files": "Finding files",
        }

        friendly_name = tool_descriptions.get(tool_name, f"Using {tool_name}")
        instance.current_activity = friendly_name
        instance.last_updated = datetime.now()

    def _handle_claude_tool_result(self, event: Dict[str, Any]) -> None:
        """Handle Claude tool result."""
        # Could track success/failure of tool uses if needed
        pass

    def _handle_claude_result(self, event: Dict[str, Any]) -> None:
        """Handle Claude final result."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]

        # Update final metrics if not already set
        metrics = data.get("metrics", {})
        if metrics:
            if instance.cost == 0.0:
                instance.cost = metrics.get("total_cost", 0.0)
            if instance.total_tokens == 0:
                instance.total_tokens = metrics.get("total_tokens", 0)
                instance.input_tokens = metrics.get("input_tokens", 0)
                instance.output_tokens = metrics.get("output_tokens", 0)

    # Instance phase event handlers

    def _handle_instance_workspace_preparing(self, event: Dict[str, Any]) -> None:
        """Handle workspace preparation event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        previous_activity = instance.current_activity
        instance.current_activity = "Preparing workspace..."
        instance.last_updated = datetime.now()
        logger.info(
            f"Instance {instance_id}: '{previous_activity}' -> 'Preparing workspace...'"
        )

    def _handle_instance_container_creating(self, event: Dict[str, Any]) -> None:
        """Handle container creation event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Creating container..."
        instance.last_updated = datetime.now()

    def _handle_instance_claude_starting(self, event: Dict[str, Any]) -> None:
        """Handle Claude starting event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Starting Claude..."
        instance.last_updated = datetime.now()

    def _handle_instance_result_collection(self, event: Dict[str, Any]) -> None:
        """Handle result collection event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Collecting results..."
        instance.last_updated = datetime.now()

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO timestamp string."""
        if not timestamp_str:
            return None
        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None


class EventFileWatcher(FileSystemEventHandler):
    """Watches events.jsonl file for changes."""

    def __init__(
        self,
        file_path: Path,
        callback: Callable[[str], None],
        initial_position: int = 0,
    ):
        """
        Initialize file watcher.

        Args:
            file_path: Path to events.jsonl
            callback: Function to call with new lines
            initial_position: Starting position in file
        """
        self.file_path = file_path
        self.callback = callback
        self._last_position = initial_position

    def on_modified(self, event):
        """Handle file modification."""
        if not isinstance(event, FileModifiedEvent):
            return
        if Path(event.src_path) != self.file_path:
            return

        try:
            with open(self.file_path, "r") as f:
                # Seek to last position
                f.seek(self._last_position)

                # Read new lines
                for line in f:
                    line = line.strip()
                    if line:
                        self.callback(line)

                # Update position
                self._last_position = f.tell()
        except (OSError, IOError) as e:
            logger.error(f"Error reading events file: {e}")


class AsyncEventStream:
    """Async event stream reader with file watching."""

    def __init__(self, event_processor: EventProcessor):
        """
        Initialize async event stream.

        Args:
            event_processor: Event processor to handle events
        """
        self.event_processor = event_processor
        self._observer: Optional[Observer] = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._shutdown = False

    async def start(self, events_file: Path, from_offset: int = 0) -> None:
        """
        Start watching events file.

        Args:
            events_file: Path to events.jsonl
            from_offset: Byte offset to start from
        """
        self._events_file = events_file
        self._last_position = from_offset

        logger.info(
            f"Starting event stream for file: {events_file}, offset: {from_offset}"
        )

        # Start event processing loop
        asyncio.create_task(self._process_events())

        # Read any existing events first if file exists
        if events_file.exists():
            await self._read_existing_events(events_file, from_offset)

        # Start file watcher for real-time updates
        self._start_file_watcher(events_file)

        # Also start polling loop as fallback (in case file watching fails)
        asyncio.create_task(self._polling_loop())

    async def stop(self) -> None:
        """Stop watching events."""
        self._shutdown = True

        if self._observer:
            self._observer.stop()
            self._observer.join()

    async def _read_existing_events(self, events_file: Path, from_offset: int) -> None:
        """Read existing events from file."""
        if not events_file.exists():
            logger.warning(f"Events file does not exist: {events_file}")
            return

        logger.info(
            f"Reading existing events from {events_file}, starting at offset {from_offset}"
        )
        try:
            with open(events_file, "r") as f:
                # Seek to offset
                f.seek(from_offset)

                # Read all existing lines
                lines_read = 0
                for line in f:
                    line = line.strip()
                    if line:
                        lines_read += 1
                        logger.debug(f"Reading existing event: {line[:100]}...")
                        await self._event_queue.put(line)

                # Update position for polling to continue from here
                self._last_position = f.tell()
                self.event_processor.state.last_event_offset = self._last_position
                logger.info(
                    f"Read {lines_read} existing events, new position: {self._last_position}"
                )
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.error(f"Error reading existing events: {e}")

    def _start_file_watcher(self, events_file: Path) -> None:
        """Start watching events file for changes."""

        def new_line_callback(line: str):
            # Put line in queue for async processing
            asyncio.create_task(self._event_queue.put(line))

        event_handler = EventFileWatcher(
            events_file, new_line_callback, self._last_position
        )

        self._observer = Observer()
        self._observer.schedule(event_handler, str(events_file.parent), recursive=False)
        self._observer.start()
        logger.info(
            f"Started file watcher for {events_file} at position {self._last_position}"
        )

    async def _polling_loop(self) -> None:
        """Poll the events file for new content."""
        logger.info(f"Starting polling loop for file: {self._events_file}")

        # Wait for file to exist first
        while not self._shutdown and not self._events_file.exists():
            logger.debug(f"Waiting for events file to exist: {self._events_file}")
            await asyncio.sleep(0.1)

        logger.info(
            f"Events file exists, starting to poll from position {self._last_position}"
        )

        while not self._shutdown:
            try:
                if self._events_file.exists():
                    # Use regular file operations instead of aiofiles for more reliable reading
                    try:
                        with open(self._events_file, "r") as f:
                            # Seek to last position
                            f.seek(self._last_position)

                            # Read new lines
                            lines_read = 0
                            while True:
                                line = f.readline()
                                if not line:
                                    break
                                line = line.strip()
                                if line:
                                    lines_read += 1
                                    logger.debug(f"Read event line: {line[:100]}...")
                                    await self._event_queue.put(line)
                                    # Yield control after each event to allow processing
                                    await asyncio.sleep(0)

                            # Update position
                            new_position = f.tell()
                            if new_position > self._last_position and lines_read > 0:
                                logger.info(
                                    f"Read {lines_read} lines, position {self._last_position} -> {new_position}"
                                )
                            self._last_position = new_position

                            # Update state
                            self.event_processor.state.last_event_offset = (
                                self._last_position
                            )
                    except (OSError, IOError) as e:
                        logger.error(f"Error reading events file: {e}")

                # Poll every 100ms
                await asyncio.sleep(0.1)

            except (OSError, IOError) as e:
                logger.error(f"Error polling events file: {e}")
                await asyncio.sleep(1.0)

    async def _process_events(self) -> None:
        """Process events from queue."""
        logger.info("Starting event processing loop")
        while not self._shutdown:
            try:
                # Get event with timeout to check shutdown
                line = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)

                # Parse and process event
                try:
                    event = json.loads(line)
                    logger.debug(f"Parsed event: {event.get('type', 'unknown')}")
                    self.event_processor.process_event(event)

                    # Yield control to allow UI to update
                    await asyncio.sleep(0)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in event: {e}, line: {line}")
                    self.event_processor.state.add_error(f"JSON error: {e}")
                except (AttributeError, TypeError, ValueError, KeyError) as e:
                    logger.error(f"Error processing event: {e}")
                    self.event_processor.state.add_error(f"Event error: {e}")

            except asyncio.TimeoutError:
                # This is normal - just checking for shutdown
                pass
