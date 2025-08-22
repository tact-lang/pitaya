"""
Dynamic strategy loader for file or spec inputs.

Supports loading a Strategy subclass from:
- A Python file path: "path/to/strategy.py"
- A Python file path with class: "path/to/strategy.py:ClassName"

Returns the Strategy class (not instance).
"""

from __future__ import annotations

import importlib.util
import sys
import hashlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import Optional, Type

from .base import Strategy


def _load_module_from_file(path: Path) -> ModuleType:
    # Generate a unique, stable module name based on file path to avoid collisions
    h = hashlib.sha256(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:8]
    mod_name = f"pitaya_strategy_{path.stem}_{h}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules so decorators (e.g., dataclass) can resolve module
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _discover_strategy_class(
    module: ModuleType, explicit_class: Optional[str] = None, file_stem: Optional[str] = None
) -> Type[Strategy]:
    # If explicit class name given, resolve it first
    if explicit_class:
        obj = getattr(module, explicit_class, None)
        if obj is None:
            raise ImportError(
                f"Class '{explicit_class}' not found in module {module.__name__}"
            )
        if inspect.isclass(obj) and issubclass(obj, Strategy):
            return obj
        raise TypeError(f"{explicit_class} is not a Strategy subclass")

    # Prefer STRATEGY attribute if provided (instance or class)
    if hasattr(module, "STRATEGY"):
        s = getattr(module, "STRATEGY")
        if inspect.isclass(s) and issubclass(s, Strategy):
            return s
        if (
            not inspect.isclass(s)
            and isinstance(getattr(s, "__class__", None), type)
            and issubclass(s.__class__, Strategy)
        ):
            return s.__class__

    # Else, discover Strategy subclasses defined in this module
    candidates: list[type[Strategy]] = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        try:
            if (
                issubclass(obj, Strategy)
                and obj is not Strategy
                and obj.__module__ == module.__name__
            ):
                candidates.append(obj)
        except Exception:
            continue

    if candidates:
        if len(candidates) > 1:
            names = ", ".join(c.__name__ for c in candidates)
            raise ImportError(
                f"Multiple Strategy subclasses found: {names}. Specify with ':ClassName'."
            )
        return candidates[0]

    raise ImportError(
        "No Strategy subclass found in module. Define STRATEGY or a Strategy subclass."
    )


def load_strategy_from_file(
    path: str | Path, class_name: Optional[str] = None
) -> Type[Strategy]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Strategy file not found: {p}")
    module = _load_module_from_file(p)
    return _discover_strategy_class(module, explicit_class=class_name, file_stem=p.stem)


def parse_strategy_spec(spec: str) -> tuple[Optional[Path], Optional[str]]:
    """Parse a strategy spec into (file_path, class_name).

    Accepts:
      - "path/to/file.py"
      - "path/to/file.py:ClassName"
    Returns (Path, class) if looks like a file spec, else (None, None).
    """
    if ":" in spec:
        before, after = spec.split(":", 1)
        if before.endswith(".py"):
            return Path(before), after or None
        return None, None
    if spec.endswith(".py"):
        return Path(spec), None
    return None, None
