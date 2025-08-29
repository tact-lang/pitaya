"""
Dynamic strategy loader for file, module, or spec inputs.

Supports loading a Strategy subclass from:
- A Python file path: "path/to/strategy.py"
- A Python file path with class: "path/to/strategy.py:ClassName"
- A Python module path with class: "package.module:ClassName"
- A Python module path (single Strategy in module): "package.module"

Returns the Strategy class (not instance).
"""

from __future__ import annotations

import importlib.util
import importlib
import sys
import hashlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import Optional, Type

from .base import Strategy


def _load_module_from_file(path: Path) -> ModuleType:
    # Generate a unique, stable module name based on file path to avoid collisions
    h = hashlib.sha256(
        str(path.resolve()).encode("utf-8", errors="ignore")
    ).hexdigest()[:8]
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
    module: ModuleType,
    explicit_class: Optional[str] = None,
    file_stem: Optional[str] = None,
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


def load_strategy_from_module(
    module_path: str, class_name: Optional[str] = None
) -> Type[Strategy]:
    """Load a Strategy subclass from a Python module path.

    Args:
        module_path: Dotted module path (e.g., "examples.custom_simple")
        class_name: Optional class name (e.g., "MyStrategy"). If omitted,
                    loader discovers a single Strategy subclass or STRATEGY.

    Returns:
        Strategy class
    """
    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        raise ImportError(f"Cannot import module '{module_path}': {e}") from e
    return _discover_strategy_class(module, explicit_class=class_name)


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


_MOD_PATH_RE = None


def _is_module_path(s: str) -> bool:
    """Best-effort check whether a string looks like a dotted module path."""
    global _MOD_PATH_RE
    if _MOD_PATH_RE is None:
        import re as _re

        _MOD_PATH_RE = _re.compile(
            r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"
        )
    return bool(_MOD_PATH_RE.match(s))


def load_strategy(spec: str) -> Type[Strategy]:
    """Load a Strategy class from a file or module spec string.

    Supported forms:
      - file: "path/to/strategy.py"
      - file + class: "path/to/strategy.py:ClassName"
      - module + class: "package.module:ClassName"
      - module (single Strategy in module): "package.module"
    """
    # 1) File forms
    path, cls = parse_strategy_spec(spec)
    if path is not None:
        return load_strategy_from_file(path, cls)

    # 2) Module forms
    if ":" in spec:
        before, after = spec.split(":", 1)
        if _is_module_path(before):
            return load_strategy_from_module(before, after or None)
        raise ImportError(
            f"Unknown strategy spec: '{spec}'. Expected file.py[:Class] or module.path:Class."
        )

    if _is_module_path(spec):
        return load_strategy_from_module(spec, None)

    raise ValueError(
        f"Unknown strategy spec: '{spec}'. Use a built-in name, file.py[:Class], or module.path[:Class]."
    )
