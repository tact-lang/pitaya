"""
Model mapping loader with checksum support.

Loads models.yaml with friendly->concrete model IDs and computes a checksum for handshake.
Falls back to a small identity mapping when the file is absent.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import hashlib
import json
import yaml


def _compute_checksum(obj: dict) -> str:
    enc = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(enc.encode("utf-8")).hexdigest()


def load_model_mapping(models_path: Optional[Path] = None) -> Tuple[Dict[str, str], str]:
    """
    Load model mapping from models.yaml.

    Returns:
        (mapping, checksum)
    """
    if models_path is None:
        models_path = Path("models.yaml")

    if not models_path.exists():
        # Fallback identity mapping used when no mapping file is present
        mapping = {"sonnet": "sonnet", "haiku": "haiku", "opus": "opus"}
        return mapping, _compute_checksum({"models": mapping})

    with open(models_path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict) or "models" not in data or not isinstance(data["models"], dict):
        raise ValueError("Invalid models.yaml: expected top-level 'models' mapping")
    mapping = {str(k): str(v) for k, v in (data.get("models") or {}).items()}
    checksum = _compute_checksum({"models": mapping})
    return mapping, checksum


def resolve_model(name: str, mapping: Optional[Dict[str, str]] = None) -> str:
    if mapping is None:
        mapping, _ = load_model_mapping()
    if name not in mapping:
        raise KeyError(f"Unknown model alias: {name}")
    return mapping[name]
