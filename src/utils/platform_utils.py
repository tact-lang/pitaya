"""Cross-platform helpers for Pitaya."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = [
    "convert_windows_path_to_wsl",
    "get_docker_socket_path",
    "get_platform_info",
    "get_platform_recommendations",
    "get_temp_dir",
    "is_wsl",
    "normalize_path_for_docker",
    "validate_docker_setup",
]


def get_platform_info() -> Dict[str, object]:
    """Return basic identifiers for the current platform."""
    system = platform.system()
    return {
        "system": system,
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "is_windows": system == "Windows",
        "is_macos": system == "Darwin",
        "is_linux": system == "Linux",
    }


def is_wsl() -> bool:
    """Return ``True`` when running inside Windows Subsystem for Linux."""
    if platform.system() != "Linux":
        return False

    try:
        with open("/proc/version", "r", encoding="utf-8") as handle:
            version_info = handle.read().lower()
    except OSError:
        return False

    return "microsoft" in version_info or "wsl" in version_info


def normalize_path_for_docker(
    path: Path, *, is_windows_docker: Optional[bool] = None
) -> str:
    """Return a Docker-compatible path for the host."""
    try:
        normalized = Path(os.path.realpath(str(path))).absolute()
    except OSError:
        normalized = Path(path).absolute()

    if is_windows_docker is None:
        is_windows_docker = platform.system() == "Windows" and not is_wsl()

    if is_windows_docker:
        path_str = str(normalized).replace("\\", "/")
        if len(path_str) >= 2 and path_str[1] == ":":
            drive_letter = path_str[0].lower()
            path_str = f"/{drive_letter}{path_str[2:]}"
        return path_str

    path_str = str(normalized)
    if platform.system() == "Darwin" and path_str.startswith("/tmp/"):
        return path_str.replace("/tmp/", "/private/tmp/", 1)
    return path_str


def get_docker_socket_path() -> str:
    """Return the default Docker socket path for the platform."""
    if platform.system() == "Windows":
        return "//./pipe/docker_engine"
    return "/var/run/docker.sock"


def validate_docker_setup() -> Tuple[bool, Optional[str]]:
    """Check whether Docker is reachable and return (is_ready, message)."""
    system = platform.system()
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=10
        )
    except FileNotFoundError:
        return False, _docker_missing_message(system)
    except subprocess.TimeoutExpired:
        return False, "Docker command timed out. The Docker daemon may be unresponsive."
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Error checking Docker: {exc}"

    if result.returncode == 0:
        return True, None

    if system == "Windows":
        return (
            False,
            "Docker is not running or not accessible.\nPlease ensure Docker Desktop is running.",
        )
    if is_wsl():
        return (
            False,
            "Docker is not accessible from WSL.\n"
            "You can either:\n"
            "1. Install Docker inside WSL2: https://docs.docker.com/engine/install/ubuntu/\n"
            "2. Use Docker Desktop with WSL2 backend: https://docs.docker.com/desktop/wsl/",
        )
    return (
        False,
        "Docker daemon is not running or not accessible.\n"
        "Please ensure Docker is installed and the daemon is running.",
    )


def get_temp_dir() -> Path:
    """Return a writable temporary directory."""
    if platform.system() == "Windows":
        return Path(os.environ.get("TEMP") or os.environ.get("TMP") or "C:/Temp")
    return Path("/tmp")


def convert_windows_path_to_wsl(windows_path: str) -> str:
    """Convert ``C:\\Users\\...`` to ``/mnt/c/Users/...`` for WSL."""
    path = windows_path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive_letter = path[0].lower()
        path = f"/mnt/{drive_letter}{path[2:]}"
    return path


def get_platform_recommendations() -> List[str]:
    """Return platform-specific tips for smoother operation."""
    recommendations: List[str] = []
    system = platform.system()

    if system == "Windows":
        if not is_wsl():
            recommendations.append(
                "Consider using WSL2 for better performance and compatibility. "
                "Install WSL2: https://learn.microsoft.com/en-us/windows/wsl/install"
            )
        else:
            recommendations.append(
                "Running under WSL2 - ensure Docker is properly configured for WSL2 backend."
            )

    temp_dir = get_temp_dir()
    if not temp_dir.exists() or not os.access(temp_dir, os.W_OK):
        recommendations.append(
            f"Temporary directory {temp_dir} is not writable. "
            "This may cause issues with workspace creation."
        )

    return recommendations


def _docker_missing_message(system: str) -> str:
    if system == "Windows":
        return (
            "Docker command not found.\n"
            "Please install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"
        )
    return (
        "Docker command not found.\n"
        "Please install Docker: https://docs.docker.com/engine/install/"
    )
