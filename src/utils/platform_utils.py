"""
Platform-specific utilities for cross-platform compatibility.

Handles differences between Windows, macOS, and Linux, particularly
for path handling and Docker-related operations.
"""

import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional, Tuple


logger = logging.getLogger(__name__)


def get_platform_info() -> dict:
    """
    Get detailed platform information.

    Returns:
        Dictionary with platform details
    """
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "is_windows": platform.system() == "Windows",
        "is_macos": platform.system() == "Darwin",
        "is_linux": platform.system() == "Linux",
    }


def is_wsl() -> bool:
    """
    Check if running under Windows Subsystem for Linux.

    Returns:
        True if running under WSL, False otherwise
    """
    if platform.system() != "Linux":
        return False

    try:
        with open("/proc/version", "r") as f:
            version_info = f.read().lower()
            return "microsoft" in version_info or "wsl" in version_info
    except (OSError, IOError):
        return False


def normalize_path_for_docker(
    path: Path, is_windows_docker: Optional[bool] = None
) -> str:
    """
    Normalize a path for use with Docker.

    On Windows, Docker expects Unix-style paths, but the format depends
    on whether we're using Docker Desktop or Docker in WSL.

    Args:
        path: Path to normalize
        is_windows_docker: Whether Docker is running on Windows (auto-detected if None)

    Returns:
        Normalized path string suitable for Docker
    """
    # Resolve symlinks to avoid macOS /tmp -> /private/tmp issues
    try:
        path = Path(os.path.realpath(str(path))).absolute()
    except Exception:
        path = Path(path).absolute()

    # Auto-detect if not specified
    if is_windows_docker is None:
        is_windows_docker = platform.system() == "Windows" and not is_wsl()

    if is_windows_docker:
        # Windows Docker Desktop expects paths like /c/Users/...
        path_str = str(path).replace("\\", "/")

        # Convert drive letter to Docker format
        if len(path_str) >= 2 and path_str[1] == ":":
            drive_letter = path_str[0].lower()
            path_str = f"/{drive_letter}{path_str[2:]}"

        return path_str
    else:
        # Unix-like systems (including WSL) use normal paths
        # On macOS, prefer /private/tmp over /tmp for Docker Desktop file sharing
        pstr = str(path)
        try:
            if platform.system() == "Darwin" and pstr.startswith("/tmp/"):
                pstr = pstr.replace("/tmp/", "/private/tmp/", 1)
        except Exception:
            pass
        return pstr


def get_docker_socket_path() -> str:
    """
    Get the appropriate Docker socket path for the current platform.

    Returns:
        Docker socket path
    """
    if platform.system() == "Windows":
        # Windows Docker Desktop
        return "//./pipe/docker_engine"
    else:
        # Unix-like systems
        return "/var/run/docker.sock"


def validate_docker_setup() -> Tuple[bool, Optional[str]]:
    """
    Validate Docker is properly set up for the current platform.

    Returns:
        Tuple of (is_valid, error_message)
    """
    system = platform.system()

    try:
        # Try to run docker info
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            if system == "Windows":
                return False, (
                    "Docker is not running or not accessible.\n"
                    "Please ensure Docker Desktop is running."
                )
            elif is_wsl():
                return False, (
                    "Docker is not accessible from WSL.\n"
                    "You can either:\n"
                    "1. Install Docker inside WSL2: https://docs.docker.com/engine/install/ubuntu/\n"
                    "2. Use Docker Desktop with WSL2 backend: https://docs.docker.com/desktop/wsl/"
                )
            else:
                return False, (
                    "Docker daemon is not running or not accessible.\n"
                    "Please ensure Docker is installed and the daemon is running."
                )

        return True, None

    except FileNotFoundError:
        if system == "Windows":
            return False, (
                "Docker command not found.\n"
                "Please install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"
            )
        else:
            return False, (
                "Docker command not found.\n"
                "Please install Docker: https://docs.docker.com/engine/install/"
            )
    except subprocess.TimeoutExpired:
        return False, "Docker command timed out. The Docker daemon may be unresponsive."
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"Error checking Docker: {e}"


def get_temp_dir() -> Path:
    """
    Get appropriate temporary directory for the current platform.

    Returns:
        Path to temporary directory
    """
    if platform.system() == "Windows":
        # Use Windows temp directory
        temp_dir = Path(os.environ.get("TEMP", os.environ.get("TMP", "C:\\Temp")))
    else:
        # Use /tmp on Unix-like systems
        temp_dir = Path("/tmp")

    return temp_dir


def convert_windows_path_to_wsl(windows_path: str) -> str:
    """
    Convert a Windows path to WSL path format.

    Args:
        windows_path: Windows-style path (e.g., C:\\Users\\...)

    Returns:
        WSL-style path (e.g., /mnt/c/Users/...)
    """
    # Replace backslashes with forward slashes
    path = windows_path.replace("\\", "/")

    # Convert drive letter
    if len(path) >= 2 and path[1] == ":":
        drive_letter = path[0].lower()
        path = f"/mnt/{drive_letter}{path[2:]}"

    return path


def get_platform_recommendations() -> list[str]:
    """
    Get platform-specific recommendations for optimal operation.

    Returns:
        List of recommendation strings
    """
    recommendations = []
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

    # Check temp directory accessibility
    temp_dir = get_temp_dir()
    if not temp_dir.exists() or not os.access(temp_dir, os.W_OK):
        recommendations.append(
            f"Temporary directory {temp_dir} is not writable. "
            "This may cause issues with workspace creation."
        )

    return recommendations
