"""Environment diagnostics (doctor) for the Pitaya CLI.

Splits checks into small helpers for clarity and testability.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console

from ...config import load_dotenv_config, load_env_config
from ...utils.platform_utils import get_temp_dir, validate_docker_setup

__all__ = ["run_doctor"]


def _print_rows(rows: list[tuple[str, str, str, list[str]]]) -> int:
    ok = all(status != "✗" for status, *_ in rows)
    for status, title, message, tries in rows:
        if status in ("✓", "✗"):
            print(f"{status} {title}: {message}")
        else:
            print(f"i {title}: {message}")
        if status == "✗" and tries:
            print("Try:")
            for t in tries[:3]:
                print(f"  • {t}")
    return 0 if ok else 1


def _check_docker(rows: list[tuple[str, str, str, list[str]]]) -> None:
    try:
        valid, _ = validate_docker_setup()
        rows.append(
            (
                "✓" if valid else "✗",
                "docker",
                "ok" if valid else "cannot connect to docker daemon",
                (
                    ["start Docker", "check $DOCKER_HOST", "run: docker info"]
                    if not valid
                    else []
                ),
            )
        )
    except (OSError, RuntimeError) as e:  # pragma: no cover - environment dependent
        rows.append(
            ("✗", "docker", str(e), ["ensure Docker installed", "run: docker info"])
        )


def _check_disk(rows: list[tuple[str, str, str, list[str]]]) -> None:
    try:
        stat = shutil.disk_usage(str(Path.cwd()))
        free_gb = stat.free / (1024**3)
        if free_gb >= 20:
            rows.append(("✓", "disk", f"{free_gb:.1f}GB free", []))
        else:
            rows.append(
                (
                    "✗",
                    "disk",
                    f"insufficient disk space: {free_gb:.1f}GB free (<20GB)",
                    ["free space on this volume", "move repo to larger disk"],
                )
            )
    except OSError as e:  # pragma: no cover - platform dependent
        rows.append(("i", "disk", f"could not check: {e}", []))


def _check_repo(
    base_branch: str | None, repo: Path, rows: list[tuple[str, str, str, list[str]]]
) -> None:
    try:
        if not (repo / ".git").exists():
            rows.append(
                ("✗", "repo", f"not a git repo: {repo}", ["git init", "verify path"])
            )
            return
        base = base_branch or "main"
        rc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", base], capture_output=True
        )
        if rc.returncode == 0:
            rows.append(("✓", "base_branch", base, []))
        else:
            rows.append(
                (
                    "✗",
                    "base_branch",
                    f"not found: '{base}'",
                    [
                        "git fetch origin --prune",
                        "verify branch name",
                        "git branch --all",
                    ],
                )
            )
    except (subprocess.SubprocessError, OSError) as e:  # pragma: no cover
        rows.append(("i", "repo", f"check failed: {e}", []))


def _check_temp(rows: list[tuple[str, str, str, list[str]]]) -> None:
    try:
        td = get_temp_dir()
        td.mkdir(parents=True, exist_ok=True)
        test = td / "_pitaya_doctor.tmp"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        rows.append(("✓", "temp", str(td), []))
    except OSError as e:  # pragma: no cover
        rows.append(
            ("✗", "temp", f"not writable: {e}", ["adjust permissions", "set TMPDIR"])
        )


def _check_auth(rows: list[tuple[str, str, str, list[str]]]) -> None:
    env = load_env_config()
    dotenv = load_dotenv_config()
    ok = any(
        (
            env.get("runner", {}).get("oauth_token"),
            dotenv.get("runner", {}).get("oauth_token"),
            env.get("runner", {}).get("api_key"),
            dotenv.get("runner", {}).get("api_key"),
        )
    )
    if ok:
        rows.append(("✓", "auth", "credentials found", []))
    else:
        rows.append(
            (
                "✗",
                "auth",
                "no credentials",
                ["set CLAUDE_CODE_OAUTH_TOKEN", "or set ANTHROPIC_API_KEY"],
            )
        )


async def run_doctor(console: Console, args: Any) -> int:
    """Run system checks and print friendly advice.

    Returns 0 on success, 1 if any failures are detected.
    """
    rows: list[tuple[str, str, str, list[str]]] = []
    _check_docker(rows)
    _check_disk(rows)
    _check_repo(
        getattr(args, "base_branch", None),
        Path(getattr(args, "repo", Path.cwd())),
        rows,
    )
    _check_temp(rows)
    _check_auth(rows)
    # Platform hints are non-fatal and printed by helper modules elsewhere.
    return _print_rows(rows)
