"""
Runner plugins for different AI coding tools.
"""

from .claude_code import ClaudeCodePlugin
from .codex import CodexPlugin

# Registry of available plugins
AVAILABLE_PLUGINS = {
    "claude-code": ClaudeCodePlugin,
    "codex": CodexPlugin,
}
