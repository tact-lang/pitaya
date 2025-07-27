"""
Runner plugins for different AI coding tools.
"""

from .claude_code import ClaudeCodePlugin

# Registry of available plugins
AVAILABLE_PLUGINS = {
    "claude-code": ClaudeCodePlugin,
}
