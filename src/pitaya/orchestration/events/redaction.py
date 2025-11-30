"""Redaction utilities for event payloads."""

from __future__ import annotations

import base64
import re
from typing import Any, List, Pattern

_SENSITIVE_KEYS = (
    "token",
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "cookie",
)

_TOKEN_USAGE_KEYS = {
    "tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cached_input_tokens",
    "reasoning_output_tokens",
    "tokens_used",
}

_PATTERNS = [
    re.compile(r"(?i)(authorization\s*:\s*Bearer)\s+[A-Za-z0-9._\-]+"),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"gh[opsu]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token)\s*[:=]\s*[A-Za-z0-9\-]{8,}"),
    re.compile(r"(?i)Basic\s+[A-Za-z0-9+/=]{20,}"),
]

_JWT_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)(?![A-Za-z0-9_-])"
)


def _redact_jwts(text: str) -> str:
    """Replace likely JWTs with a redaction marker without touching regular hostnames."""

    def _decode(segment: str) -> bytes | None:
        padding = "=" * (-len(segment) % 4)
        try:
            return base64.urlsafe_b64decode(segment + padding)
        except Exception:
            return None

    def _maybe_redact(match: re.Match[str]) -> str:
        token = match.group(1)
        parts = token.split(".")
        if len(parts) != 3:
            return token
        header = _decode(parts[0])
        payload = _decode(parts[1])
        if not header or not payload:
            return token
        if not header.strip().startswith(b"{"):
            return token
        if not payload.strip():
            return token
        return "[REDACTED]"

    return _JWT_CANDIDATE_RE.sub(_maybe_redact, text)


class EventRedactor:
    """Redact sensitive values in event payloads."""

    def __init__(self, custom_patterns: List[str] | None = None) -> None:
        self._custom_redaction_patterns: List[Pattern] = []
        if custom_patterns:
            self.set_custom_patterns(custom_patterns)

    def set_custom_patterns(self, patterns: List[str]) -> None:
        compiled: List[Pattern] = []
        for pat in patterns or []:
            try:
                compiled.append(re.compile(pat))
            except Exception:
                continue
        self._custom_redaction_patterns = compiled

    def sanitize(self, obj: Any) -> Any:
        """Redact secrets recursively: field-name redaction + pattern sweep."""
        try:
            if isinstance(obj, dict):
                return {
                    k: ("[REDACTED]" if self._is_sensitive_key(k) else self.sanitize(v))
                    for k, v in obj.items()
                }
            if isinstance(obj, list):
                return [self.sanitize(v) for v in obj]
            if isinstance(obj, str):
                return self._scrub_string(obj)
            return obj
        except Exception:
            return obj

    def _scrub_string(self, value: str) -> str:
        out = value
        for pat in list(_PATTERNS) + list(self._custom_redaction_patterns):
            try:
                if pat.pattern.lower().startswith("(?i)(authorization"):
                    out = pat.sub(r"\1 [REDACTED]", out)
                else:
                    out = pat.sub("[REDACTED]", out)
            except Exception:
                continue
        return _redact_jwts(out)

    @staticmethod
    def _is_sensitive_key(key: Any) -> bool:
        try:
            kl = str(key).lower()
        except Exception:
            return False
        if kl in _TOKEN_USAGE_KEYS:
            return False
        return any(s in kl for s in _SENSITIVE_KEYS)
