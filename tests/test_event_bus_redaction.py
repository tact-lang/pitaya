import base64


from src.orchestration.event_bus import EventBus


def _make_jwt(payload: bytes = b'{"sub":"123"}') -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b"=")
    body = base64.urlsafe_b64encode(payload).rstrip(b"=")
    signature = base64.urlsafe_b64encode(b'signature').rstrip(b"=")
    return b".".join((header, body, signature)).decode("ascii")


def test_sanitize_preserves_regular_urls() -> None:
    bus = EventBus()
    url = "https://en.wikipedia.org/wiki/Merkle_tree"
    sanitized = bus._sanitize(url)
    assert sanitized == url


def test_sanitize_redacts_jwt_tokens() -> None:
    bus = EventBus()
    token = _make_jwt()
    sanitized = bus._sanitize(f"Authorization: Bearer {token}")
    assert "[REDACTED]" in sanitized
    assert token not in sanitized
