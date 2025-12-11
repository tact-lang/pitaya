import pytest

from pitaya.runner.plugins.codex import (
    CodexPlugin,
    _collect_codex_env,
    _select_provider_env_key,
)


def test_selects_codex_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEX_ENV_KEY", raising=False)
    monkeypatch.setenv("CODEX_API_KEY", "codex-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert _select_provider_env_key() == "CODEX_API_KEY"
    env = _collect_codex_env({})
    assert env["CODEX_API_KEY"] == "codex-key"


@pytest.mark.asyncio
async def test_build_command_resume_includes_subcommand() -> None:
    plugin = CodexPlugin()
    cmd = await plugin.build_command(
        prompt="",
        model="gpt-5.1-codex",
        session_id="sess-123",
    )
    assert "resume" in cmd
    idx = cmd.index("resume")
    assert cmd[idx + 1] == "sess-123"
    # No trailing empty prompt when omitted
    assert cmd[-1] == "sess-123"


@pytest.mark.asyncio
async def test_build_command_resume_with_prompt() -> None:
    plugin = CodexPlugin()
    cmd = await plugin.build_command(
        prompt="do thing",
        model="gpt-5.1-codex",
        session_id="sess-123",
    )
    assert any("features.web_search_request=true" in arg for arg in cmd)
    assert cmd[-1] == "do thing"
    assert cmd[-2] == "sess-123"


@pytest.mark.asyncio
async def test_build_command_forces_env_provider_for_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_ENV_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    plugin = CodexPlugin()
    cmd = await plugin.build_command(
        prompt="hello",
        model="openai/gpt-5.1",
        session_id=None,
    )

    # Should override provider even when using OPENAI_API_KEY so Codex doesn't
    # require interactive OpenAI auth.
    assert any("model_provider=pitaya_env" in arg for arg in cmd)
    assert any("model_providers.pitaya_env" in arg for arg in cmd)
    assert any('wire_api="responses"' in arg for arg in cmd)
