"""Configuration validation and LLM construction without network calls."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from medical_multi_agent.config import llm
from medical_multi_agent.config.settings import Settings


@pytest.mark.parametrize("kwargs", [
    {"max_diagnosis_retries": -1},
    {"llm_timeout_seconds": 0},
    {"llm_timeout_seconds": -1},
    {"llm_timeout_seconds": float("inf")},
    {"llm_timeout_seconds": float("nan")},
])
def test_invalid_pipeline_configuration_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)


def test_settings_read_dotenv_and_allow_other_tools_variables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    (tmp_path / ".env").write_text("OPENAI_MODEL=test-model\nUNRELATED_TOOL_OPTION=value\n")
    assert Settings().openai_model == "test-model"


def test_thinking_options_require_explicit_opt_in():
    assert Settings(_env_file=None).llm_disable_thinking is False


def test_codex_backend_requires_explicit_model():
    with pytest.raises(ValidationError, match="CODEX_MODEL"):
        Settings(_env_file=None, llm_backend="codex", codex_model="")


def test_unknown_llm_backend_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_backend="unsupported")


def test_remote_backend_requires_api_key(monkeypatch):
    settings = Settings(_env_file=None, openai_api_key="", openai_base_url="")
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        llm.get_llm()


def test_public_custom_backend_requires_api_key(monkeypatch):
    settings = Settings(_env_file=None, openai_api_key="", openai_base_url="https://example.com/v1")
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        llm.get_llm()


def test_blank_process_base_url_uses_default_openai_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    settings = Settings(_env_file=None, openai_api_key="test-placeholder")
    monkeypatch.setattr(llm, "get_settings", lambda: settings)

    client = llm.get_llm()

    assert str(client.root_client.base_url) == "https://api.openai.com/v1/"
    assert str(client.root_async_client.base_url) == "https://api.openai.com/v1/"


def test_custom_base_url_is_honored_by_real_clients(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8001/v1")
    settings = Settings(_env_file=None, openai_api_key="test-placeholder")
    monkeypatch.setattr(llm, "get_settings", lambda: settings)

    client = llm.get_llm()

    assert str(client.root_client.base_url) == "http://localhost:8001/v1/"
    assert str(client.root_async_client.base_url) == "http://localhost:8001/v1/"


def test_local_backend_allows_placeholder_key_without_thinking_option(monkeypatch):
    settings = Settings(_env_file=None, openai_api_key="", openai_base_url="http://localhost:8001/v1")
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "ChatOpenAI", lambda **kwargs: SimpleNamespace(**kwargs))
    client = llm.get_llm()
    assert client.api_key == "local"
    assert client.base_url == "http://localhost:8001/v1"
    assert not hasattr(client, "extra_body")


def test_local_thinking_option_is_sent_when_enabled(monkeypatch):
    settings = Settings(
        _env_file=None,
        openai_base_url="http://localhost:8001/v1",
        llm_disable_thinking=True,
    )
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "ChatOpenAI", lambda **kwargs: SimpleNamespace(**kwargs))
    client = llm.get_llm()
    assert client.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
