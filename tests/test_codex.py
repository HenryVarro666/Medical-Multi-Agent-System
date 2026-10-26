"""Exercise the optional Codex transport through a local fake executable."""

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from medical_multi_agent.config import llm
from medical_multi_agent.config.settings import Settings


@pytest.fixture
def codex_module():
    from medical_multi_agent.config import codex

    codex.get_codex_version.cache_clear()
    return codex


@pytest.fixture
def fake_cli(tmp_path):
    def create(body):
        executable = tmp_path / "fake-codex"
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, subprocess, sys, time\n"
            "if sys.argv[1:] == ['--version']:\n"
            "    print('codex-cli 0.163.0')\n"
            "    raise SystemExit(0)\n"
            "args = sys.argv[1:]\n"
            "output = pathlib.Path(args[args.index('--output-last-message') + 1])\n"
            "human = sys.stdin.read()\n"
            + body,
            encoding="utf-8",
        )
        executable.chmod(0o700)
        return str(executable)

    return create


SUCCESS = """
text = json.dumps({'received': human})
output.write_text(text, encoding='utf-8')
print(json.dumps({'type': 'thread.started', 'thread_id': 'private-thread-id'}))
print(json.dumps({'type': 'turn.started'}))
print(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': text}}))
print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 12, 'output_tokens': 7}}))
"""


def model(codex_module, path, timeout=2):
    return codex_module.CodexChatModel(Settings(
        _env_file=None,
        llm_backend="codex",
        codex_model="gpt-6-astra",
        codex_reasoning_effort="ultra",
        codex_cli_path=path,
        llm_timeout_seconds=timeout,
    ))


def messages(text="Synthetic patient narrative"):
    return [SystemMessage(content="Original clinical system prompt\nReturn JSON."), HumanMessage(content=text)]


def test_factory_selects_codex_without_openai_credentials(monkeypatch):
    settings = Settings(_env_file=None, llm_backend="codex", codex_model="gpt-6-astra", openai_api_key="")
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    assert llm.get_llm().__class__.__name__ == "CodexChatModel"


def test_preserves_original_messages_and_isolates_cli_environment(codex_module, fake_cli, tmp_path, monkeypatch):
    inspection = tmp_path / "inspection.json"
    path = fake_cli(f"""
import tomllib
config = tomllib.loads(chr(10).join(args[index + 1] for index, arg in enumerate(args) if arg == '-c'))
pathlib.Path({str(inspection)!r}).write_text(json.dumps({{
    'argv': args, 'cwd': os.getcwd(), 'config': config,
    'system': pathlib.Path(config['model_instructions_file']).read_text(),
    'credential_names': [name for name in ('CODEX_API_KEY', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL') if name in os.environ],
}}))
""" + SUCCESS)
    for name in ("CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.setenv(name, "private-do-not-copy")
    with codex_module.capture_codex_calls() as calls:
        response = model(codex_module, path).invoke(messages())
    observed = json.loads(inspection.read_text())
    assert observed["system"] == messages()[0].content
    assert json.loads(response.content) == {"received": messages()[1].content}
    assert observed["credential_names"] == []
    assert observed["cwd"] != str(Path.cwd())
    assert not Path(observed["cwd"]).exists()
    assert observed["argv"][-1] == "-"
    for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral", "--skip-git-repo-check", "--json", "--strict-config"):
        assert flag in observed["argv"]
    assert observed["config"]["forced_login_method"] == "chatgpt"
    assert observed["config"]["model_reasoning_effort"] == "ultra"
    assert observed["config"]["features"]["shell_tool"] is False
    assert observed["config"]["features"]["skip_host_skill_discovery"] is True
    assert observed["config"]["features"]["unbounded_connection_retries"] is False
    assert observed["config"]["skills"]["max_context_tokens"] == 1
    assert len(calls) == 1
    call = calls[0]
    assert call["status"] == "completed"
    assert call["requested_model"] == "gpt-6-astra"
    assert call["transport"] == "saved_chatgpt"
    assert call["cli_version"] == "codex-cli 0.163.0"
    assert call["usage"] == {"input_tokens": 12, "output_tokens": 7}
    assert call["messages"][0]["content"] == messages()[0].content
    assert call["messages"][1]["content"] == messages()[1].content
    assert len(call["prompt_sha256"]) == len(call["output_sha256"]) == 64
    assert "turn.completed" in call["event_types"]
    assert call["tool_use"] == []
    trace = json.dumps(call)
    assert "private-thread-id" not in trace
    assert "private-do-not-copy" not in trace
    assert str(tmp_path) not in trace


@pytest.mark.parametrize(("body", "expected_error"), [
    ("sys.stderr.write('/private/auth.json secret-token')\n" + SUCCESS + "\nraise SystemExit(1)", "unsuccessfully"),
    ("output.write_text('{}')\nprint(json.dumps({'type': 'turn.failed', 'error': {'message': '/private/auth.json secret-token'}}))", "failed"),
    ("output.write_text('{}')\nprint(json.dumps({'type': 'turn.started'}))", "completion"),
    (SUCCESS.replace("'text': text", "'text': 'different final message'"), "match"),
    (SUCCESS.replace("text = json.dumps({'received': human})", "text = '[]'"), "JSON object"),
    (SUCCESS.replace("text = json.dumps({'received': human})", "text = 'not-json'"), "JSON object"),
])
def test_failed_calls_are_captured_without_raw_errors(codex_module, fake_cli, body, expected_error):
    path = fake_cli(body)
    with codex_module.capture_codex_calls() as calls:
        with pytest.raises(RuntimeError, match=expected_error):
            model(codex_module, path).invoke(messages())
    assert len(calls) == 1
    assert calls[0]["status"] == "failed"
    assert calls[0]["elapsed_seconds"] >= 0
    assert "secret-token" not in json.dumps(calls)
    assert "/private/auth.json" not in json.dumps(calls)


def test_rejects_tool_events_even_with_successful_final_output(codex_module, fake_cli):
    path = fake_cli("print(json.dumps({'type':'item.started','item':{'type':'command_execution','command':'private-command'}}))\n" + SUCCESS)
    with codex_module.capture_codex_calls() as calls:
        with pytest.raises(RuntimeError, match="tool"):
            model(codex_module, path).invoke(messages())
    assert calls[0]["tool_use"] == ["command_execution"]
    assert "private-command" not in json.dumps(calls)


def test_known_skill_catalog_omission_is_a_safe_startup_warning(codex_module, fake_cli):
    warning = (
        "Exceeded skills context budget. All skill descriptions were removed and "
        "140 additional skills were not included in the model-visible skills list."
    )
    body = f"print(json.dumps({{'type': 'item.completed', 'item': {{'type': 'error', 'message': {warning!r}}}}}))\n"
    path = fake_cli(body + SUCCESS)
    with codex_module.capture_codex_calls() as calls:
        response = model(codex_module, path).invoke(messages())
    assert json.loads(response.content)["received"] == messages()[1].content
    assert calls[0]["status"] == "completed"
    assert calls[0]["warnings"] == ["skill_catalog_omitted"]
    assert calls[0]["tool_use"] == []
    assert warning not in json.dumps(calls)


def test_unknown_error_item_is_rejected_without_leaking_its_message(codex_module, fake_cli):
    body = "print(json.dumps({'type': 'item.completed', 'item': {'type': 'error', 'message': '/private/auth.json secret-token'}}))\n"
    path = fake_cli(body + SUCCESS)
    with codex_module.capture_codex_calls() as calls:
        with pytest.raises(RuntimeError, match="error item"):
            model(codex_module, path).invoke(messages())
    assert calls[0]["status"] == "failed"
    assert calls[0]["tool_use"] == []
    assert "/private/auth.json" not in json.dumps(calls)
    assert "secret-token" not in json.dumps(calls)


def test_timeout_retains_already_written_output(codex_module, fake_cli):
    path = fake_cli("output.write_text('{\"ready\":true}')\ntime.sleep(60)")
    with codex_module.capture_codex_calls() as calls:
        with pytest.raises(RuntimeError, match="timed out"):
            model(codex_module, path, timeout=.25).invoke(messages())
    assert calls[0]["output_text"] == '{"ready":true}'
    assert len(calls[0]["output_sha256"]) == 64
    assert calls[0]["status"] == "failed"
    assert calls[0]["error"] == "Codex CLI request timed out."


def test_missing_executable_has_safe_configuration_error(codex_module, tmp_path):
    path = str(tmp_path / "private" / "missing-codex")
    with codex_module.capture_codex_calls() as calls:
        with pytest.raises(llm.LLMConfigurationError, match="Codex CLI") as error:
            model(codex_module, path).invoke(messages())
    assert str(tmp_path) not in str(error.value)
    assert calls[0]["status"] == "failed"
    assert str(tmp_path) not in json.dumps(calls)


@pytest.mark.skipif(os.name != "posix", reason="Process group cleanup is POSIX specific")
def test_timeout_terminates_cli_process_group(codex_module, fake_cli, tmp_path):
    pid_file = tmp_path / "child.pid"
    path = fake_cli(f"""
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))
time.sleep(60)
""")
    child_pid = None
    try:
        with codex_module.capture_codex_calls() as calls:
            with pytest.raises(RuntimeError, match="timed out"):
                model(codex_module, path, timeout=.25).invoke(messages())
        child_pid = int(pid_file.read_text())
        for _ in range(30):
            state = subprocess.run(["ps", "-o", "stat=", "-p", str(child_pid)], capture_output=True, text=True).stdout.strip()
            if not state or state.startswith("Z"):
                break
            time.sleep(.02)
        assert not state or state.startswith("Z")
        assert calls[0]["status"] == "failed"
        assert calls[0]["error"] == "Codex CLI request timed out."
    finally:
        if child_pid:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_capture_isolated_between_threads_and_nested_contexts(codex_module, fake_cli):
    path = fake_cli(SUCCESS)

    def run(text):
        with codex_module.capture_codex_calls() as calls:
            model(codex_module, path).invoke(messages(text))
        return calls

    with ThreadPoolExecutor(max_workers=2) as executor:
        traces = list(executor.map(run, ["first synthetic case", "second synthetic case"]))
    assert [trace[0]["messages"][1]["content"] for trace in traces] == ["first synthetic case", "second synthetic case"]
    with codex_module.capture_codex_calls() as outer:
        model(codex_module, path).invoke(messages("outer"))
        with codex_module.capture_codex_calls() as inner:
            model(codex_module, path).invoke(messages("inner"))
        model(codex_module, path).invoke(messages("outer again"))
    assert [call["messages"][1]["content"] for call in outer] == ["outer", "outer again"]
    assert [call["messages"][1]["content"] for call in inner] == ["inner"]
