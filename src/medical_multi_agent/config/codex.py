"""Optional synchronous Codex CLI transport using the saved ChatGPT login.

Each invocation runs with the original stage instructions and narrative in an
isolated temporary directory. Only final JSON and allowlisted provenance are
captured; raw CLI streams and authentication files are never archived here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from .llm import LLMConfigurationError
from .settings import Settings

_CAPTURE: ContextVar[list[dict] | None] = ContextVar("medical_codex_calls", default=None)


@contextmanager
def capture_codex_calls():
    """Collect only invocations made in this context, including failed calls."""
    calls: list[dict] = []
    token = _CAPTURE.set(calls)
    try:
        yield calls
    finally:
        _CAPTURE.reset(token)


@lru_cache(maxsize=8)
def get_codex_version(cli_path: str = "codex") -> str:
    """Read a bounded, non-sensitive version string without invoking a model."""
    try:
        result = subprocess.run(
            [cli_path, "--version"], capture_output=True, text=True,
            timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        raise LLMConfigurationError(
            "Codex CLI is unavailable; install it or set CODEX_CLI_PATH."
        ) from None
    version = result.stdout.strip()
    if not re.fullmatch(r"codex-cli \d+\.\d+\.\d+(?:[-+][A-Za-z0-9._-]+)?", version):
        raise LLMConfigurationError("Unable to identify the Codex CLI version.")
    return version


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _capture_output(path: Path, call: dict) -> None:
    """Salvage written output even when the CLI exits or times out unsuccessfully."""
    try:
        output = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return
    call["output_text"] = output
    call["output_sha256"] = _sha256(output)


def _child_environment() -> dict[str, str]:
    # Retain login discovery and ordinary platform/network requirements without
    # forwarding API credentials, model overrides, or the parent Codex session.
    allowed = {
        "PATH", "HOME", "USER", "LOGNAME", "CODEX_HOME", "TMPDIR", "TMP", "TEMP",
        "SYSTEMROOT", "WINDIR", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "LANG", "LC_ALL", "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def _stop_process(process: subprocess.Popen) -> None:
    """Terminate the invocation and, on POSIX, its whole process group."""
    def send(terminate):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM if terminate else signal.SIGKILL)
            elif terminate:
                process.terminate()
            else:
                process.kill()
        except ProcessLookupError:
            pass

    send(True)
    try:
        process.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        send(False)
        process.communicate()


def _read_events(stdout: str, call: dict) -> str | None:
    final_message = None
    completions = 0
    failed = False
    error_item = False
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            event_type = event["type"]
            if not isinstance(event_type, str):
                raise ValueError
        except (ValueError, TypeError, KeyError):
            raise RuntimeError("Codex CLI returned an invalid event stream.") from None
        if event_type not in call["event_types"]:
            call["event_types"].append(event_type)
        if event_type in {"turn.failed", "error"}:
            failed = True
        if event_type == "turn.completed":
            completions += 1
            usage = event.get("usage") or {}
            keys = {"input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens", "total_tokens"}
            if isinstance(usage, dict):
                call["usage"] = {
                    key: value for key, value in usage.items()
                    if key in keys and isinstance(value, int) and not isinstance(value, bool) and value >= 0
                }
        if event_type.startswith("item."):
            item = event.get("item")
            if not isinstance(item, dict):
                raise RuntimeError("Codex CLI returned an invalid event stream.")
            item_type = item.get("type")
            if item_type == "error":
                message = item.get("message")
                omitted_catalog = (
                    r"Exceeded skills context budget\. All skill descriptions were removed and "
                    r"\d+ additional skills were not included in the model-visible skills list\."
                )
                if isinstance(message, str) and re.fullmatch(omitted_catalog, message):
                    if "skill_catalog_omitted" not in call["warnings"]:
                        call["warnings"].append("skill_catalog_omitted")
                else:
                    error_item = True
            elif item_type not in {"agent_message", "reasoning"}:
                kind = item_type if isinstance(item_type, str) else "unknown"
                if kind not in call["tool_use"]:
                    call["tool_use"].append(kind)
            if event_type == "item.completed" and item_type == "agent_message":
                final_message = item.get("text")
    if call["tool_use"]:
        raise RuntimeError("Codex CLI attempted a tool call; this transport permits text only.")
    if error_item:
        raise RuntimeError("Codex CLI reported an error item.")
    if failed:
        raise RuntimeError("Codex CLI reported a failed turn.")
    if completions != 1:
        raise RuntimeError("Codex CLI did not report exactly one successful completion.")
    return final_message


class CodexChatModel:
    """Small .invoke adapter for the pipeline's system/human message pairs."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def invoke(self, messages) -> SimpleNamespace:
        started = time.perf_counter()
        call = {
            "backend": "codex", "transport": "saved_chatgpt",
            "requested_model": self.settings.codex_model,
            "reasoning_effort": self.settings.codex_reasoning_effort,
            "cli_version": None, "messages": [], "prompt_sha256": None,
            "output_text": "", "output_sha256": _sha256(""), "usage": {},
            "elapsed_seconds": 0.0, "status": "failed", "tool_use": [],
            "event_types": [], "warnings": [], "error": None,
        }
        capture = _CAPTURE.get()
        if capture is not None:
            capture.append(call)
        try:
            system, human = [], []
            for message in messages:
                role = getattr(message, "type", None)
                content = getattr(message, "content", None)
                if role not in {"system", "human"} or not isinstance(content, str):
                    raise RuntimeError("Codex transport requires text system and human messages.")
                call["messages"].append({"role": role, "content": content, "sha256": _sha256(content)})
                (system if role == "system" else human).append(content)
            if not system or not human:
                raise RuntimeError("Codex transport requires text system and human messages.")
            call["prompt_sha256"] = _sha256(json.dumps(call["messages"], ensure_ascii=False, sort_keys=True))
            call["cli_version"] = get_codex_version(self.settings.codex_cli_path)
            with TemporaryDirectory(prefix="medical-codex-") as directory:
                instructions = Path(directory) / "instructions.md"
                last_message = Path(directory) / "last-message.txt"
                instructions.write_text("\n\n".join(system), encoding="utf-8")
                args = [
                    self.settings.codex_cli_path, "exec", "--strict-config", "--ignore-user-config", "--ignore-rules",
                    "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
                    "--json", "--color", "never", "-m", self.settings.codex_model,
                    "--output-last-message", str(last_message),
                ]
                config = {
                    "model_instructions_file": str(instructions),
                    "model_reasoning_effort": self.settings.codex_reasoning_effort,
                    "forced_login_method": "chatgpt", "approval_policy": "never",
                    "suppress_unstable_features_warning": True,
                    "web_search": "disabled", "project_doc_max_bytes": 0,
                    "skills.max_context_tokens": 1, "features.skip_host_skill_discovery": True,
                    "memories.use_memories": False, "memories.generate_memories": False,
                }
                for feature in (
                    "shell_tool", "shell_snapshot", "apps", "memories", "multi_agent",
                    "plugins", "hooks", "browser_use", "computer_use", "image_generation",
                    "in_app_browser", "view_image", "sleep_tool", "unbounded_connection_retries",
                ):
                    config[f"features.{feature}"] = False
                for key, value in config.items():
                    args.extend(["-c", f"{key}={json.dumps(value)}"])
                args.append("-")
                try:
                    process = subprocess.Popen(
                        args, cwd=directory, env=_child_environment(),
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, encoding="utf-8", start_new_session=os.name == "posix",
                    )
                except OSError:
                    raise LLMConfigurationError("Codex CLI could not be started.") from None
                try:
                    stdout, _ = process.communicate(
                        input="\n\n".join(human), timeout=self.settings.llm_timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    _stop_process(process)
                    raise RuntimeError("Codex CLI request timed out.") from None
                except BaseException:
                    _stop_process(process)
                    raise
                finally:
                    _capture_output(last_message, call)
                final_message = _read_events(stdout, call)
                if process.returncode != 0:
                    raise RuntimeError("Codex CLI exited unsuccessfully.")
                if not call["output_text"]:
                    raise RuntimeError("Codex CLI returned no final output.")
                if not isinstance(final_message, str) or final_message.strip() != call["output_text"]:
                    raise RuntimeError("Codex CLI final message did not match the saved output.")
                try:
                    content = json.loads(call["output_text"])
                    if not isinstance(content, dict):
                        raise ValueError
                except ValueError:
                    raise RuntimeError("Codex CLI final output must be exactly one JSON object.") from None
                call["status"] = "completed"
                return SimpleNamespace(content=call["output_text"])
        except (LLMConfigurationError, RuntimeError) as error:
            call["error"] = str(error)
            raise
        except Exception:
            call["error"] = "Codex CLI invocation failed."
            raise RuntimeError(call["error"]) from None
        finally:
            call["elapsed_seconds"] = round(time.perf_counter() - started, 3)
