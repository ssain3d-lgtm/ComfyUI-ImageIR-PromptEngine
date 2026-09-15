"""llama.cpp router-mode model discovery, loading, and local launch support.

This is deliberately separate from the legacy single-model launcher.  A router
server starts without one eager ``--model`` and exposes a catalogue through
``/v1/models``; selected models are then loaded/unloaded through the router API.
The resulting selection is converted back into the ordinary ``BackendConfig``
used everywhere else in ImageIR, so the analyzer and H3 authoring code do not
need to know that a router exists.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

from .base import BackendConfig, BackendError, Secret, mask, post_json
from .llama_cpp_launcher import FAILED, RUNNING_EXTERNAL, RUNNING_OURS, STOPPED, ServerStatus

CONNECTION_MODES = ("connect_existing", "launch_router")
MODEL_STATES = ("loaded", "unloaded", "loading", "sleeping", "failed", "unknown")


@dataclass(frozen=True)
class LlamaRouterConfig:
    """Connection, launch and per-model settings for a llama.cpp router."""

    connection_mode: str = "connect_existing"
    base_url: str = "http://127.0.0.1:8080"
    api_token: Secret = field(default_factory=Secret)

    llama_server_path: str = "llama-server"
    models_dir: str = ""
    host: str = "127.0.0.1"
    port: int = 8080
    models_max: int = 1
    models_autoload: bool = False
    router_extra_args: str = ""

    context_size: int = 32768
    gpu_layers: int = 999
    model_extra_args: str = "--jinja"

    max_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 0.9
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if self.connection_mode not in CONNECTION_MODES:
            raise BackendError(f"connection_mode must be one of {', '.join(CONNECTION_MODES)}")
        if not 1 <= int(self.port) <= 65535:
            raise BackendError("port must be between 1 and 65535")
        if int(self.models_max) < 0:
            raise BackendError("models_max must be 0 (unlimited) or greater")
        if int(self.context_size) < 512:
            raise BackendError("context_size must be at least 512")
        if int(self.gpu_layers) < 0:
            raise BackendError("gpu_layers cannot be negative")
        if int(self.max_tokens) < 1:
            raise BackendError("max_tokens must be at least 1")
        if not 0.0 <= float(self.temperature) <= 2.0:
            raise BackendError("temperature must be between 0 and 2")
        if not 0.0 <= float(self.top_p) <= 1.0:
            raise BackendError("top_p must be between 0 and 1")
        if float(self.timeout) <= 0:
            raise BackendError("timeout must be positive")

    @property
    def local_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def server_url(self) -> str:
        return self.local_url if self.connection_mode == "launch_router" else self.base_url.rstrip("/")

    def as_backend(self, model_id: str) -> BackendConfig:
        """Return the ordinary backend wire consumed by Analyzer/Intent nodes."""
        model_id = (model_id or "").strip()
        if not model_id:
            raise BackendError("select a llama.cpp model before creating the backend")
        return BackendConfig(
            provider="llama_cpp",
            server_mode="connect_existing",
            base_url=self.server_url,
            api_token=self.api_token,
            model_name=model_id,
            max_tokens=int(self.max_tokens),
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            timeout=float(self.timeout),
        )


@dataclass(frozen=True)
class LlamaModelInfo:
    id: str
    status: str = "unknown"
    vision: bool = False
    path: str = ""
    name: str = ""

    @property
    def loaded(self) -> bool:
        return self.status == "loaded"


def build_router_command(config: LlamaRouterConfig) -> list[str]:
    """Build current llama-server router arguments without eagerly loading a model."""
    command = [
        (config.llama_server_path or "llama-server").strip(),
        "--host", config.host,
        "--port", str(config.port),
        "--models-dir", config.models_dir,
        "--models-max", str(config.models_max),
        "--models-autoload" if config.models_autoload else "--no-models-autoload",
    ]
    if config.router_extra_args.strip():
        command.extend(shlex.split(config.router_extra_args.strip()))
    return command


def build_model_load_args(config: LlamaRouterConfig) -> list[str]:
    """Arguments applied to each selected model instance by POST /models/load."""
    args = [
        "--ctx-size", str(config.context_size),
        "--n-gpu-layers", str(config.gpu_layers),
    ]
    if config.model_extra_args.strip():
        args.extend(shlex.split(config.model_extra_args.strip()))
    return args


def _headers(config: LlamaRouterConfig) -> dict[str, str]:
    headers: dict[str, str] = {}
    if config.api_token:
        headers["Authorization"] = f"Bearer {config.api_token.reveal()}"
    return headers


def _get_json(url: str, headers: dict[str, str], timeout: float, *, secret: Secret) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured local/http(s) server
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:  # noqa: BLE001
            pass
        raise BackendError(mask(f"HTTP {exc.code} from {url}: {detail}", secret)) from None
    except urllib.error.URLError as exc:
        raise BackendError(mask(f"could not reach {url}: {exc.reason}", secret)) from None
    except TimeoutError:
        raise BackendError(f"timed out after {timeout:g}s waiting for {url}") from None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BackendError(mask(f"{url} did not return JSON ({exc}): {raw[:300]}", secret)) from None
    if not isinstance(value, dict):
        raise BackendError(f"{url} returned JSON that is not an object")
    return value


Getter = Callable[[str, dict[str, str], float], dict[str, Any]]
Sender = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


class LlamaCppModelManager:
    """Small synchronous client for llama.cpp's router model endpoints."""

    def __init__(self, config: LlamaRouterConfig, *, getter: Getter | None = None, sender: Sender | None = None):
        self.config = config
        self._get = getter or self._default_get
        self._send = sender or self._default_send

    @property
    def base_url(self) -> str:
        return self.config.server_url.rstrip("/")

    def _default_get(self, url: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
        return _get_json(url, headers, timeout, secret=self.config.api_token)

    def _default_send(self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
        return post_json(url, payload, headers, timeout, secrets=(self.config.api_token,))

    def list_models(self) -> tuple[LlamaModelInfo, ...]:
        response = self._get(f"{self.base_url}/v1/models", _headers(self.config), self.config.timeout)
        data = response.get("data")
        if not isinstance(data, list):
            raise BackendError("llama.cpp /v1/models response has no data array; router mode may be unavailable")
        models = []
        for entry in data:
            if not isinstance(entry, dict) or not str(entry.get("id") or "").strip():
                continue
            status_obj = entry.get("status") if isinstance(entry.get("status"), dict) else {}
            status = str(status_obj.get("value") or "unknown").lower()
            if status not in MODEL_STATES:
                status = "unknown"
            architecture = entry.get("architecture") if isinstance(entry.get("architecture"), dict) else {}
            modalities = architecture.get("input_modalities")
            modalities = modalities if isinstance(modalities, list) else []
            models.append(LlamaModelInfo(
                id=str(entry["id"]),
                name=str(entry.get("name") or entry["id"]),
                status=status,
                vision="image" in {str(value).lower() for value in modalities},
                path=str(entry.get("path") or ""),
            ))
        return tuple(models)

    def _find(self, model_id: str, models: tuple[LlamaModelInfo, ...] | None = None) -> LlamaModelInfo | None:
        for model in models if models is not None else self.list_models():
            if model.id == model_id:
                return model
        return None

    def load(self, model_id: str, *, wait_timeout: float = 180.0, poll_interval: float = 0.25) -> LlamaModelInfo:
        model_id = (model_id or "").strip()
        if not model_id:
            raise BackendError("select a model before loading")
        payload = {"model": model_id, "extra_args": build_model_load_args(self.config)}
        self._send(f"{self.base_url}/models/load", payload, _headers(self.config), self.config.timeout)
        return self._wait_for(model_id, {"loaded"}, wait_timeout, poll_interval)

    def unload(self, model_id: str, *, wait_timeout: float = 90.0, poll_interval: float = 0.25) -> LlamaModelInfo:
        model_id = (model_id or "").strip()
        if not model_id:
            raise BackendError("select a model before unloading")
        self._send(f"{self.base_url}/models/unload", {"model": model_id}, _headers(self.config), self.config.timeout)
        return self._wait_for(model_id, {"unloaded", "sleeping"}, wait_timeout, poll_interval, absent_ok=True)

    def _wait_for(self, model_id: str, wanted: set[str], wait_timeout: float, poll_interval: float,
                  *, absent_ok: bool = False) -> LlamaModelInfo:
        deadline = time.monotonic() + max(0.0, float(wait_timeout))
        last: LlamaModelInfo | None = None
        while True:
            models = self.list_models()
            last = self._find(model_id, models)
            if last is None and absent_ok:
                return LlamaModelInfo(model_id, status="unloaded")
            if last is not None and last.status in wanted:
                return last
            if last is not None and last.status == "failed":
                raise BackendError(f"llama.cpp reported model {model_id!r} failed to load")
            if time.monotonic() >= deadline:
                status = last.status if last else "missing"
                raise BackendError(f"timed out waiting for model {model_id!r}; last status: {status}")
            time.sleep(max(0.0, float(poll_interval)))

    def selected_config(self, model_id: str) -> BackendConfig:
        return self.config.as_backend(model_id)


@dataclass
class _OwnedRouter:
    process: Any
    command: list[str]

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


def _port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _spawn(command: list[str], env: dict[str, str]):
    return subprocess.Popen(  # noqa: S603 - argv list, never shell=True
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env=env,
    )


class LlamaRouterLauncher:
    """Own router-mode llama-server processes and never stop external ones."""

    def __init__(self, *, spawn=None, probe=None):
        self._spawn = spawn or _spawn
        self._probe = probe or _port_open
        self._owned: dict[tuple[str, int], _OwnedRouter] = {}
        self._lock = threading.RLock()

    def _validate(self, config: LlamaRouterConfig) -> str:
        executable = (config.llama_server_path or "").strip()
        if not executable:
            return "llama_server_path is empty"
        if os.sep in executable or (os.altsep and os.altsep in executable):
            if not os.path.isfile(executable):
                return f"llama-server not found at {executable}"
        elif shutil.which(executable) is None:
            return f"{executable!r} is not on PATH; give the full llama-server path"
        models_dir = (config.models_dir or "").strip()
        if not models_dir:
            return "models_dir is empty; select the directory llama.cpp router should scan"
        if not os.path.isdir(models_dir):
            return f"models directory not found at {models_dir}"
        return ""

    def status(self, config: LlamaRouterConfig) -> ServerStatus:
        key = (config.host, config.port)
        with self._lock:
            owned = self._owned.get(key)
            if owned and owned.alive:
                return ServerStatus(RUNNING_OURS, config.local_url, pid=getattr(owned.process, "pid", None),
                                    command=shlex.join(owned.command))
            if owned:
                self._owned.pop(key, None)
        if self._probe(config.host, config.port):
            return ServerStatus(RUNNING_EXTERNAL, config.local_url,
                                detail="router is already running outside this extension; it will not be stopped here")
        return ServerStatus(STOPPED, config.local_url)

    def start(self, config: LlamaRouterConfig, *, ready_timeout: float = 90.0) -> ServerStatus:
        key = (config.host, config.port)
        with self._lock:
            owned = self._owned.get(key)
            if owned and owned.alive:
                return ServerStatus(RUNNING_OURS, config.local_url, detail="already running",
                                    pid=getattr(owned.process, "pid", None), command=shlex.join(owned.command))
            if self._probe(config.host, config.port):
                return ServerStatus(RUNNING_EXTERNAL, config.local_url,
                                    detail="a server already answers on this port; using it without taking ownership")
            problem = self._validate(config)
            if problem:
                return ServerStatus(FAILED, config.local_url, detail=problem)
            command = build_router_command(config)
            env = dict(os.environ)
            if config.api_token:
                env["LLAMA_ARG_API_KEY"] = config.api_token.reveal()
            try:
                process = self._spawn(command, env)
            except (OSError, ValueError) as exc:
                return ServerStatus(FAILED, config.local_url,
                                    detail=mask(f"could not start llama-server router: {exc}", config.api_token),
                                    command=shlex.join(command))
            owned = _OwnedRouter(process, command)
            self._owned[key] = owned

        deadline = time.monotonic() + max(0.1, float(ready_timeout))
        while time.monotonic() < deadline:
            if not owned.alive:
                with self._lock:
                    self._owned.pop(key, None)
                return ServerStatus(FAILED, config.local_url,
                                    detail=f"llama-server router exited during startup (code {process.poll()})",
                                    command=shlex.join(command))
            if self._probe(config.host, config.port):
                return ServerStatus(RUNNING_OURS, config.local_url, detail="ready",
                                    pid=getattr(process, "pid", None), command=shlex.join(command))
            time.sleep(0.25)
        return ServerStatus(FAILED, config.local_url,
                            detail=f"router did not answer within {ready_timeout:g}s; it may still be starting",
                            pid=getattr(process, "pid", None), command=shlex.join(command))

    def stop(self, config: LlamaRouterConfig, *, timeout: float = 10.0) -> ServerStatus:
        key = (config.host, config.port)
        with self._lock:
            owned = self._owned.pop(key, None)
        if owned is None:
            if self._probe(config.host, config.port):
                return ServerStatus(RUNNING_EXTERNAL, config.local_url,
                                    detail="this router was not started here, so it was left alone")
            return ServerStatus(STOPPED, config.local_url, detail="nothing to stop")
        if owned.alive:
            owned.process.terminate()
            try:
                owned.process.wait(timeout=timeout)
            except Exception:  # noqa: BLE001
                owned.process.kill()
        return ServerStatus(STOPPED, config.local_url, detail="stopped", command=shlex.join(owned.command))


ROUTER_LAUNCHER = LlamaRouterLauncher()


__all__ = [
    "CONNECTION_MODES",
    "LlamaCppModelManager",
    "LlamaModelInfo",
    "LlamaRouterConfig",
    "LlamaRouterLauncher",
    "ROUTER_LAUNCHER",
    "build_model_load_args",
    "build_router_command",
]
