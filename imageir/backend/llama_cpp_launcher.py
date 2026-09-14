"""Start, watch and stop a local llama-server, and nothing else.

This module knows about processes and ports. It does not know about ComfyUI,
about HTTP chat payloads, or about IMAGE_IR — talking to the server it starts is
the OpenAI-compatible client's job, and LM Studio needs that client without ever
touching this file.

The rules it exists to enforce are about restraint:

*Never start a second server on a port that already answers.* Two llama-servers
on one port is not a race the second one wins; it is a crash, or worse, a
silently unused process holding a model in VRAM.

*Never stop a process this extension did not start.* A port answering on 8080
is very often the user's own llama-server or LM Studio, running work unrelated
to ComfyUI. Ownership is tracked explicitly, and stop() on a port we merely
found is refused rather than best-guessed.

*Never let a bad path or a dead process take ComfyUI down with it.* Every
failure returns a status carrying the reason and the last few lines the server
printed, because "it did not start" without the log is a support thread.

CLI flags follow current llama-server conventions (--model, --mmproj, --host,
--port, --ctx-size, --n-gpu-layers), long forms only: the short spellings have
churned across releases and the long ones have not.
"""

from __future__ import annotations

import os
import shlex
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

from .base import BackendConfig, Secret, mask

# How many lines of server output to keep. Enough to hold a CUDA error or a
# "failed to load model", short enough to paste into a node's debug output.
LOG_LINES = 40

STOPPED = "stopped"
RUNNING_OURS = "running (launched here)"
RUNNING_EXTERNAL = "running (started outside ComfyUI)"
FAILED = "failed"


@dataclass
class ServerStatus:
    state: str
    url: str
    detail: str = ""
    pid: int | None = None
    log_tail: str = ""
    command: str = ""

    @property
    def ok(self) -> bool:
        return self.state in (RUNNING_OURS, RUNNING_EXTERNAL)

    def render(self) -> str:
        lines = [f"state: {self.state}", f"url:   {self.url}"]
        if self.pid is not None:
            lines.append(f"pid:   {self.pid}")
        if self.detail:
            lines.append(f"note:  {self.detail}")
        if self.command:
            lines.append(f"command: {self.command}")
        if self.log_tail:
            lines.append("")
            lines.append("last output")
            lines.append("-" * 11)
            lines.append(self.log_tail)
        return "\n".join(lines)


@dataclass
class _Owned:
    """A server this extension started, and the output it has produced."""

    process: Any
    command: list[str] = field(default_factory=list)
    log: deque = field(default_factory=lambda: deque(maxlen=LOG_LINES))

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


def build_command(config: BackendConfig) -> list[str]:
    """The argv for a llama-server matching ``config``.

    Pure, so the exact flags are a test rather than something discovered when a
    user's server refuses to start. The API token is deliberately absent: a
    credential passed as an argument is visible to every process on the machine
    through ``ps``, so it travels in the environment instead (see build_env).
    """
    executable = (config.llama_server_path or "llama-server").strip()
    command = [
        executable,
        "--model", config.gguf_model_path,
        "--host", config.host,
        "--port", str(config.port),
        "--ctx-size", str(config.context_size),
        "--n-gpu-layers", str(config.gpu_layers),
    ]
    if config.mmproj_path.strip():
        command += ["--mmproj", config.mmproj_path.strip()]
    if config.extra_args.strip():
        command += shlex.split(config.extra_args.strip())
    return command


def build_env(config: BackendConfig, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the child, carrying the credential out of sight.

    llama-server reads LLAMA_ARG_API_KEY for --api-key, which keeps the token
    off the command line and therefore out of ps, out of crash reports, and out
    of the status text this module returns.
    """
    env = dict(os.environ if base_env is None else base_env)
    if config.api_token:
        env["LLAMA_ARG_API_KEY"] = config.api_token.reveal()
    return env


def port_is_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _default_spawn(command: list[str], env: dict[str, str]):
    return subprocess.Popen(  # noqa: S603 - argv list, no shell
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )


class LlamaServerLauncher:
    """Owns the llama-server processes this extension started.

    One instance is shared for the whole plugin (see ``LAUNCHER``): ComfyUI
    rebuilds node objects freely, and ownership that lived on a node would be
    forgotten the moment the graph changed, leaving orphaned servers holding
    models in memory.
    """

    def __init__(
        self,
        *,
        spawn: Callable[[list[str], dict[str, str]], Any] | None = None,
        probe: Callable[[str, int], bool] | None = None,
    ) -> None:
        self._spawn = spawn or _default_spawn
        self._probe = probe or (lambda host, port: port_is_open(host, port))
        self._owned: dict[tuple[str, int], _Owned] = {}
        self._lock = threading.RLock()

    # -- inspection ---------------------------------------------------------

    def status(self, config: BackendConfig) -> ServerStatus:
        key = (config.host, config.port)
        url = config.local_url
        with self._lock:
            owned = self._owned.get(key)
            if owned is not None and owned.alive:
                return ServerStatus(
                    state=RUNNING_OURS,
                    url=url,
                    pid=getattr(owned.process, "pid", None),
                    log_tail=self._tail(owned, config.api_token),
                    command=shlex.join(owned.command),
                )
            if owned is not None:
                code = owned.process.poll()
                status = ServerStatus(
                    state=FAILED if code else STOPPED,
                    url=url,
                    detail=f"the server this extension started has exited (code {code})",
                    log_tail=self._tail(owned, config.api_token),
                    command=shlex.join(owned.command),
                )
                self._owned.pop(key, None)
                return status

        if self._probe(config.host, config.port):
            return ServerStatus(
                state=RUNNING_EXTERNAL,
                url=url,
                detail="something else is already listening here; this extension will use it but never stop it",
            )
        return ServerStatus(state=STOPPED, url=url)

    # -- lifecycle ----------------------------------------------------------

    def start(self, config: BackendConfig, *, ready_timeout: float = 90.0) -> ServerStatus:
        key = (config.host, config.port)
        with self._lock:
            owned = self._owned.get(key)
            if owned is not None and owned.alive:
                # Idempotent on purpose: a user pressing Queue twice should not
                # end up with two servers fighting over one port and one GPU.
                return ServerStatus(
                    state=RUNNING_OURS,
                    url=config.local_url,
                    detail="already running; not starting a second one",
                    pid=getattr(owned.process, "pid", None),
                    log_tail=self._tail(owned, config.api_token),
                    command=shlex.join(owned.command),
                )

            if self._probe(config.host, config.port):
                return ServerStatus(
                    state=RUNNING_EXTERNAL,
                    url=config.local_url,
                    detail=(
                        f"port {config.port} is already in use by a server this extension did not start. "
                        f"Use it as-is with server_mode=connect_existing, or choose another port."
                    ),
                )

            problem = self._check_paths(config)
            if problem:
                return ServerStatus(state=FAILED, url=config.local_url, detail=problem)

            command = build_command(config)
            try:
                process = self._spawn(command, build_env(config))
            except (OSError, ValueError) as exc:
                return ServerStatus(
                    state=FAILED,
                    url=config.local_url,
                    detail=mask(f"could not start llama-server: {exc}", config.api_token),
                    command=shlex.join(command),
                )

            owned = _Owned(process=process, command=command)
            self._owned[key] = owned
            self._drain(owned)

        return self._await_ready(config, owned, ready_timeout)

    def stop(self, config: BackendConfig, *, timeout: float = 10.0) -> ServerStatus:
        key = (config.host, config.port)
        with self._lock:
            owned = self._owned.pop(key, None)

        if owned is None:
            # The refusal that matters: a port we merely found answering is very
            # likely the user's own server doing unrelated work.
            if self._probe(config.host, config.port):
                return ServerStatus(
                    state=RUNNING_EXTERNAL,
                    url=config.local_url,
                    detail="this server was not started here, so it was left alone",
                )
            return ServerStatus(state=STOPPED, url=config.local_url, detail="nothing to stop")

        log_tail = self._tail(owned, config.api_token)
        command = shlex.join(owned.command)
        if not owned.alive:
            return ServerStatus(state=STOPPED, url=config.local_url, detail="already exited",
                                log_tail=log_tail, command=command)

        owned.process.terminate()
        try:
            owned.process.wait(timeout=timeout)
        except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired, or a stub that cannot wait
            owned.process.kill()
        return ServerStatus(state=STOPPED, url=config.local_url, detail="stopped",
                            log_tail=log_tail, command=command)

    def restart(self, config: BackendConfig, **kwargs) -> ServerStatus:
        stopped = self.stop(config)
        if stopped.state == RUNNING_EXTERNAL:
            return stopped  # refused to stop someone else's; starting would collide
        return self.start(config, **kwargs)

    def stop_all(self) -> None:
        """Release every process this extension owns, for shutdown hooks."""
        with self._lock:
            owned_items = list(self._owned.items())
            self._owned.clear()
        for _key, owned in owned_items:
            if owned.alive:
                try:
                    owned.process.terminate()
                except Exception:  # noqa: BLE001 - shutdown must not raise
                    pass

    def owns(self, host: str, port: int) -> bool:
        with self._lock:
            owned = self._owned.get((host, port))
            return owned is not None and owned.alive

    # -- internals ----------------------------------------------------------

    def _check_paths(self, config: BackendConfig) -> str:
        executable = (config.llama_server_path or "").strip()
        if not executable:
            return "llama_server_path is empty; give the path to the llama-server binary"
        if os.sep in executable or (os.altsep and os.altsep in executable):
            if not os.path.isfile(executable):
                return f"llama-server not found at {executable}"
        elif shutil.which(executable) is None:
            return f"{executable!r} is not on PATH; give the full path to the llama-server binary"

        model = (config.gguf_model_path or "").strip()
        if not model:
            return "gguf_model_path is empty; point it at the .gguf weights"
        if not os.path.isfile(model):
            return f"GGUF model not found at {model}"

        mmproj = (config.mmproj_path or "").strip()
        if mmproj and not os.path.isfile(mmproj):
            return f"mmproj file not found at {mmproj} (leave it empty for a text-only model)"
        return ""

    def _drain(self, owned: _Owned) -> None:
        """Keep the last few lines of output without ever blocking on the pipe."""
        stream = getattr(owned.process, "stdout", None)
        if stream is None:
            return

        def pump() -> None:
            try:
                for line in stream:
                    owned.log.append(line.rstrip("\n"))
            except Exception:  # noqa: BLE001 - the pipe closing is the normal end
                pass

        thread = threading.Thread(target=pump, name="llama-server-log", daemon=True)
        thread.start()

    def _tail(self, owned: _Owned, token: Secret) -> str:
        return mask("\n".join(owned.log), token)

    def _await_ready(self, config: BackendConfig, owned: _Owned, ready_timeout: float) -> ServerStatus:
        deadline = time.monotonic() + ready_timeout
        command = shlex.join(owned.command)
        while time.monotonic() < deadline:
            if not owned.alive:
                code = owned.process.poll()
                with self._lock:
                    self._owned.pop((config.host, config.port), None)
                return ServerStatus(
                    state=FAILED,
                    url=config.local_url,
                    detail=f"llama-server exited during startup (code {code})",
                    log_tail=self._tail(owned, config.api_token),
                    command=command,
                )
            if self._probe(config.host, config.port):
                return ServerStatus(
                    state=RUNNING_OURS,
                    url=config.local_url,
                    pid=getattr(owned.process, "pid", None),
                    detail="ready",
                    log_tail=self._tail(owned, config.api_token),
                    command=command,
                )
            time.sleep(0.25)

        return ServerStatus(
            state=FAILED,
            url=config.local_url,
            detail=f"llama-server did not answer within {ready_timeout:g}s (it may still be loading the model)",
            pid=getattr(owned.process, "pid", None),
            log_tail=self._tail(owned, config.api_token),
            command=command,
        )


# One owner for the whole plugin. Node classes are rebuilt by ComfyUI whenever
# the graph changes; ownership tracked on them would be lost and the processes
# orphaned.
LAUNCHER = LlamaServerLauncher()
