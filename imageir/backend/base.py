"""Backend-neutral plumbing: configuration, secrets, transport, and the interface.

Three rules shape this module.

*Secrets never travel with the data.* An API token reaches exactly one place —
the Authorization header of the request that needs it — and is wrapped in a type
that refuses to render itself anywhere else. Debug output, exception text and
backend diagnostics pass through ``mask``, because the usual way a key leaks is
not a print statement someone wrote on purpose; it is an HTTP 401 body echoed
into a traceback that ends up in a bug report.

*Payload construction is separate from transport.* Every backend builds a plain
dict and hands it to a sender. That split is what lets the tests assert the
exact JSON each provider sends — the part that is easy to get subtly wrong —
without a network, a server, or a mocked socket.

*api_token and max_tokens are different things.* One authenticates, one bounds
the length of the reply. They are named apart here and everywhere downstream,
because collapsing them is a mistake that produces a working-looking config
that either rejects every request or truncates every answer. ComfyUI itself
serializes direct string widgets: use the environment-variable-name option to
keep credentials out of saved workflows and PNG metadata.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from base64 import b64encode
from dataclasses import dataclass, field, replace
from typing import Any
from collections.abc import Callable

PROVIDERS = ("llama_cpp", "openai_compatible", "gemini")
SERVER_MODES = ("launch_local", "connect_existing")

MASK = "***"


class BackendError(RuntimeError):
    """A backend could not produce a result. The message is already masked."""


class Secret:
    """A string that will not render itself.

    Every accidental disclosure path — f-strings, ``repr`` in a traceback,
    ``json.dumps`` of a config, a logger formatting its argument — goes through
    ``__str__`` or ``__repr__``. Making both return a constant means a token can
    only escape through ``reveal()``, which is greppable and appears in exactly
    one place per backend.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str | None = "") -> None:
        self._value = (value or "").strip()

    def reveal(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __repr__(self) -> str:
        return f"Secret({MASK})" if self._value else "Secret(empty)"

    def __str__(self) -> str:
        return MASK if self._value else ""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and self._value == other._value

    def __hash__(self) -> int:
        return hash(("Secret", self._value))


def mask(text: str, *secrets: Secret | str) -> str:
    """Replace every secret occurrence in ``text``.

    Applied to anything that may be shown: debug panels, error messages, the
    excerpt of a malformed model reply. Longest first, so a token that contains
    a shorter one is not partially unmasked by the shorter replacement.
    """
    values = sorted(
        (s.reveal() if isinstance(s, Secret) else str(s or "") for s in secrets),
        key=len,
        reverse=True,
    )
    out = text or ""
    for value in values:
        if len(value) >= 4:  # below that, masking would eat ordinary words
            out = out.replace(value, MASK)
    return out


@dataclass(frozen=True)
class ImagePayload:
    """One encoded still, ready for whichever shape a provider wants."""

    data: bytes
    media_type: str = "image/png"

    @property
    def b64(self) -> str:
        return b64encode(self.data).decode("ascii")

    @property
    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.b64}"


@dataclass(frozen=True)
class BackendConfig:
    """Everything a backend needs, and nothing about what it is asked to do.

    ``api_token`` authenticates the request. ``max_tokens`` bounds the length of
    the generated reply. They are unrelated, and the names say so.
    """

    provider: str = "openai_compatible"
    server_mode: str = "connect_existing"

    # Endpoint and credentials
    base_url: str = "http://127.0.0.1:8080"
    api_token: Secret = field(default_factory=Secret)
    model_name: str = ""

    # Generation
    max_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.9
    timeout: float = 120.0

    # Local llama.cpp launch. Ignored entirely by the other providers and by
    # connect_existing, but kept on one config so a single wire carries a whole
    # backend choice through a ComfyUI graph.
    llama_server_path: str = "llama-server"
    gguf_model_path: str = ""
    mmproj_path: str = ""
    host: str = "127.0.0.1"
    port: int = 8080
    context_size: int = 8192
    gpu_layers: int = 0
    extra_args: str = ""

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise BackendError(f"provider must be one of {', '.join(PROVIDERS)}, got {self.provider!r}")
        if self.server_mode not in SERVER_MODES:
            raise BackendError(f"server_mode must be one of {', '.join(SERVER_MODES)}, got {self.server_mode!r}")
        if self.max_tokens < 1:
            raise BackendError("max_tokens must be at least 1 (it bounds the reply, not the credential)")
        if self.timeout <= 0:
            raise BackendError("timeout must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise BackendError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if not 0.0 <= self.top_p <= 1.0:
            raise BackendError(f"top_p must be between 0.0 and 1.0, got {self.top_p}")

    @property
    def local_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def with_token(self, token: str) -> BackendConfig:
        return replace(self, api_token=Secret(token))

    def redacted(self) -> dict[str, Any]:
        """A view safe to print, save in a workflow, or paste into an issue."""
        out: dict[str, Any] = {
            "provider": self.provider,
            "server_mode": self.server_mode,
            "base_url": self.base_url,
            "api_token": MASK if self.api_token else "(none)",
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
        }
        if self.provider == "llama_cpp":
            out.update(
                server_mode=self.server_mode,
                llama_server_path=self.llama_server_path,
                gguf_model_path=self.gguf_model_path,
                mmproj_path=self.mmproj_path,
                host=self.host,
                port=self.port,
                context_size=self.context_size,
                gpu_layers=self.gpu_layers,
                extra_args=self.extra_args,
            )
        return out

    def mask(self, text: str) -> str:
        return mask(text, self.api_token)


@dataclass(frozen=True)
class BackendResult:
    """What a backend returned, with the wire detail kept for debugging."""

    text: str
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    request_summary: str = ""  # already masked


Sender = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    *,
    secrets: tuple[Secret | str, ...] = (),
) -> dict[str, Any]:
    """POST JSON and decode JSON back, with every error path masked.

    A 401 body frequently quotes the credential it rejected, and an exception
    raised here lands in a ComfyUI console and then in a screenshot, so the
    masking is not decoration.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        request.add_header(key, value)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed http(s) scheme
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:  # noqa: BLE001 - a body we cannot read is not the interesting failure
            detail = ""
        raise BackendError(mask(f"HTTP {exc.code} from {url}: {detail}", *secrets)) from None
    except urllib.error.URLError as exc:
        raise BackendError(mask(f"could not reach {url}: {exc.reason}", *secrets)) from None
    except TimeoutError:
        raise BackendError(f"timed out after {timeout:g}s waiting for {url}") from None

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BackendError(mask(f"{url} did not return JSON ({exc}): {raw[:300]}", *secrets)) from None


class VisionBackend(ABC):
    """What the analyzer and the composers are allowed to ask of a provider.

    Two calls, because that is all this project needs: describe a picture, and
    continue some text. Anything provider-shaped — message roles, image
    encodings, endpoint paths — stays behind this line.
    """

    name = "backend"

    def __init__(self, config: BackendConfig, sender: Sender | None = None) -> None:
        self.config = config
        # Injected so tests drive the full request/response path without a
        # socket, and without patching module globals out from under themselves.
        self._send: Sender = sender or self._default_sender

    def _default_sender(self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
        return post_json(url, payload, headers, timeout, secrets=(self.config.api_token,))

    @abstractmethod
    def build_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image: ImagePayload | None = None,
    ) -> dict[str, Any]:
        """The exact JSON body this provider will be sent. Pure, so it is testable."""

    @abstractmethod
    def endpoint(self) -> str:
        """The full URL the payload goes to."""

    @abstractmethod
    def headers(self) -> dict[str, str]:
        """Request headers, including whatever carries the credential."""

    @abstractmethod
    def extract_text(self, response: dict[str, Any]) -> str:
        """Pull the generated text out of this provider's response shape."""

    def _call(self, *, system_prompt: str, user_prompt: str, image: ImagePayload | None) -> BackendResult:
        payload = self.build_payload(system_prompt=system_prompt, user_prompt=user_prompt, image=image)
        response = self._send(self.endpoint(), payload, self.headers(), self.config.timeout)
        text = self.extract_text(response)
        if not text.strip():
            raise BackendError(f"{self.name} returned an empty response")
        return BackendResult(
            text=text,
            model=str(response.get("model") or self.config.model_name),
            usage=dict(response.get("usage") or {}),
            request_summary=self.config.mask(summarize_payload(payload)),
        )

    def analyze_image(self, *, image: ImagePayload, system_prompt: str, user_prompt: str) -> BackendResult:
        return self._call(system_prompt=system_prompt, user_prompt=user_prompt, image=image)

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> BackendResult:
        return self._call(system_prompt=system_prompt, user_prompt=user_prompt, image=None)


def summarize_payload(payload: dict[str, Any]) -> str:
    """A debug view of a request with the image bytes elided.

    A base64 still is hundreds of kilobytes; printed in full it buries the two
    lines of the request anyone actually wants to read.
    """

    def shrink(value: Any) -> Any:
        if isinstance(value, str):
            if len(value) > 120:
                return f"<{len(value)} chars elided>"
            return value
        if isinstance(value, dict):
            return {k: shrink(v) for k, v in value.items()}
        if isinstance(value, list):
            return [shrink(v) for v in value]
        return value

    return json.dumps(shrink(payload), indent=2, ensure_ascii=False)
