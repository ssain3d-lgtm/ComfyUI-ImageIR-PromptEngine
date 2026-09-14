"""Backends: one interface, three ways to reach a vision model."""

from .base import (
    PROVIDERS,
    SERVER_MODES,
    BackendConfig,
    BackendError,
    BackendResult,
    ImagePayload,
    Secret,
    VisionBackend,
    mask,
)
from .gemini import GeminiBackend
from .llama_cpp_launcher import LAUNCHER, LlamaServerLauncher, ServerStatus, build_command
from .openai_compatible import OpenAICompatibleBackend


def get_backend(config: BackendConfig, sender=None) -> VisionBackend:
    """The client for ``config``.

    llama_cpp and openai_compatible deliberately resolve to the same class. A
    llama-server speaks the OpenAI chat shape, so the only thing its provider
    setting changes is which URL to use — the one this extension launched, or
    the one the user pointed at. Giving it a class of its own would duplicate
    the client and let the copies drift.
    """
    if config.provider == "gemini":
        return GeminiBackend(config, sender)
    base_url = config.local_url if (config.provider == "llama_cpp" and config.server_mode == "launch_local") else config.base_url
    return OpenAICompatibleBackend(config, sender, base_url=base_url)


__all__ = [
    "LAUNCHER",
    "PROVIDERS",
    "SERVER_MODES",
    "BackendConfig",
    "BackendError",
    "BackendResult",
    "GeminiBackend",
    "ImagePayload",
    "LlamaServerLauncher",
    "OpenAICompatibleBackend",
    "Secret",
    "ServerStatus",
    "VisionBackend",
    "build_command",
    "get_backend",
    "mask",
]
