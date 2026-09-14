"""Gemini through its REST API.

Deliberately no SDK. The alternative — ``google-generativeai`` guarded by a
try/except at import time — buys nothing here: the whole surface this project
uses is one POST, and a missing package that turns a provider into a runtime
surprise is worse than a dependency that never existed. This way the plugin
imports and the Gemini option appears whatever is installed, requirements.txt
stays empty, and there is no optional-dependency failure mode to handle.

The model is a configuration field, not a constant. Gemini model names turn over
on a schedule this repository does not control, and a hard-coded one is a
guaranteed future bug report.
"""

from __future__ import annotations

from typing import Any

from .base import BackendConfig, BackendError, ImagePayload, VisionBackend

DEFAULT_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


class GeminiBackend(VisionBackend):
    """generateContent, with the still sent as inline_data."""

    name = "gemini"

    def __init__(self, config: BackendConfig, sender=None, *, api_root: str | None = None) -> None:
        super().__init__(config, sender)
        # base_url is reused as an override so a proxy or a pinned API version
        # is reachable without a second field that means almost the same thing.
        override = (config.base_url or "").strip().rstrip("/")
        self._api_root = api_root or (override if override.startswith("https://") else DEFAULT_API_ROOT)

    def endpoint(self) -> str:
        model = (self.config.model_name or "").strip()
        if not model:
            raise BackendError("model_name is empty; set a Gemini model, e.g. gemini-2.5-flash")
        return f"{self._api_root}/models/{model}:generateContent"

    def headers(self) -> dict[str, str]:
        if not self.config.api_token:
            raise BackendError("Gemini needs an api_token (the API key); max_tokens is a different setting")
        # The header form, not ?key= — a URL carrying a credential ends up in
        # proxy logs, error messages and shell history.
        return {"x-goog-api-key": self.config.api_token.reveal()}

    def build_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image: ImagePayload | None = None,
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = [{"text": user_prompt}]
        if image is not None:
            parts.append({"inline_data": {"mime_type": image.media_type, "data": image.b64}})

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "maxOutputTokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "topP": self.config.top_p,
            },
        }
        if system_prompt.strip():
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    def extract_text(self, response: dict[str, Any]) -> str:
        # A refusal arrives as a 200 with no candidates and a reason buried in
        # promptFeedback; reported plainly it saves a long debugging detour.
        feedback = response.get("promptFeedback") or {}
        blocked = feedback.get("blockReason")
        candidates = response.get("candidates")
        if not candidates:
            if blocked:
                raise BackendError(f"Gemini declined the request (blockReason: {blocked})")
            raise BackendError(self.config.mask(f"no candidates in response: {str(response)[:300]}"))

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not text.strip():
            reason = candidate.get("finishReason")
            if reason == "MAX_TOKENS":
                raise BackendError(
                    f"the reply hit max_tokens ({self.config.max_tokens}) before producing any text"
                )
            raise BackendError(f"Gemini returned no text (finishReason: {reason})")
        return text
