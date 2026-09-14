"""One client for every server that speaks the OpenAI chat-completions shape.

llama-server, LM Studio, vLLM, Ollama's compatibility endpoint and a dozen
others all accept the same request. Writing a class per product would mean the
same bug fixed three times and a fourth copy quietly diverging, so there is one
client here and the difference between products is a URL.

Local launching is a separate concern and lives in llama_cpp_launcher: starting
a process and speaking HTTP to it are unrelated jobs, and LM Studio needs the
second without the first.
"""

from __future__ import annotations

from typing import Any

from .base import BackendConfig, BackendError, ImagePayload, VisionBackend


def chat_completions_url(base_url: str) -> str:
    """Join a base URL to the chat endpoint, whatever the user pasted.

    People paste the server root, the ``/v1`` root, or the full endpoint — all
    three are what their tool of choice showed them — and a naive join turns two
    of the three into a 404 that reads like the server is down.
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise BackendError("base_url is empty; point it at the server, e.g. http://127.0.0.1:8080")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


class OpenAICompatibleBackend(VisionBackend):
    """Chat completions, with an optional image in the user turn."""

    name = "openai-compatible"

    def __init__(self, config: BackendConfig, sender=None, *, base_url: str | None = None) -> None:
        super().__init__(config, sender)
        # A llama.cpp config in launch_local mode addresses the process this
        # extension started, which is host:port rather than whatever base_url
        # happens to hold.
        self._base_url = base_url or config.base_url

    def endpoint(self) -> str:
        return chat_completions_url(self._base_url)

    def headers(self) -> dict[str, str]:
        # The only place the credential is unwrapped. Local servers usually need
        # no key, and sending an empty bearer makes some of them reject the call.
        if self.config.api_token:
            return {"Authorization": f"Bearer {self.config.api_token.reveal()}"}
        return {}

    def build_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image: ImagePayload | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] | str
        if image is None:
            content = user_prompt
        else:
            # Text first: several local vision stacks attend to the instruction
            # better when it precedes the image, and none require the reverse.
            content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image.data_url}},
            ]

        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        return {
            "model": self.config.model_name,
            "messages": messages,
            # max_tokens, not max_completion_tokens: the local servers this
            # targets implement the older spelling, and it is the one they all
            # agree on. It bounds the reply; the credential is a header.
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": False,
        }

    def extract_text(self, response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise BackendError(self.config.mask(f"no choices in response: {str(response)[:300]}"))
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            # Some servers answer in the structured-content shape they accept.
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str):
            raise BackendError(self.config.mask(f"unexpected message content: {str(message)[:300]}"))
        finish = choices[0].get("finish_reason")
        if finish == "length" and not content.strip():
            raise BackendError(f"the reply hit max_tokens ({self.config.max_tokens}) before producing any text")
        return content
