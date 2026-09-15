"""Small client for llama.cpp multi-model router endpoints.

The current llama-server router exposes GET /models plus POST /models/load and
/models/unload.  Keeping this client below the ComfyUI layer lets tests exercise
that contract without a browser, GPU, or running llama-server.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

from .base import Secret, mask


class RouterError(RuntimeError):
    """A llama.cpp router request or response was invalid."""


@dataclass(frozen=True)
class RouterModel:
    """The small subset of router model metadata the UI needs."""

    id: str
    status: str
    vision: bool
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "status": self.status, "vision": self.vision, "path": self.path}


Transport = Callable[[str, str, dict[str, Any] | None, dict[str, str], float], dict[str, Any]]


def _default_transport(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    headers: dict[str, str],
    timeout: float,
    *,
    token: Secret,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured local/http(s) endpoint
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - error body is optional
            detail = ""
        suffix = f": {detail}" if detail else ""
        raise RouterError(mask(f"llama.cpp router HTTP {exc.code}{suffix}", token)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RouterError(mask(f"could not reach llama.cpp router: {exc}", token)) from None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RouterError(mask(f"llama.cpp router returned non-JSON data: {exc}", token)) from None
    if not isinstance(data, dict):
        raise RouterError("llama.cpp router response must be a JSON object")
    return data


class LlamaRouterClient:
    """List, load and unload models from a llama.cpp router server."""

    def __init__(
        self,
        base_url: str,
        *,
        api_token: Secret | str | None = None,
        timeout: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        url = (base_url or "").strip().rstrip("/")
        if not (url.startswith("http://") or url.startswith("https://")):
            raise RouterError("base_url must start with http:// or https://")
        if timeout <= 0:
            raise RouterError("timeout must be positive")
        self.base_url = url
        self.api_token = api_token if isinstance(api_token, Secret) else Secret(api_token)
        self.timeout = float(timeout)
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token.reveal()}"
        return headers

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.base_url + path
        try:
            if self._transport is not None:
                data = self._transport(method, url, payload, self._headers(), self.timeout)
            else:
                data = _default_transport(
                    method,
                    url,
                    payload,
                    self._headers(),
                    self.timeout,
                    token=self.api_token,
                )
        except RouterError:
            raise
        except Exception as exc:  # noqa: BLE001 - injected transports may raise arbitrary network errors
            raise RouterError(mask(f"llama.cpp router request failed: {exc}", self.api_token)) from None
        if not isinstance(data, dict):
            raise RouterError("llama.cpp router response must be a JSON object")
        return data

    def list_models(self, *, reload: bool = False) -> tuple[RouterModel, ...]:
        data = self._request("GET", "/models?reload=1" if reload else "/models")
        raw_models = data.get("data")
        if not isinstance(raw_models, list):
            raise RouterError("llama.cpp router /models response is missing a data array")

        models = []
        for item in raw_models:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
                raise RouterError("llama.cpp router returned a model without a valid id")
            status_obj = item.get("status")
            status = status_obj.get("value", "unknown") if isinstance(status_obj, dict) else "unknown"
            if not isinstance(status, str):
                status = "unknown"
            architecture = item.get("architecture")
            modalities = architecture.get("input_modalities", []) if isinstance(architecture, dict) else []
            vision = isinstance(modalities, list) and "image" in modalities
            path = item.get("path", "")
            models.append(RouterModel(item["id"].strip(), status, vision, path if isinstance(path, str) else ""))
        return tuple(models)

    def _change_model_state(self, endpoint: str, model_id: str) -> None:
        model = (model_id or "").strip()
        if not model:
            raise RouterError("model id must not be empty")
        data = self._request("POST", endpoint, {"model": model})
        if data.get("success") is not True:
            detail = data.get("error") or data.get("message") or "request was not accepted"
            raise RouterError(mask(f"llama.cpp router model operation failed: {detail}", self.api_token))

    def load_model(self, model_id: str) -> None:
        self._change_model_state("/models/load", model_id)

    def unload_model(self, model_id: str) -> None:
        self._change_model_state("/models/unload", model_id)

    def models_json(self, *, reload: bool = False) -> str:
        return json.dumps([model.to_dict() for model in self.list_models(reload=reload)], ensure_ascii=False)
