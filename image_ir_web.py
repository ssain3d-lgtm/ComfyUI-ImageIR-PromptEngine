"""Optional ComfyUI web proxy for the llama.cpp router manager.

The pure helpers stay importable without ComfyUI/aiohttp so packaging tests and
the engine remain dependency-light. Route registration happens only when the
ComfyUI server modules are present.
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlsplit, urlunsplit

from .imageir.backend import Secret
from .imageir.backend.llama_router import LlamaRouterClient, RouterError

_REGISTERED = False


def validate_local_router_url(value: str) -> str:
    """Accept only loopback HTTP router URLs; this proxy must not become SSRF."""
    raw = (value or "").strip()
    parsed = urlsplit(raw)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("router URL must be a local http://localhost/127.0.0.1/::1 address")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("router URL must not contain credentials, query, or fragment")
    if parsed.path not in ("", "/"):
        raise ValueError("router URL must be the server root")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("router URL has an invalid port") from exc
    if port is None:
        port = 80
    if not 1 <= port <= 65535:
        raise ValueError("router port is out of range")
    host = f"[{parsed.hostname}]" if parsed.hostname == "::1" else parsed.hostname
    netloc = f"{host}:{port}" if port != 80 else host
    return urlunsplit(("http", netloc, "", "", ""))


def _env_token(name: str) -> Secret:
    env_name = (name or "").strip()
    if not env_name:
        return Secret("")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
        raise ValueError("api_token_env must be an environment variable name")
    token = os.environ.get(env_name, "")
    if not token.strip():
        raise ValueError("api_token_env is missing or empty in the ComfyUI process environment")
    return Secret(token)


def perform_router_action(payload, *, client_class=LlamaRouterClient):
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    action = payload.get("action", "status")
    if action not in ("status", "refresh", "load", "unload"):
        raise ValueError("unknown router action")
    base_url = validate_local_router_url(payload.get("base_url", "http://127.0.0.1:8080"))
    selected = str(payload.get("selected_model", "") or "").strip()
    token = _env_token(str(payload.get("api_token_env", "") or ""))
    timeout = float(payload.get("timeout", 30.0))
    client = client_class(base_url, api_token=token, timeout=timeout)

    if action in ("load", "unload") and not selected:
        raise ValueError("select a model first")
    if action == "load":
        client.load_model(selected)
    elif action == "unload":
        client.unload_model(selected)

    models = client.list_models(reload=action == "refresh")
    if not selected and models:
        selected = next((m.id for m in models if m.status == "loaded"), models[0].id)
    return {
        "models": [model.to_dict() for model in models],
        "selected_model": selected,
        "message": f"Connected to {base_url}; {len(models)} model(s) found.",
    }


def register_routes() -> bool:
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        from aiohttp import web
        from server import PromptServer
    except ModuleNotFoundError:
        return False

    routes = getattr(getattr(PromptServer, "instance", None), "routes", None)
    if routes is None:
        return False

    @routes.post("/imageir/llama-router")
    async def llama_router_proxy(request):
        try:
            payload = await request.json()
            result = perform_router_action(payload)
            return web.json_response(result)
        except (ValueError, RouterError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception:
            return web.json_response({"error": "llama.cpp router request failed"}, status=500)

    _REGISTERED = True
    return True


__all__ = ["perform_router_action", "register_routes", "validate_local_router_url"]
