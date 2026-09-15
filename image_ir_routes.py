"""Optional ComfyUI HTTP routes for the interactive llama.cpp model manager.

Imports of ComfyUI/aiohttp stay inside ``register_routes`` so the plugin remains
importable and testable on a bare Python interpreter.
"""

from __future__ import annotations

from .image_ir_model_manager_nodes import build_router_config
from .imageir.backend import BackendError
from .imageir.backend.llama_cpp_models import LlamaCppModelManager, ROUTER_LAUNCHER

_REGISTERED = False


def register_routes() -> bool:
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return False

    prompt_server = getattr(PromptServer, "instance", None)
    routes = getattr(prompt_server, "routes", None)
    if routes is None:
        return False

    @routes.post("/imageir/llama/models")
    async def llama_models(request):
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            action = str(payload.get("action") or "refresh")
            config = build_router_config(
                connection_mode=payload.get("connection_mode", "connect_existing"),
                base_url=payload.get("base_url", "http://127.0.0.1:8080"),
                api_token_env=payload.get("api_token_env", ""),
                llama_server_path=payload.get("llama_server_path", "llama-server"),
                models_dir=payload.get("models_dir", ""),
                host=payload.get("host", "127.0.0.1"),
                port=payload.get("port", 8080),
                models_max=payload.get("models_max", 1),
                models_autoload=payload.get("models_autoload", False),
                router_extra_args=payload.get("router_extra_args", ""),
                context_size=payload.get("context_size", 32768),
                gpu_layers=payload.get("gpu_layers", 999),
                model_extra_args=payload.get("model_extra_args", "--jinja"),
                max_tokens=payload.get("max_tokens", 4096),
                temperature=payload.get("temperature", 0.2),
                top_p=payload.get("top_p", 0.9),
                timeout=payload.get("timeout", 120.0),
            )

            if action == "stop":
                if config.connection_mode != "launch_router":
                    return web.json_response({"ok": False, "error": "connect_existing servers are never stopped by ImageIR"}, status=400)
                status = ROUTER_LAUNCHER.stop(config)
                return web.json_response({"ok": status.state != "failed", "status": status.render(), "models": []})

            if config.connection_mode == "launch_router":
                status = ROUTER_LAUNCHER.start(config, ready_timeout=float(config.timeout))
                if not status.ok:
                    return web.json_response({"ok": False, "error": status.render()}, status=500)
                status_text = status.render()
            else:
                status_text = f"connected target: {config.server_url}"

            manager = LlamaCppModelManager(config)
            model_id = str(payload.get("model_id") or "").strip()
            if action == "load":
                if not model_id:
                    raise BackendError("select a model before loading")
                manager.load(model_id, wait_timeout=float(config.timeout))
            elif action == "unload":
                if not model_id:
                    raise BackendError("select a model before unloading")
                manager.unload(model_id, wait_timeout=float(config.timeout))
            elif action not in ("connect", "refresh"):
                raise ValueError(f"unknown model-manager action: {action}")

            models = manager.list_models()
            return web.json_response({
                "ok": True,
                "status": status_text,
                "models": [
                    {"id": model.id, "name": model.name, "status": model.status,
                     "vision": model.vision, "path": model.path}
                    for model in models
                ],
            })
        except (BackendError, ValueError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001 - node endpoint must report, not crash the server
            return web.json_response({"ok": False, "error": f"llama.cpp model manager: {exc}"}, status=500)

    _REGISTERED = True
    return True


__all__ = ["register_routes"]
