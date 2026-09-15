"""ComfyUI surface for the llama.cpp router model manager.

The visible model dropdown/buttons are enhanced by ``web/llama_router_manager.js``.
The Python node remains fully usable without JavaScript: ``model_id`` is a plain
fallback field and Queue will verify/start/load the selected model before emitting
the ordinary IMAGEIR_BACKEND used by Analyzer and H3 authoring nodes.
"""

from __future__ import annotations

import json
import os
import re

from .imageir.backend import BackendError, Secret
from .imageir.backend.llama_cpp_models import (
    CONNECTION_MODES,
    LlamaCppModelManager,
    LlamaRouterConfig,
    ROUTER_LAUNCHER,
)

CATEGORY = "IMAGE_IR"


def _text(default: str, tooltip: str):
    return ("STRING", {"default": default, "multiline": False, "tooltip": tooltip})


def _token_from_env(name: str) -> Secret:
    name = (name or "").strip()
    if not name:
        return Secret()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("api_token_env must be an environment variable name")
    value = os.environ.get(name, "")
    if not value.strip():
        raise ValueError("api_token_env is missing or empty in the ComfyUI process environment")
    return Secret(value)


def build_router_config(
    connection_mode="connect_existing",
    base_url="http://127.0.0.1:8080",
    api_token_env="",
    llama_server_path="llama-server",
    models_dir="",
    host="127.0.0.1",
    port=8080,
    models_max=1,
    models_autoload=False,
    router_extra_args="",
    context_size=32768,
    gpu_layers=999,
    model_extra_args="--jinja",
    max_tokens=4096,
    temperature=0.2,
    top_p=0.9,
    timeout=120.0,
):
    try:
        return LlamaRouterConfig(
            connection_mode=connection_mode,
            base_url=(base_url or "").strip(),
            api_token=_token_from_env(api_token_env),
            llama_server_path=(llama_server_path or "llama-server").strip(),
            models_dir=(models_dir or "").strip(),
            host=(host or "127.0.0.1").strip(),
            port=int(port),
            models_max=int(models_max),
            models_autoload=bool(models_autoload),
            router_extra_args=(router_extra_args or "").strip(),
            context_size=int(context_size),
            gpu_layers=int(gpu_layers),
            model_extra_args=(model_extra_args or "").strip(),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            timeout=float(timeout),
        )
    except BackendError as exc:
        raise ValueError(f"llama.cpp model manager: {exc}") from None


class ImageIRLlamaCppModelManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection_mode": (list(CONNECTION_MODES), {
                    "default": "connect_existing",
                    "tooltip": "connect_existing uses an already-running llama.cpp server. launch_router starts llama-server with --models-dir.",
                }),
                "base_url": _text("http://127.0.0.1:8080", "Existing router URL. Ignored for launch_router, which uses host/port."),
                "model_id": _text("", "Selected model id. Connect / Refresh fills the model selector; this field is the saved fallback value."),
                "max_tokens": ("INT", {"default": 4096, "min": 64, "max": 131072, "step": 64,
                                      "tooltip": "Maximum reply tokens for Analyzer/Intent calls using the selected model."}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05,
                                           "tooltip": "Generation temperature for the selected backend."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Nucleus sampling cutoff for the selected backend."}),
                "timeout": ("FLOAT", {"default": 120.0, "min": 5.0, "max": 1800.0, "step": 5.0,
                                      "tooltip": "HTTP timeout and default model-load wait budget in seconds."}),
            },
            "optional": {
                "api_token_env": _text("", "Environment variable NAME containing a llama.cpp API key. The secret itself is never saved in the workflow."),
                "llama_server_path": _text("llama-server", "launch_router: llama-server executable path or executable name on PATH."),
                "models_dir": _text("", "launch_router: directory exposed to llama.cpp router with --models-dir."),
                "host": _text("127.0.0.1", "launch_router: bind address."),
                "port": ("INT", {"default": 8080, "min": 1, "max": 65535,
                                 "tooltip": "launch_router: bind port."}),
                "models_max": ("INT", {"default": 1, "min": 0, "max": 32,
                                       "tooltip": "Maximum simultaneously loaded router models. 0 means unlimited; 1 is recommended for one RTX GPU."}),
                "models_autoload": ("BOOLEAN", {"default": False,
                                                "tooltip": "Let llama.cpp automatically load requested models. Off keeps loading explicit and predictable."}),
                "router_extra_args": _text("", "Extra llama-server arguments applied when starting router mode."),
                "context_size": ("INT", {"default": 32768, "min": 512, "max": 1048576, "step": 512,
                                        "tooltip": "Per-model --ctx-size passed to /models/load."}),
                "gpu_layers": ("INT", {"default": 999, "min": 0, "max": 1024,
                                       "tooltip": "Per-model --n-gpu-layers passed to /models/load."}),
                "model_extra_args": _text("--jinja", "Extra per-model llama.cpp arguments passed through /models/load, e.g. --jinja."),
                "ensure_loaded": ("BOOLEAN", {"default": True,
                                              "tooltip": "On Queue, load the selected model if it is not already loaded."}),
            },
        }

    RETURN_TYPES = ("IMAGEIR_BACKEND", "STRING", "STRING")
    RETURN_NAMES = ("backend_config", "manager_status", "models_json")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Connect to llama.cpp router mode, select/load a model, and emit a normal ImageIR backend config."

    def run(
        self,
        connection_mode="connect_existing",
        base_url="http://127.0.0.1:8080",
        model_id="",
        max_tokens=4096,
        temperature=0.2,
        top_p=0.9,
        timeout=120.0,
        api_token_env="",
        llama_server_path="llama-server",
        models_dir="",
        host="127.0.0.1",
        port=8080,
        models_max=1,
        models_autoload=False,
        router_extra_args="",
        context_size=32768,
        gpu_layers=999,
        model_extra_args="--jinja",
        ensure_loaded=True,
    ):
        config = build_router_config(
            connection_mode, base_url, api_token_env, llama_server_path, models_dir,
            host, port, models_max, models_autoload, router_extra_args,
            context_size, gpu_layers, model_extra_args,
            max_tokens, temperature, top_p, timeout,
        )
        if config.connection_mode == "launch_router":
            status = ROUTER_LAUNCHER.start(config, ready_timeout=float(timeout))
            if not status.ok:
                raise ValueError(status.render())

        manager = LlamaCppModelManager(config)
        try:
            models = manager.list_models()
            selected = (model_id or "").strip()
            if not selected:
                loaded_vision = next((m.id for m in models if m.loaded and m.vision), "")
                loaded_any = next((m.id for m in models if m.loaded), "")
                selected = loaded_vision or loaded_any or (models[0].id if models else "")
            if not selected:
                raise BackendError("llama.cpp returned no models; check router configuration and models directory")
            info = next((m for m in models if m.id == selected), None)
            if info is None:
                raise BackendError(f"selected model {selected!r} is not in the router model list; press Connect / Refresh")
            if ensure_loaded and not info.loaded:
                info = manager.load(selected, wait_timeout=float(timeout))
                models = manager.list_models()
            backend = manager.selected_config(selected)
        except BackendError as exc:
            raise ValueError(config.api_token and str(exc).replace(config.api_token.reveal(), "***") or str(exc)) from None

        payload = [
            {"id": model.id, "status": model.status, "vision": model.vision, "path": model.path}
            for model in models
        ]
        summary = f"llama.cpp router: {config.server_url}\nselected: {selected}\nstatus: {info.status}\nvision: {'yes' if info.vision else 'no'}"
        return backend, summary, json.dumps(payload, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {"ImageIRLlamaCppModelManager": ImageIRLlamaCppModelManager}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageIRLlamaCppModelManager": "Image IR llama.cpp Model Manager"}
