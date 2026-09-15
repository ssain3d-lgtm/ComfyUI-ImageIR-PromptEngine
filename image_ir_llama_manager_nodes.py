"""ComfyUI convenience node for llama.cpp multi-model router mode."""

from __future__ import annotations

import json
import os
import re

from .imageir.backend import BackendConfig, BackendError, Secret
from .imageir.backend.llama_router import LlamaRouterClient, RouterError

CATEGORY = "IMAGE_IR"


def _resolve_token(api_token: str, api_token_env: str) -> Secret:
    env_name = (api_token_env or "").strip()
    direct = api_token or ""
    if env_name:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
            raise ValueError("api_token_env must be an environment variable name")
        if direct:
            raise ValueError("choose api_token_env or direct api_token, not both")
        direct = os.environ.get(env_name, "")
        if not direct.strip():
            raise ValueError("api_token_env is missing or empty in the ComfyUI process environment")
    return Secret(direct)


class ImageIRLlamaModelManager:
    """Select/load a llama.cpp router model and emit an ordinary backend config."""

    ROUTER_CLIENT_CLASS = LlamaRouterClient

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {
                    "default": "http://127.0.0.1:8080", "multiline": False,
                    "tooltip": "llama.cpp router server started with --models-dir. Connect/Refresh in the web UI queries this address.",
                }),
                "selected_model": ([""], {
                    "tooltip": "Model id returned by llama.cpp /models. The frontend refresh button fills this dropdown.",
                }),
                "action": (["status", "refresh", "load", "unload"], {
                    "default": "status",
                    "tooltip": "status lists models; refresh reloads the router model source; load/unload changes the selected model state.",
                }),
                "max_tokens": ("INT", {
                    "default": 4096, "min": 64, "max": 131072, "step": 64,
                    "tooltip": "Maximum response length for Analyzer/User Intent requests using the selected model.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Generation temperature for the backend config emitted by this manager.",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Nucleus sampling cutoff for the emitted backend config.",
                }),
                "timeout": ("FLOAT", {
                    "default": 120.0, "min": 5.0, "max": 1800.0, "step": 5.0,
                    "tooltip": "HTTP timeout used by router operations and generation requests.",
                }),
            },
            "optional": {
                "api_token": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Optional router/API bearer token. Direct values are saved in workflows; prefer api_token_env.",
                }),
                "api_token_env": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Environment variable NAME containing the token. The secret itself is never returned by this node.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGEIR_BACKEND", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("backend_config", "models_json", "status", "selected_model")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Connect to llama.cpp router mode, refresh/select/load/unload a model, then emit a reusable ImageIR backend config."

    def run(
        self,
        base_url="http://127.0.0.1:8080",
        selected_model="",
        action="status",
        max_tokens=4096,
        temperature=0.2,
        top_p=0.9,
        timeout=120.0,
        api_token="",
        api_token_env="",
    ):
        if action not in ("status", "refresh", "load", "unload"):
            raise ValueError("action must be status, refresh, load, or unload")
        token = _resolve_token(api_token, api_token_env)
        model_id = (selected_model or "").strip()
        url = (base_url or "").strip().rstrip("/")

        try:
            client = self.ROUTER_CLIENT_CLASS(url, api_token=token, timeout=float(timeout))
            if action == "load":
                if not model_id:
                    raise ValueError("select a model before loading it")
                client.load_model(model_id)
            elif action == "unload":
                if not model_id:
                    raise ValueError("select a model before unloading it")
                client.unload_model(model_id)
            models = client.list_models(reload=action == "refresh")
        except RouterError as exc:
            raise ValueError(f"llama.cpp model manager: {exc}") from None

        if not model_id and models:
            loaded = next((model.id for model in models if model.status == "loaded"), None)
            model_id = loaded or models[0].id

        try:
            config = BackendConfig(
                provider="llama_cpp",
                server_mode="connect_existing",
                base_url=url,
                api_token=token,
                model_name=model_id,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                timeout=float(timeout),
            )
        except BackendError as exc:
            raise ValueError(f"llama.cpp model manager backend config: {exc}") from None

        model_data = [model.to_dict() for model in models]
        models_json = json.dumps(model_data, ensure_ascii=False, indent=2)
        loaded_count = sum(model.status == "loaded" for model in models)
        vision_count = sum(model.vision for model in models)
        selected = next((model for model in models if model.id == model_id), None)
        selected_status = selected.status if selected else "not listed"
        selected_vision = "vision" if selected and selected.vision else "text/unknown"
        status = (
            f"llama.cpp router: {url}\n"
            f"models: {len(models)} ({loaded_count} loaded, {vision_count} vision-capable)\n"
            f"selected: {model_id or '(none)'} [{selected_status}; {selected_vision}]\n"
            f"action: {action}"
        )
        return config, models_json, status, model_id


NODE_CLASS_MAPPINGS = {"ImageIRLlamaModelManager": ImageIRLlamaModelManager}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageIRLlamaModelManager": "ImageIR Llama.cpp Model Manager"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
