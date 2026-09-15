"""llama.cpp router-mode model discovery, loading, and ComfyUI manager UX."""

from __future__ import annotations

import importlib
import unittest

import harness

from imageir.backend import BackendConfig, Secret
from imageir.backend.llama_cpp_launcher import build_command

ROOT = harness.ROOT
TOKEN = "sk-test-fake"


def router_config(**kwargs):
    fields = BackendConfig.__dataclass_fields__
    required = {"llama_router_mode", "models_dir", "models_max", "models_autoload"}
    missing = required - set(fields)
    if missing:
        raise AssertionError(f"router config fields not implemented: {sorted(missing)}")
    base = dict(
        provider="llama_cpp",
        server_mode="launch_local",
        llama_server_path="llama-server",
        host="127.0.0.1",
        port=8080,
        model_name="",
        context_size=32768,
        gpu_layers=999,
        extra_args="--jinja",
        llama_router_mode=True,
        models_dir="/models",
        models_max=1,
        models_autoload=False,
    )
    base.update(kwargs)
    return BackendConfig(**base)


def manager_module():
    try:
        return importlib.import_module("imageir.backend.llama_cpp_models")
    except ImportError as exc:
        raise AssertionError("imageir.backend.llama_cpp_models is not implemented") from exc


class RouterLaunchTests(unittest.TestCase):
    def test_router_config_fields_exist(self):
        fields = BackendConfig.__dataclass_fields__
        for name in ("llama_router_mode", "models_dir", "models_max", "models_autoload"):
            with self.subTest(name=name):
                self.assertIn(name, fields)

    def test_router_launch_uses_models_dir_without_eager_model(self):
        command = build_command(router_config())
        self.assertIn("--models-dir", command)
        self.assertEqual(command[command.index("--models-dir") + 1], "/models")
        self.assertIn("--models-max", command)
        self.assertEqual(command[command.index("--models-max") + 1], "1")
        self.assertIn("--no-models-autoload", command)
        self.assertNotIn("--model", command)
        self.assertNotIn("--mmproj", command)
        self.assertNotIn("--ctx-size", command)
        self.assertNotIn("--n-gpu-layers", command)

    def test_single_model_launch_remains_unchanged(self):
        cfg = BackendConfig(
            provider="llama_cpp", server_mode="launch_local", gguf_model_path="/models/a.gguf",
            mmproj_path="/models/mmproj.gguf", context_size=16384, gpu_layers=99,
        )
        command = build_command(cfg)
        self.assertIn("--model", command)
        self.assertIn("--mmproj", command)
        self.assertIn("--ctx-size", command)
        self.assertIn("--n-gpu-layers", command)


class RouterModelClientTests(unittest.TestCase):
    def make_manager(self, responses=None):
        mod = manager_module()
        calls = []
        responses = list(responses or [])

        def getter(url, headers, timeout):
            calls.append(("GET", url, None, headers, timeout))
            if responses:
                return responses.pop(0)
            return {"object": "list", "data": []}

        def sender(url, payload, headers, timeout):
            calls.append(("POST", url, payload, headers, timeout))
            if responses:
                return responses.pop(0)
            return {"success": True}

        cfg = router_config(server_mode="connect_existing", base_url="http://127.0.0.1:8080",
                            api_token=Secret(TOKEN))
        return mod.LlamaCppModelManager(cfg, getter=getter, sender=sender), calls

    def test_refresh_reads_v1_models_and_preserves_status_and_vision(self):
        manager, calls = self.make_manager([{
            "object": "list",
            "data": [
                {"id": "gemma-vl", "object": "model", "owned_by": "llamacpp", "created": 0,
                 "in_cache": True, "path": "/models/gemma-vl", "status": {"value": "loaded"},
                 "architecture": {"input_modalities": ["text", "image"]}},
                {"id": "qwen-text", "object": "model", "owned_by": "llamacpp", "created": 0,
                 "in_cache": True, "path": "/models/qwen-text", "status": {"value": "unloaded"},
                 "architecture": {"input_modalities": ["text"]}},
            ],
        }])
        models = manager.list_models()
        self.assertEqual([m.id for m in models], ["gemma-vl", "qwen-text"])
        self.assertTrue(models[0].vision)
        self.assertFalse(models[1].vision)
        self.assertEqual(models[0].status, "loaded")
        self.assertTrue(calls[0][1].endswith("/v1/models"))
        self.assertEqual(calls[0][3]["Authorization"], f"Bearer {TOKEN}")

    def test_load_posts_router_payload_and_waits_until_loaded(self):
        manager, calls = self.make_manager([
            {"success": True},
            {"object": "list", "data": [
                {"id": "gemma-vl", "object": "model", "owned_by": "llamacpp", "created": 0,
                 "in_cache": True, "path": "/models/gemma-vl", "status": {"value": "loaded"},
                 "architecture": {"input_modalities": ["text", "image"]}},
            ]},
        ])
        model = manager.load("gemma-vl", wait_timeout=1.0, poll_interval=0.0)
        self.assertEqual(model.status, "loaded")
        method, url, payload, _headers, _timeout = calls[0]
        self.assertEqual(method, "POST")
        self.assertTrue(url.endswith("/models/load"))
        self.assertEqual(payload["model"], "gemma-vl")
        self.assertIn("--ctx-size", payload["extra_args"])
        self.assertIn("32768", payload["extra_args"])
        self.assertIn("--n-gpu-layers", payload["extra_args"])
        self.assertIn("999", payload["extra_args"])
        self.assertIn("--jinja", payload["extra_args"])

    def test_unload_posts_router_payload_and_accepts_unloaded_state(self):
        manager, calls = self.make_manager([
            {"success": True},
            {"object": "list", "data": [
                {"id": "gemma-vl", "object": "model", "owned_by": "llamacpp", "created": 0,
                 "in_cache": True, "path": "/models/gemma-vl", "status": {"value": "unloaded"}},
            ]},
        ])
        model = manager.unload("gemma-vl", wait_timeout=1.0, poll_interval=0.0)
        self.assertEqual(model.status, "unloaded")
        self.assertTrue(calls[0][1].endswith("/models/unload"))
        self.assertEqual(calls[0][2], {"model": "gemma-vl"})

    def test_selected_backend_config_uses_the_router_model_id(self):
        manager, _calls = self.make_manager()
        selected = manager.selected_config("gemma-vl")
        self.assertEqual(selected.model_name, "gemma-vl")
        self.assertEqual(selected.provider, "llama_cpp")


class ComfyManagerSurfaceTests(unittest.TestCase):
    def test_manager_node_and_web_extension_ship(self):
        package = harness.load_node_package()
        self.assertIn("ImageIRLlamaCppModelManager", package.NODE_CLASS_MAPPINGS)
        self.assertEqual(getattr(package, "WEB_DIRECTORY", None), "./web")
        js = ROOT / "web" / "llama_router_manager.js"
        self.assertTrue(js.is_file())
        text = harness.read_text(js)
        self.assertIn("app.registerExtension", text)
        self.assertIn("/imageir/llama/models", text)
        self.assertIn("Connect / Refresh", text)
        self.assertIn("Load", text)
        self.assertIn("Unload", text)

    def test_manager_node_exposes_router_connection_and_selected_model(self):
        package = harness.load_node_package()
        if "ImageIRLlamaCppModelManager" not in package.NODE_CLASS_MAPPINGS:
            self.fail("ImageIRLlamaCppModelManager is not implemented")
        cls = package.NODE_CLASS_MAPPINGS["ImageIRLlamaCppModelManager"]
        spec = cls.INPUT_TYPES()
        widgets = {**spec.get("required", {}), **spec.get("optional", {})}
        for name in ("connection_mode", "base_url", "models_dir", "model_id", "api_token_env"):
            with self.subTest(name=name):
                self.assertIn(name, widgets)
        self.assertIn("IMAGEIR_BACKEND", cls.RETURN_TYPES)


if __name__ == "__main__":
    unittest.main()
