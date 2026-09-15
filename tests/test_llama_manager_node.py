import json
import os
import unittest
from unittest.mock import patch

import harness

nodes = harness.load_node_package()
NODES = nodes.NODE_CLASS_MAPPINGS


class FakeModel:
    def __init__(self, model_id, status="unloaded", vision=True, path=""):
        self.id = model_id
        self.status = status
        self.vision = vision
        self.path = path

    def to_dict(self):
        return {"id": self.id, "status": self.status, "vision": self.vision, "path": self.path}


class FakeRouterClient:
    calls = []
    models = (FakeModel("vision-a", "loaded", True, "D:/models/a.gguf"),)

    def __init__(self, base_url, *, api_token=None, timeout=30.0):
        self.base_url = base_url
        self.api_token = api_token
        self.timeout = timeout
        type(self).calls.append(("init", base_url, bool(api_token), timeout))

    def list_models(self, *, reload=False):
        type(self).calls.append(("list", reload))
        return type(self).models

    def load_model(self, model_id):
        type(self).calls.append(("load", model_id))

    def unload_model(self, model_id):
        type(self).calls.append(("unload", model_id))


class LlamaModelManagerNodeTests(unittest.TestCase):
    def setUp(self):
        FakeRouterClient.calls = []
        self.cls = NODES["ImageIRLlamaModelManager"]
        self.node = self.cls()
        self.old_client = self.cls.ROUTER_CLIENT_CLASS
        self.cls.ROUTER_CLIENT_CLASS = FakeRouterClient

    def tearDown(self):
        self.cls.ROUTER_CLIENT_CLASS = self.old_client

    def test_mapping_and_display_name_exist(self):
        self.assertEqual(nodes.NODE_DISPLAY_NAME_MAPPINGS["ImageIRLlamaModelManager"], "ImageIR Llama.cpp Model Manager")

    def test_status_lists_models_and_emits_existing_backend_config(self):
        config, models_json, status, selected = self.node.run(
            base_url="http://127.0.0.1:8080",
            selected_model="vision-a",
            action="status",
            max_tokens=4096,
            temperature=0.15,
            top_p=0.9,
            timeout=60.0,
        )

        self.assertEqual(config.provider, "llama_cpp")
        self.assertEqual(config.server_mode, "connect_existing")
        self.assertEqual(config.base_url, "http://127.0.0.1:8080")
        self.assertEqual(config.model_name, "vision-a")
        self.assertEqual(config.max_tokens, 4096)
        self.assertEqual(selected, "vision-a")
        self.assertEqual(json.loads(models_json)[0]["vision"], True)
        self.assertIn("loaded", status)
        self.assertEqual(FakeRouterClient.calls[-1], ("list", False))

    def test_refresh_requests_router_reload(self):
        self.node.run("http://127.0.0.1:8080", "vision-a", "refresh")
        self.assertIn(("list", True), FakeRouterClient.calls)

    def test_load_and_unload_operate_then_refresh(self):
        self.node.run("http://127.0.0.1:8080", "vision-a", "load")
        self.assertIn(("load", "vision-a"), FakeRouterClient.calls)
        self.assertEqual(FakeRouterClient.calls[-1], ("list", False))

        FakeRouterClient.calls = []
        self.node.run("http://127.0.0.1:8080", "vision-a", "unload")
        self.assertIn(("unload", "vision-a"), FakeRouterClient.calls)
        self.assertEqual(FakeRouterClient.calls[-1], ("list", False))

    def test_load_requires_a_selected_model(self):
        with self.assertRaises(ValueError):
            self.node.run("http://127.0.0.1:8080", "", "load")

    def test_env_token_is_resolved_but_never_rendered(self):
        with patch.dict(os.environ, {"LLAMA_MANAGER_TOKEN": "super-secret-value"}, clear=False):
            config, _models, status, _selected = self.node.run(
                "http://127.0.0.1:8080", "vision-a", "status", api_token_env="LLAMA_MANAGER_TOKEN"
            )
        self.assertTrue(config.api_token)
        self.assertNotIn("super-secret-value", status)
        self.assertEqual(FakeRouterClient.calls[0][2], True)

    def test_direct_and_env_tokens_are_mutually_exclusive(self):
        with patch.dict(os.environ, {"LLAMA_MANAGER_TOKEN": "secret"}, clear=False):
            with self.assertRaises(ValueError):
                self.node.run(
                    "http://127.0.0.1:8080", "vision-a", "status",
                    api_token="direct", api_token_env="LLAMA_MANAGER_TOKEN",
                )


if __name__ == "__main__":
    unittest.main()
