import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import harness


class FakeModel:
    def __init__(self, model_id="vision-a", status="unloaded", vision=True, path="D:/models/a.gguf"):
        self.id = model_id
        self.status = status
        self.vision = vision
        self.path = path

    def to_dict(self):
        return {"id": self.id, "status": self.status, "vision": self.vision, "path": self.path}


class FakeClient:
    calls = []

    def __init__(self, base_url, *, api_token=None, timeout=30.0):
        type(self).calls.append(("init", base_url, bool(api_token), timeout))

    def list_models(self, *, reload=False):
        type(self).calls.append(("list", reload))
        return (FakeModel(status="loaded"), FakeModel("text-b", "unloaded", False, "D:/models/b.gguf"))

    def load_model(self, model_id):
        type(self).calls.append(("load", model_id))

    def unload_model(self, model_id):
        type(self).calls.append(("unload", model_id))


class LlamaManagerWebTests(unittest.TestCase):
    def setUp(self):
        FakeClient.calls = []

    def test_web_directory_and_frontend_asset_are_shipped(self):
        package = harness.load_node_package()
        self.assertEqual(package.WEB_DIRECTORY, "./web")
        self.assertTrue((harness.ROOT / "web" / "llama_model_manager.js").is_file())

    def test_proxy_only_allows_loopback_router_urls(self):
        from image_ir_web import validate_local_router_url

        self.assertEqual(validate_local_router_url("http://127.0.0.1:8080/"), "http://127.0.0.1:8080")
        self.assertEqual(validate_local_router_url("http://localhost:8080"), "http://localhost:8080")
        self.assertEqual(validate_local_router_url("http://[::1]:8080"), "http://[::1]:8080")
        for url in ("http://10.0.0.5:8080", "https://example.com", "file:///tmp/model", "http://user:pass@localhost:8080"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_local_router_url(url)

    def test_proxy_refresh_and_model_actions_use_router_contract(self):
        from image_ir_web import perform_router_action

        response = perform_router_action(
            {"base_url": "http://localhost:8080", "action": "refresh", "selected_model": "vision-a"},
            client_class=FakeClient,
        )
        self.assertIn(("list", True), FakeClient.calls)
        self.assertEqual(response["selected_model"], "vision-a")
        self.assertEqual(response["models"][0]["vision"], True)

        FakeClient.calls = []
        perform_router_action(
            {"base_url": "http://localhost:8080", "action": "load", "selected_model": "vision-a"},
            client_class=FakeClient,
        )
        self.assertIn(("load", "vision-a"), FakeClient.calls)
        self.assertEqual(FakeClient.calls[-1], ("list", False))

    def test_proxy_resolves_env_token_without_returning_it(self):
        from image_ir_web import perform_router_action

        with patch.dict(os.environ, {"LLAMA_MANAGER_TOKEN": "not-for-the-browser"}, clear=False):
            result = perform_router_action(
                {
                    "base_url": "http://localhost:8080",
                    "action": "status",
                    "selected_model": "vision-a",
                    "api_token_env": "LLAMA_MANAGER_TOKEN",
                },
                client_class=FakeClient,
            )
        self.assertTrue(FakeClient.calls[0][2])
        self.assertNotIn("not-for-the-browser", json.dumps(result))

    def test_frontend_uses_proxy_and_exposes_connect_refresh_load_unload(self):
        source = (harness.ROOT / "web" / "llama_model_manager.js").read_text(encoding="utf-8")
        self.assertIn("/imageir/llama-router", source)
        for label in ("Connect / Refresh", "Load", "Unload"):
            with self.subTest(label=label):
                self.assertIn(label, source)
        self.assertIn("selected_model", source)
        self.assertNotIn("api_token\"", source)


if __name__ == "__main__":
    unittest.main()
