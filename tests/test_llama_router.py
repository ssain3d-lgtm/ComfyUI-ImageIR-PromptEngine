import json
import unittest

from imageir.backend.base import Secret
from imageir.backend.llama_router import LlamaRouterClient, RouterError


class LlamaRouterClientTests(unittest.TestCase):
    def test_list_models_parses_status_and_vision_capability(self):
        calls = []

        def transport(method, url, payload, headers, timeout):
            calls.append((method, url, payload, headers, timeout))
            return {
                "data": [
                    {
                        "id": "gemma-vision",
                        "path": "D:/models/gemma.gguf",
                        "status": {"value": "loaded"},
                        "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
                    },
                    {
                        "id": "text-only",
                        "status": {"value": "unloaded"},
                        "architecture": {"input_modalities": ["text"]},
                    },
                ]
            }

        client = LlamaRouterClient("http://127.0.0.1:8080", api_token=Secret("secret-token"), transport=transport)
        models = client.list_models()

        self.assertEqual([m.id for m in models], ["gemma-vision", "text-only"])
        self.assertTrue(models[0].vision)
        self.assertFalse(models[1].vision)
        self.assertEqual(models[0].status, "loaded")
        self.assertEqual(calls[0][0], "GET")
        self.assertEqual(calls[0][1], "http://127.0.0.1:8080/models")
        self.assertEqual(calls[0][3]["Authorization"], "Bearer secret-token")

    def test_reload_uses_router_reload_query(self):
        seen = []

        def transport(method, url, payload, headers, timeout):
            seen.append(url)
            return {"data": []}

        LlamaRouterClient("http://localhost:8080/", transport=transport).list_models(reload=True)
        self.assertEqual(seen, ["http://localhost:8080/models?reload=1"])

    def test_load_and_unload_send_exact_model_payload(self):
        calls = []

        def transport(method, url, payload, headers, timeout):
            calls.append((method, url, payload))
            return {"success": True}

        client = LlamaRouterClient("http://localhost:8080", transport=transport)
        client.load_model("gemma-vision")
        client.unload_model("gemma-vision")

        self.assertEqual(calls, [
            ("POST", "http://localhost:8080/models/load", {"model": "gemma-vision"}),
            ("POST", "http://localhost:8080/models/unload", {"model": "gemma-vision"}),
        ])

    def test_empty_or_malformed_model_list_fails_closed(self):
        for response in ({}, {"data": "not-a-list"}, {"data": [{"status": {"value": "loaded"}}]}):
            with self.subTest(response=response):
                client = LlamaRouterClient("http://localhost:8080", transport=lambda *_, response=response: response)
                with self.assertRaises(RouterError):
                    client.list_models()

    def test_failed_load_reports_failure_without_leaking_token(self):
        client = LlamaRouterClient(
            "http://localhost:8080",
            api_token=Secret("very-secret-token"),
            transport=lambda *_: {"success": False, "error": "very-secret-token refused"},
        )
        with self.assertRaises(RouterError) as ctx:
            client.load_model("gemma")
        self.assertNotIn("very-secret-token", str(ctx.exception))
        self.assertIn("***", str(ctx.exception))

    def test_models_json_is_frontend_safe_and_stable(self):
        client = LlamaRouterClient(
            "http://localhost:8080",
            transport=lambda *_: {
                "data": [{
                    "id": "vision",
                    "path": "D:/models/vision.gguf",
                    "status": {"value": "loading"},
                    "architecture": {"input_modalities": ["text", "image"]},
                }]
            },
        )
        payload = json.loads(client.models_json())
        self.assertEqual(payload, [{
            "id": "vision",
            "status": "loading",
            "vision": True,
            "path": "D:/models/vision.gguf",
        }])


if __name__ == "__main__":
    unittest.main()
