"""Backend configuration, secret handling, and the exact JSON each provider sends.

Payloads are asserted rather than mocked-and-forgotten: the request body is the
part that silently breaks when a provider shifts, and it is cheap to pin.
"""

import json
import unittest

import harness  # noqa: F401

from imageir.backend import BackendConfig, BackendError, Secret, get_backend, mask
from imageir.backend.base import ImagePayload, summarize_payload
from imageir.backend.gemini import GeminiBackend
from imageir.backend.openai_compatible import OpenAICompatibleBackend, chat_completions_url

# Deliberately too short to look like a real credential; see test_packaging.
TOKEN = "sk-test-fake"
IMAGE = ImagePayload(b"\x89PNG\r\n\x1a\nfake-bytes", "image/png")


def collect(backend):
    """Run a call and hand back the request it would have made."""
    captured = {}

    def sender(url, payload, headers, timeout):
        captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
        return backend_reply(backend)

    backend._send = sender
    return captured


def backend_reply(backend):
    if isinstance(backend, GeminiBackend):
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
    return {"choices": [{"message": {"content": "ok"}}]}


class SecretTests(unittest.TestCase):
    def test_a_secret_refuses_to_render_itself(self):
        secret = Secret(TOKEN)
        self.assertNotIn(TOKEN, repr(secret))
        self.assertNotIn(TOKEN, str(secret))
        self.assertNotIn(TOKEN, f"{secret}")
        self.assertNotIn(TOKEN, "{}".format(secret))  # noqa: UP032 - the point is the format path

    def test_reveal_is_the_only_way_out(self):
        self.assertEqual(Secret(TOKEN).reveal(), TOKEN)

    def test_an_empty_secret_is_falsey_and_renders_empty(self):
        self.assertFalse(Secret(""))
        self.assertEqual(str(Secret("")), "")

    def test_mask_replaces_every_occurrence(self):
        text = f"Bearer {TOKEN} rejected; retry with {TOKEN}"
        self.assertNotIn(TOKEN, mask(text, Secret(TOKEN)))

    def test_mask_leaves_short_strings_alone(self):
        # Masking a two-character "secret" would redact ordinary prose.
        self.assertEqual(mask("the cat sat", Secret("at")), "the cat sat")

    def test_a_config_never_prints_its_token(self):
        config = BackendConfig(api_token=Secret(TOKEN))
        rendered = json.dumps(config.redacted())
        self.assertNotIn(TOKEN, rendered)
        self.assertIn("***", rendered)

    def test_a_config_with_no_token_says_so(self):
        self.assertEqual(BackendConfig().redacted()["api_token"], "(none)")


class ConfigTests(unittest.TestCase):
    def test_api_token_and_max_tokens_are_separate_fields(self):
        config = BackendConfig(api_token=Secret(TOKEN), max_tokens=4096)
        self.assertEqual(config.api_token.reveal(), TOKEN)
        self.assertEqual(config.max_tokens, 4096)

    def test_an_unknown_provider_is_rejected(self):
        with self.assertRaises(BackendError):
            BackendConfig(provider="anthropic_ish")

    def test_an_unknown_server_mode_is_rejected(self):
        with self.assertRaises(BackendError):
            BackendConfig(server_mode="maybe")

    def test_generation_settings_are_range_checked(self):
        for kwargs in ({"max_tokens": 0}, {"temperature": 5.0}, {"top_p": 1.5}, {"timeout": 0}):
            with self.subTest(**kwargs), self.assertRaises(BackendError):
                BackendConfig(**kwargs)

    def test_local_url_is_built_from_host_and_port(self):
        self.assertEqual(BackendConfig(host="0.0.0.0", port=9001).local_url, "http://0.0.0.0:9001")

    def test_llama_fields_are_only_reported_for_llama(self):
        self.assertNotIn("gguf_model_path", BackendConfig(provider="gemini").redacted())
        self.assertIn("gguf_model_path", BackendConfig(provider="llama_cpp").redacted())


class EndpointTests(unittest.TestCase):
    def test_a_server_root_gains_the_full_path(self):
        self.assertEqual(chat_completions_url("http://127.0.0.1:8080"), "http://127.0.0.1:8080/v1/chat/completions")

    def test_a_v1_root_is_not_doubled(self):
        self.assertEqual(chat_completions_url("http://127.0.0.1:1234/v1"), "http://127.0.0.1:1234/v1/chat/completions")

    def test_a_full_endpoint_is_left_alone(self):
        url = "http://x/v1/chat/completions"
        self.assertEqual(chat_completions_url(url), url)

    def test_a_trailing_slash_does_not_change_the_result(self):
        self.assertEqual(chat_completions_url("http://x:8080/"), "http://x:8080/v1/chat/completions")

    def test_an_empty_base_url_is_a_clear_error(self):
        with self.assertRaises(BackendError):
            chat_completions_url("")


class OpenAICompatibleTests(unittest.TestCase):
    def setUp(self):
        self.config = BackendConfig(model_name="gemma-3-12b", api_token=Secret(TOKEN), max_tokens=1536)
        self.backend = OpenAICompatibleBackend(self.config)

    def test_a_text_request_has_the_expected_shape(self):
        captured = collect(self.backend)
        self.backend.generate_text(system_prompt="SYS", user_prompt="USER")
        payload = captured["payload"]
        self.assertEqual(payload["model"], "gemma-3-12b")
        self.assertEqual(payload["max_tokens"], 1536)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "SYS"})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "USER"})
        self.assertFalse(payload["stream"])

    def test_a_vision_request_carries_the_image_as_a_data_url(self):
        captured = collect(self.backend)
        self.backend.analyze_image(image=IMAGE, system_prompt="SYS", user_prompt="USER")
        content = captured["payload"]["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "USER"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_the_instruction_precedes_the_image(self):
        captured = collect(self.backend)
        self.backend.analyze_image(image=IMAGE, system_prompt="", user_prompt="USER")
        self.assertEqual(captured["payload"]["messages"][0]["content"][0]["type"], "text")

    def test_an_empty_system_prompt_sends_no_system_turn(self):
        captured = collect(self.backend)
        self.backend.generate_text(system_prompt="   ", user_prompt="USER")
        self.assertEqual([m["role"] for m in captured["payload"]["messages"]], ["user"])

    def test_the_token_travels_in_the_authorization_header_only(self):
        captured = collect(self.backend)
        self.backend.generate_text(system_prompt="SYS", user_prompt="USER")
        self.assertEqual(captured["headers"]["Authorization"], f"Bearer {TOKEN}")
        self.assertNotIn(TOKEN, json.dumps(captured["payload"]))

    def test_no_token_means_no_authorization_header(self):
        # Several local servers reject an empty bearer outright.
        backend = OpenAICompatibleBackend(BackendConfig(model_name="m"))
        self.assertEqual(backend.headers(), {})

    def test_text_is_pulled_out_of_the_reply(self):
        self.assertEqual(
            self.backend.extract_text({"choices": [{"message": {"content": "hello"}}]}), "hello"
        )

    def test_structured_content_in_the_reply_is_joined(self):
        reply = {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}
        self.assertEqual(self.backend.extract_text(reply), "ab")

    def test_a_reply_with_no_choices_is_an_error(self):
        with self.assertRaises(BackendError):
            self.backend.extract_text({})

    def test_an_error_never_quotes_the_token(self):
        try:
            self.backend.extract_text({"note": f"key {TOKEN} rejected"})
        except BackendError as exc:
            self.assertNotIn(TOKEN, str(exc))
        else:  # pragma: no cover - the call above always raises
            self.fail("expected a BackendError")


class GeminiTests(unittest.TestCase):
    def setUp(self):
        self.config = BackendConfig(provider="gemini", model_name="gemini-2.5-flash",
                                    api_token=Secret(TOKEN), max_tokens=900, base_url="")
        self.backend = GeminiBackend(self.config)

    def test_the_endpoint_names_the_configured_model(self):
        # Not hard-coded: Gemini model names turn over on their own schedule.
        self.assertTrue(self.backend.endpoint().endswith("/models/gemini-2.5-flash:generateContent"))

    def test_a_missing_model_is_a_clear_error(self):
        with self.assertRaises(BackendError):
            GeminiBackend(BackendConfig(provider="gemini", base_url="")).endpoint()

    def test_the_key_goes_in_a_header_not_the_url(self):
        # A credential in a URL ends up in proxy logs and shell history.
        self.assertEqual(self.backend.headers(), {"x-goog-api-key": TOKEN})
        self.assertNotIn(TOKEN, self.backend.endpoint())

    def test_a_missing_key_says_which_field_is_missing(self):
        backend = GeminiBackend(BackendConfig(provider="gemini", model_name="m", base_url=""))
        with self.assertRaises(BackendError) as caught:
            backend.headers()
        self.assertIn("api_token", str(caught.exception))
        self.assertIn("max_tokens", str(caught.exception))

    def test_the_payload_uses_gemini_field_names(self):
        captured = collect(self.backend)
        self.backend.analyze_image(image=IMAGE, system_prompt="SYS", user_prompt="USER")
        payload = captured["payload"]
        self.assertEqual(payload["systemInstruction"], {"parts": [{"text": "SYS"}]})
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 900)
        parts = payload["contents"][0]["parts"]
        self.assertEqual(parts[0], {"text": "USER"})
        self.assertEqual(parts[1]["inline_data"]["mime_type"], "image/png")

    def test_a_text_request_carries_no_image_part(self):
        captured = collect(self.backend)
        self.backend.generate_text(system_prompt="SYS", user_prompt="USER")
        self.assertEqual(captured["payload"]["contents"][0]["parts"], [{"text": "USER"}])

    def test_text_is_joined_across_parts(self):
        reply = {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}
        self.assertEqual(self.backend.extract_text(reply), "ab")

    def test_a_blocked_prompt_reports_the_reason(self):
        with self.assertRaises(BackendError) as caught:
            self.backend.extract_text({"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []})
        self.assertIn("SAFETY", str(caught.exception))

    def test_hitting_max_tokens_names_the_setting(self):
        reply = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}
        with self.assertRaises(BackendError) as caught:
            self.backend.extract_text(reply)
        self.assertIn("max_tokens", str(caught.exception))


class FactoryTests(unittest.TestCase):
    def test_llama_cpp_and_lm_studio_share_one_client(self):
        # The whole point of the openai-compatible layer: no second copy to drift.
        for provider in ("llama_cpp", "openai_compatible"):
            with self.subTest(provider=provider):
                self.assertIsInstance(get_backend(BackendConfig(provider=provider)), OpenAICompatibleBackend)

    def test_gemini_gets_its_own_client(self):
        self.assertIsInstance(get_backend(BackendConfig(provider="gemini", base_url="")), GeminiBackend)

    def test_a_launched_llama_server_is_addressed_by_host_and_port(self):
        config = BackendConfig(provider="llama_cpp", server_mode="launch_local",
                               base_url="http://unused:1", host="127.0.0.1", port=8099)
        self.assertTrue(get_backend(config).endpoint().startswith("http://127.0.0.1:8099/"))

    def test_connect_existing_uses_base_url(self):
        config = BackendConfig(provider="llama_cpp", server_mode="connect_existing",
                               base_url="http://elsewhere:1234", port=8099)
        self.assertTrue(get_backend(config).endpoint().startswith("http://elsewhere:1234/"))


class DebugOutputTests(unittest.TestCase):
    def test_image_bytes_are_elided_from_a_request_summary(self):
        payload = {"messages": [{"content": [{"image_url": {"url": "data:image/png;base64," + "A" * 5000}}]}]}
        summary = summarize_payload(payload)
        self.assertNotIn("A" * 200, summary)
        self.assertIn("elided", summary)

    def test_short_values_survive_the_summary(self):
        self.assertIn("gemma-3-12b", summarize_payload({"model": "gemma-3-12b"}))


if __name__ == "__main__":
    unittest.main()


class TransportTests(unittest.TestCase):
    """The failure paths, where a credential is most likely to escape.

    A 401 body routinely quotes the key it rejected, and an exception raised
    here lands in a ComfyUI console, then in a screenshot, then in an issue.
    """

    def post(self, opener):
        import urllib.request

        from imageir.backend import base

        original = urllib.request.urlopen
        urllib.request.urlopen = opener
        try:
            return base.post_json("http://server/v1/chat/completions", {"model": "m"},
                                  {"Authorization": f"Bearer {TOKEN}"}, 5.0,
                                  secrets=(Secret(TOKEN),))
        finally:
            urllib.request.urlopen = original

    def test_a_json_reply_is_decoded(self):
        class Response:
            def read(self):
                return b'{"ok": true}'

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        self.assertEqual(self.post(lambda request, timeout=None: Response()), {"ok": True})

    def test_an_http_error_body_is_masked(self):
        import urllib.error

        def opener(request, timeout=None):
            raise urllib.error.HTTPError(
                "http://server", 401, "Unauthorized", {},
                __import__("io").BytesIO(f'{{"error":"invalid key {TOKEN}"}}'.encode()),
            )

        with self.assertRaises(BackendError) as caught:
            self.post(opener)
        message = str(caught.exception)
        self.assertNotIn(TOKEN, message)
        self.assertIn("401", message)

    def test_an_unreachable_server_says_so(self):
        import urllib.error

        def opener(request, timeout=None):
            raise urllib.error.URLError("Connection refused")

        with self.assertRaises(BackendError) as caught:
            self.post(opener)
        self.assertIn("could not reach", str(caught.exception))

    def test_a_timeout_names_the_setting(self):
        def opener(request, timeout=None):
            raise TimeoutError()

        with self.assertRaises(BackendError) as caught:
            self.post(opener)
        self.assertIn("timed out", str(caught.exception))

    def test_a_non_json_reply_is_reported_with_an_excerpt(self):
        class Response:
            def read(self):
                return b"<html>502 Bad Gateway</html>"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with self.assertRaises(BackendError) as caught:
            self.post(lambda request, timeout=None: Response())
        self.assertIn("did not return JSON", str(caught.exception))
        self.assertIn("502", str(caught.exception))

    def test_the_default_sender_is_wired_to_post_json(self):
        # Without this, a backend built for real use would silently have no transport.
        backend = OpenAICompatibleBackend(BackendConfig(model_name="m"))
        self.assertEqual(backend._send.__func__.__name__, "_default_sender")
