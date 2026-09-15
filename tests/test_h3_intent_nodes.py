import json
import os
import unittest
from unittest.mock import patch
from pathlib import Path

import harness
from test_intent import intent_data

NODES = harness.load_node_package().NODE_CLASS_MAPPINGS


class IntentNodeTests(unittest.TestCase):
    def test_import_router_composer_guard_graph(self):
        intent, _ = NODES["H3UserIntentFromJSON"]().run(json.dumps(intent_data()))
        mode, _ = NODES["H3ModeRouter"]().run("AUTO", user_intent=intent)
        prompt, text, trace, frames = NODES["MiniMaxH3ModeComposer"]().run(mode, intent)
        self.assertEqual(frames, 158)
        guarded, report = NODES["ImageIRProvenanceGuard"]().run(prompt)["result"]
        self.assertEqual(guarded, text)
        self.assertIn("USER_INTENT", trace)
        self.assertIn("clean", report)
        repaired, report = NODES["ImageIRProvenanceGuard"]().run(prompt, candidate=text + " handbag", on_violation="filter")["result"]
        self.assertEqual(repaired, text)
        with self.assertRaises(ValueError):
            NODES["ImageIRProvenanceGuard"]().run(prompt, candidate=text + " handbag")

    def test_reference_builder(self):
        ir = NODES["ImageIRFromText"]().build("subject.identity = a woman")[0]
        pack, raw = NODES["H3ReferencePack"]().run("Picture 1", "image", "identity", "1", image_ir=ir)
        self.assertIn("REFERENCE_PACK_v1", raw)
        pack, _ = NODES["H3ReferencePack"]().run("Video 1", "video", "motion", "1", reference_pack=pack)
        self.assertEqual(len(pack.references), 2)
        mode, _ = NODES["H3ModeRouter"]().run("AUTO", reference_pack=pack)
        self.assertEqual(mode, "Ref2VA")

    def test_author_node_reuses_backend(self):
        package = harness.load_node_package()
        module = __import__(package.__name__ + ".image_ir_intent_nodes", fromlist=["get_backend"])
        config = NODES["ImageIRBackendConfig"]().run()[0]
        from imageir.intent import parse_intent
        with patch.object(module, "author_intent", return_value=parse_intent(intent_data())) as author:
            result, raw = NODES["H3UserIntentAuthor"]().run(config, "She waves.", 6.0, "AUTO", 1)
        self.assertTrue(author.called)
        self.assertIn("USER_INTENT_v1", raw)
        self.assertEqual(result.raw_request, "She waves.")

    def test_env_token_and_masked_output(self):
        key = "IMAGEIR_TEST_TOKEN"
        token = "fake-env-private"
        with patch.dict(os.environ, {key: token}):
            config, debug = NODES["ImageIRBackendConfig"]().run(api_token_env=key)
        self.assertEqual(config.api_token.reveal(), token)
        self.assertNotIn(token, debug)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            NODES["ImageIRBackendConfig"]().run(api_token_env=key)
        with patch.dict(os.environ, {key: token}), self.assertRaises(ValueError):
            NODES["ImageIRBackendConfig"]().run(api_token_env=key, api_token="fake-direct")

    def test_direct_token_warning(self):
        _, debug = NODES["ImageIRBackendConfig"]().run(api_token="fake-direct")
        self.assertIn("workflow", debug)
        self.assertNotIn("fake-direct", debug)

    def test_shipped_five_mode_workflow_executes_real_node_links(self):
        raw = (Path(harness.ROOT) / "example_workflows/image-ir-h3-five-modes.json").read_text(encoding="utf-8")
        workflow = json.loads(raw)
        links = {link[0]: link for link in workflow["links"]}
        results, modes, guarded = {}, [], 0
        for node in workflow["nodes"]:
            cls = NODES[node["type"]]
            spec = cls.INPUT_TYPES()
            widget_names = [name for group in ("required", "optional") for name, definition in spec.get(group, {}).items()
                            if isinstance(definition[0], list) or definition[0] in ("STRING", "INT", "FLOAT", "BOOLEAN")]
            args = dict(zip(widget_names, node["widgets_values"], strict=True))
            for slot, socket in enumerate(node["inputs"]):
                if socket["link"] is None:
                    continue
                _, source, source_slot, target, target_slot, dtype = links[socket["link"]]
                self.assertEqual((target, target_slot), (node["id"], slot))
                self.assertEqual(socket["type"], dtype)
                args[socket["name"]] = results[source][source_slot]
            result = getattr(cls(), cls.FUNCTION)(**args)
            results[node["id"]] = result["result"] if isinstance(result, dict) else result
            if node["type"] == "MiniMaxH3ModeComposer":
                modes.append(result[0].mode)
            if node["type"] == "ImageIRProvenanceGuard":
                self.assertIn("clean", result["result"][1])
                self.assertTrue(result["ui"]["text"][0])
                guarded += 1
        self.assertEqual(modes, ["T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"])
        self.assertEqual(guarded, 5)

    def test_invalid_guard_option_and_env_name(self):
        with self.assertRaises(ValueError):
            NODES["ImageIRProvenanceGuard"]().run(None, on_violation="ignore")
        with self.assertRaises(ValueError):
            NODES["ImageIRBackendConfig"]().run(api_token_env="not a name")
