"""The ComfyUI surface: node contracts, widget wiring, and the failure modes."""

import sys
import unittest

import harness

nodes = harness.load_node_package()
NODES = nodes.NODE_CLASS_MAPPINGS
DISPLAY = nodes.NODE_DISPLAY_NAME_MAPPINGS

IR_TEXT = """\
@frame viewer
@laterality unconfirmed
subject.identity      = a woman
subject.pose          = seated
subject.gaze          = toward camera
subject.hands         = resting on the lap
subject.hair          = shoulder-length
wardrobe.top          = blouse
wardrobe.top_material ~ smooth | satin, silk
camera.shot           = medium shot
"""


def build_ir(text=IR_TEXT, **kwargs):
    return NODES["ImageIRFromText"]().build(text, **kwargs)[0]


class MappingTests(unittest.TestCase):
    def test_every_node_has_a_display_name(self):
        self.assertEqual(set(NODES), set(DISPLAY))

    def test_the_mappings_are_reachable_from_the_package_root(self):
        # ComfyUI imports these two names and nothing else.
        self.assertTrue(NODES)
        self.assertIn("ImageIRPromptEngine", NODES)

    def test_every_node_declares_the_comfyui_contract(self):
        for name, cls in NODES.items():
            with self.subTest(node=name):
                self.assertTrue(callable(getattr(cls, "INPUT_TYPES", None)))
                self.assertTrue(cls.RETURN_TYPES)
                self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))
                self.assertTrue(hasattr(cls, cls.FUNCTION))
                self.assertTrue(cls.CATEGORY)
                self.assertTrue(cls.DESCRIPTION)

    def test_every_widget_has_a_tooltip(self):
        for name, cls in NODES.items():
            spec = cls.INPUT_TYPES()
            for group in ("required", "optional"):
                for widget, definition in spec.get(group, {}).items():
                    # Sockets carry no widget, so they carry no tooltip.
                    if definition[0] in ("IMAGE_IR", "IMAGEIR_BACKEND", "IMAGE"):
                        continue
                    with self.subTest(node=name, widget=widget):
                        self.assertIn("tooltip", definition[1] if len(definition) > 1 else {})

    def test_the_node_function_accepts_every_declared_widget(self):
        import inspect

        for name, cls in NODES.items():
            spec = cls.INPUT_TYPES()
            declared = set(spec.get("required", {})) | set(spec.get("optional", {}))
            parameters = set(inspect.signature(getattr(cls, cls.FUNCTION)).parameters) - {"self"}
            with self.subTest(node=name):
                self.assertEqual(declared - parameters, set())


class FromTextTests(unittest.TestCase):
    def test_the_default_widget_text_parses(self):
        default = NODES["ImageIRFromText"].INPUT_TYPES()["required"]["ir_text"][1]["default"]
        ir = build_ir(default)
        self.assertTrue(ir.facts)

    def test_json_is_detected_automatically(self):
        ir = build_ir('{"sections": {"subject": {"gaze": "toward camera"}}}')
        self.assertEqual(ir.get("subject.gaze").value, "toward camera")

    def test_the_format_can_be_forced(self):
        with self.assertRaises(ValueError):
            build_ir("subject.gaze = toward camera", format="json")

    def test_the_json_output_round_trips(self):
        node = NODES["ImageIRFromText"]()
        ir, text = node.build(IR_TEXT)
        self.assertEqual(node.build(text)[0].to_dict(), ir.to_dict())

    def test_empty_text_is_a_clear_error(self):
        with self.assertRaises(ValueError) as caught:
            build_ir("   ")
        self.assertIn("empty", str(caught.exception))

    def test_a_parse_error_is_raised_as_a_value_error_for_comfyui(self):
        with self.assertRaises(ValueError) as caught:
            build_ir("this is not a fact")
        self.assertIn("IMAGE_IR", str(caught.exception))

    def test_merge_into_extends_an_existing_document(self):
        base = build_ir("subject.identity = a woman")
        ir = build_ir("subject.gaze = toward camera", merge_into=base)
        self.assertEqual(ir.get("subject.identity").value, "a woman")
        self.assertEqual(ir.get("subject.gaze").value, "toward camera")

    def test_a_conflicting_observation_raises_by_default(self):
        base = build_ir("subject.gaze = toward camera")
        with self.assertRaises(ValueError):
            build_ir("subject.gaze = away from camera", merge_into=base)


class MergeNodeTests(unittest.TestCase):
    def test_it_merges_and_reports_json(self):
        a = build_ir("subject.identity = a woman")
        b = build_ir("subject.pose = seated")
        ir, text = NODES["ImageIRMerge"]().run(a, b)
        self.assertEqual(len(ir.facts), 2)
        self.assertIn("seated", text)

    def test_a_conflict_is_raised_as_a_value_error(self):
        a = build_ir("subject.gaze = toward camera")
        b = build_ir("subject.gaze = away from camera")
        with self.assertRaises(ValueError):
            NODES["ImageIRMerge"]().run(a, b)

    def test_the_uncertain_policy_collapses_the_conflict(self):
        a = build_ir("subject.gaze = toward camera")
        b = build_ir("subject.gaze = away from camera")
        ir, _ = NODES["ImageIRMerge"]().run(a, b, on_conflict="uncertain")
        self.assertTrue(ir.get("subject.gaze").is_uncertain)


class PromptEngineTests(unittest.TestCase):
    def setUp(self):
        self.ir = build_ir()
        self.node = NODES["ImageIRPromptEngine"]()

    def test_it_returns_the_six_declared_outputs(self):
        out = self.node.run(self.ir)
        self.assertEqual(len(out), len(NODES["ImageIRPromptEngine"].RETURN_TYPES))
        self.assertTrue(out[0])

    def test_motion_toggles_reach_the_prompt(self):
        self.assertIn("blink", self.node.run(self.ir, blink=True)[0])
        self.assertNotIn("blink", self.node.run(self.ir)[0])

    def test_a_motion_with_no_anchor_is_dropped_and_explained(self):
        bare = build_ir("subject.identity = a woman\nsubject.pose = seated")
        prompt, *_rest, trace, _audit = self.node.run(bare, hair_movement=True)
        self.assertNotIn("hair", prompt)
        self.assertIn("hair_movement", trace)

    def test_the_literal_reading_of_rule_6_can_be_selected(self):
        bare = build_ir("subject.identity = a woman\nsubject.pose = seated")
        prompt = self.node.run(bare, hair_movement=True, require_motion_anchor=False)[0]
        self.assertIn("hair", prompt)

    def test_an_uncertain_attribute_is_hedged_in_every_output(self):
        prompt, h1, h2, *_ = self.node.run(self.ir)
        for text in (prompt, h1, h2):
            self.assertNotIn("satin", text)
        self.assertIn("smooth blouse", prompt)

    def test_directives_survive_and_are_marked_in_the_trace(self):
        prompt, *_rest, trace, _audit = self.node.run(self.ir, directives="5 seconds, 16:9")
        self.assertIn("5 seconds", prompt)
        self.assertIn("directive", trace)

    def test_a_describing_directive_is_filtered_out(self):
        prompt = self.node.run(self.ir, directives="holding a red umbrella")[0]
        self.assertNotIn("umbrella", prompt)

    def test_error_mode_stops_the_run_on_a_describing_directive(self):
        with self.assertRaises(ValueError) as caught:
            self.node.run(self.ir, directives="holding a red umbrella", on_violation="error")
        self.assertIn("rule 1", str(caught.exception))

    def test_report_only_passes_the_text_through(self):
        prompt = self.node.run(self.ir, directives="holding a red umbrella", on_violation="report_only")[0]
        self.assertIn("umbrella", prompt)

    def test_the_audit_output_is_a_report(self):
        self.assertIn("IMAGE_IR grounding audit", self.node.run(self.ir)[5])

    def test_the_negative_prompt_holds_only_what_the_ir_rules_out(self):
        negative = self.node.run(self.ir)[3]
        self.assertIn("away from camera", negative)
        # An unconfirmed reading is not an exclusion: "we could not tell whether
        # it is satin" must never become "it is not satin".
        self.assertNotIn("satin", negative)
        self.assertNotIn("silk", negative)


class GuardNodeTests(unittest.TestCase):
    def setUp(self):
        self.ir = build_ir()
        self.node = NODES["ImageIRGroundingGuard"]()

    def test_a_clean_prompt_passes_through_untouched(self):
        prompt, _report, count, clean = self.node.run(self.ir, "a woman, seated, medium shot")
        self.assertTrue(clean)
        self.assertEqual(count, 0)
        self.assertIn("a woman", prompt)

    def test_the_rules_own_bad_example_is_stripped(self):
        prompt, _report, count, clean = self.node.run(
            self.ir, "she turns from looking away toward the camera"
        )
        self.assertFalse(clean)
        self.assertEqual(count, 1)
        self.assertEqual(prompt, "")

    def test_error_mode_stops_the_run(self):
        with self.assertRaises(ValueError):
            self.node.run(self.ir, "in a satin blouse", on_violation="error")

    def test_report_only_keeps_the_text_and_still_counts(self):
        prompt, _report, count, clean = self.node.run(
            self.ir, "in a satin blouse", on_violation="report_only"
        )
        self.assertIn("satin", prompt)
        self.assertEqual(count, 1)
        self.assertFalse(clean)

    def test_style_terms_can_be_refused(self):
        self.assertTrue(self.node.run(self.ir, "a woman, cinematic")[3])
        self.assertFalse(self.node.run(self.ir, "a woman, cinematic", allow_style_terms=False)[3])

    def test_an_empty_prompt_is_clean(self):
        self.assertTrue(self.node.run(self.ir, "")[3])

    def test_the_engine_output_survives_the_guard(self):
        # The two nodes are meant to be chained; a prompt the engine produced
        # must never be something the guard then rejects.
        engine = NODES["ImageIRPromptEngine"]()
        prompt = engine.run(self.ir, blink=True, breathing=True, directives="5 seconds, 16:9")[0]
        _text, report, count, clean = self.node.run(self.ir, prompt)
        self.assertTrue(clean, report)
        self.assertEqual(count, 0)


def inspect(ir):
    """Unwrap the OUTPUT_NODE payload into the declared return tuple."""
    out = NODES["ImageIRInspect"]().run(ir)
    return out["result"]


class InspectTests(unittest.TestCase):
    def test_it_reports_through_the_output_node_payload(self):
        out = NODES["ImageIRInspect"]().run(build_ir())
        self.assertEqual(len(out["result"]), len(NODES["ImageIRInspect"].RETURN_TYPES))
        self.assertEqual(out["ui"]["text"], [out["result"][1]])

    def test_it_summarises_certainty_and_geometry(self):
        _json_text, summary, motions = inspect(build_ir())
        self.assertIn("viewer-relative", summary)
        self.assertIn("NOT confirmed", summary)
        self.assertIn("= toward camera", summary)
        self.assertIn("~ smooth", summary)
        self.assertIn("never emit: satin, silk", summary)
        self.assertIn("blink", motions)

    def test_it_lists_the_rules_it_enforces(self):
        summary = inspect(build_ir())[1]
        for number in range(1, 10):
            self.assertIn(f"{number}. ", summary)

    def test_an_ir_with_no_anchors_reports_no_motions(self):
        _json_text, _summary, motions = inspect(build_ir("scene.setting = a studio"))
        self.assertIn("none", motions)

    def test_notes_are_shown_but_flagged_as_never_composed(self):
        summary = inspect(build_ir("@note phone capture\nsubject.a = x"))[1]
        self.assertIn("phone capture", summary)
        self.assertIn("never composed", summary)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# The nodes added in the second pass: backends, the analyzer, MiniMax H3.
# ---------------------------------------------------------------------------

TOKEN = "sk-test-fake"

ANALYZER_REPLY = """Here is my reading.

```json
{"sections": {
  "subject": {"identity": {"value": "a woman", "certainty": "observed", "confidence": 0.95},
              "gaze": {"value": "toward camera", "certainty": "observed", "confidence": 0.9},
              "hair": {"value": "shoulder-length", "certainty": "observed", "confidence": 0.8}},
  "wardrobe": {"top": {"value": "blouse", "certainty": "observed", "confidence": 0.85},
               "top_material": {"value": null, "certainty": "uncertain", "hedge": "smooth",
                                "candidates": ["satin", "silk"], "confidence": 0.35,
                                "evidence": "even sheen"}}}}
```
"""

TINY_IMAGE = [[[[0.4, 0.3, 0.3], [0.5, 0.4, 0.4]], [[0.2, 0.2, 0.2], [0.6, 0.5, 0.5]]]]


def backend_config(**kwargs):
    defaults = dict(provider="openai_compatible", model_name="gemma-3-12b",
                    base_url="http://127.0.0.1:1234", api_token=TOKEN)
    defaults.update(kwargs)
    return NODES["ImageIRBackendConfig"]().run(**defaults)


class BackendConfigNodeTests(unittest.TestCase):
    def test_it_returns_a_config_and_a_readable_summary(self):
        config, summary = backend_config()
        self.assertEqual(config.model_name, "gemma-3-12b")
        self.assertIn("provider", summary)

    def test_the_summary_never_carries_the_token(self):
        _config, summary = backend_config()
        self.assertNotIn(TOKEN, summary)
        self.assertIn("***", summary)

    def test_api_token_and_max_tokens_stay_separate(self):
        config, _summary = backend_config(api_token=TOKEN, max_tokens=4096)
        self.assertEqual(config.api_token.reveal(), TOKEN)
        self.assertEqual(config.max_tokens, 4096)

    def test_a_llama_launch_config_shows_the_command_it_would_run(self):
        _config, summary = backend_config(provider="llama_cpp", server_mode="launch_local",
                                          gguf_model_path="/m/model.gguf", mmproj_path="/m/mmproj.gguf")
        self.assertIn("--mmproj /m/mmproj.gguf", summary)
        self.assertIn("--n-gpu-layers", summary)

    def test_the_shown_command_never_carries_the_token(self):
        _config, summary = backend_config(provider="llama_cpp", server_mode="launch_local",
                                          gguf_model_path="/m/model.gguf", api_token=TOKEN)
        self.assertNotIn(TOKEN, summary)

    def test_an_invalid_setting_is_a_node_error_not_a_crash(self):
        with self.assertRaises(ValueError):
            backend_config(max_tokens=0)


class BackendControlNodeTests(unittest.TestCase):
    def test_a_non_llama_provider_is_told_there_is_nothing_to_launch(self):
        config, _summary = backend_config(provider="gemini", model_name="gemini-2.5-flash", base_url="")
        _config, status, running = NODES["ImageIRBackendControl"]().run(config, action="status")
        self.assertIn("does not launch", status)
        self.assertFalse(running)

    def test_status_on_a_free_port_reports_stopped(self):
        config, _summary = backend_config(provider="llama_cpp", server_mode="launch_local",
                                          gguf_model_path="/m/model.gguf", port=59997)
        _config, status, running = NODES["ImageIRBackendControl"]().run(config, action="status")
        self.assertIn("stopped", status)
        self.assertFalse(running)

    def test_starting_with_a_bad_path_fails_cleanly(self):
        # A bad path must be a node result, never an exception out of ComfyUI.
        config, _summary = backend_config(provider="llama_cpp", server_mode="launch_local",
                                          llama_server_path="/nowhere/llama-server",
                                          gguf_model_path="/nowhere/model.gguf", port=59998)
        _config, status, running = NODES["ImageIRBackendControl"]().run(config, action="start")
        self.assertFalse(running)
        self.assertIn("/nowhere/llama-server", status)

    def test_stopping_something_we_never_started_is_not_an_error(self):
        config, _summary = backend_config(provider="llama_cpp", gguf_model_path="/m/model.gguf", port=59999)
        _config, status, _running = NODES["ImageIRBackendControl"]().run(config, action="stop")
        self.assertIn("nothing to stop", status)


class AnalyzerNodeTests(unittest.TestCase):
    def analyze(self, reply=ANALYZER_REPLY, **kwargs):
        """Run the analyzer node against a stubbed transport."""
        import imageir.backend as backend_package

        config, _summary = backend_config()
        node = NODES["ImageIRAnalyzer"]()
        original = backend_package.get_backend
        module = sys.modules[NODES["ImageIRAnalyzer"].__module__]

        def stub(cfg, sender=None):
            return original(cfg, lambda url, payload, headers, timeout: {
                "choices": [{"message": {"content": reply}}], "model": "gemma-3-12b",
                "usage": {"total_tokens": 700},
            })

        module.get_backend = stub
        try:
            return node.run(TINY_IMAGE, config, **kwargs)
        finally:
            module.get_backend = original

    def test_an_image_becomes_an_ir_without_anyone_typing_one(self):
        ir, ir_json, _low, summary, _debug = self.analyze()
        self.assertEqual(ir.get("subject.gaze").value, "toward camera")
        self.assertIn("toward camera", ir_json)
        self.assertIn("identity=a woman", summary)

    def test_uncertainty_survives_the_node(self):
        ir, *_rest = self.analyze()
        material = ir.get("wardrobe.top_material")
        self.assertTrue(material.is_uncertain)
        self.assertEqual(material.candidates, ("satin", "silk"))

    def test_low_confidence_attributes_are_listed(self):
        _ir, _json, low, _summary, _debug = self.analyze(confidence_threshold=0.6)
        self.assertIn("wardrobe.top_material", low)
        self.assertIn("0.35", low)

    def test_the_threshold_is_honoured(self):
        _ir, _json, low, _summary, _debug = self.analyze(confidence_threshold=0.1)
        self.assertIn("no attribute", low)

    def test_the_debug_output_never_carries_the_token(self):
        _ir, _json, _low, _summary, debug = self.analyze()
        self.assertNotIn(TOKEN, debug)

    def test_the_debug_output_names_the_endpoint_and_model(self):
        _ir, _json, _low, _summary, debug = self.analyze()
        self.assertIn("127.0.0.1:1234", debug)
        self.assertIn("gemma-3-12b", debug)

    def test_a_malformed_reply_is_a_node_error_with_the_reason(self):
        with self.assertRaises(ValueError) as caught:
            self.analyze(reply="I cannot describe this image.")
        self.assertIn("no JSON object", str(caught.exception))

    def test_each_detail_level_runs(self):
        for detail in ("fast", "balanced", "detailed"):
            with self.subTest(detail=detail):
                ir, *_rest = self.analyze(analysis_detail=detail)
                self.assertTrue(ir.facts)

    def test_a_missing_image_is_a_clear_error(self):
        config, _summary = backend_config()
        with self.assertRaises(ValueError) as caught:
            NODES["ImageIRAnalyzer"]().run(None, config)
        self.assertIn("IMAGE", str(caught.exception))


class MiniMaxH3NodeTests(unittest.TestCase):
    def setUp(self):
        self.ir = build_ir()
        self.node = NODES["MiniMaxH3PromptComposer"]()

    def test_the_official_fields_come_back_in_order(self):
        prompt, description, soundscape, music, _trace, _audit = self.node.run(self.ir)
        positions = [prompt.index(f"{f}:") for f in
                     ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")]
        self.assertEqual(positions, sorted(positions))
        self.assertIn(description, prompt)
        self.assertEqual(soundscape, "N/A")
        self.assertEqual(music, "N/A")

    def test_it_opens_with_the_reference_sentence(self):
        self.assertTrue(self.node.run(self.ir)[0].startswith("For the target video, at 0.00 seconds"))

    def test_an_uncertain_material_is_never_sharpened(self):
        prompt = self.node.run(self.ir)[0]
        self.assertIn("smooth blouse", prompt)
        self.assertNotIn("satin", prompt)

    def test_subject_and_camera_motion_do_not_bleed_together(self):
        _prompt, description, *_rest = self.node.run(self.ir, blink=True, camera_motion="slow_pan_left")
        self.assertIn("blink", description)
        self.assertIn("The camera pans slowly toward the viewer-left", description)

    def test_a_refused_motion_is_explained_in_the_trace(self):
        bare = build_ir("subject.identity = a woman\nsubject.pose = seated")
        _prompt, description, _s, _m, trace, _audit = self.node.run(bare, hair_movement=True)
        self.assertNotIn("hair", description)
        self.assertIn("hair_movement", trace)

    def test_an_ungrounded_style_is_filtered(self):
        prompt = self.node.run(self.ir, style="on a rooftop in Tokyo")[0]
        self.assertNotIn("Tokyo", prompt)

    def test_error_mode_stops_the_run(self):
        with self.assertRaises(ValueError):
            self.node.run(self.ir, style="on a rooftop in Tokyo", on_violation="error")


class GuardFormatModeTests(unittest.TestCase):
    def setUp(self):
        self.ir = build_ir()
        self.document = NODES["MiniMaxH3PromptComposer"]().run(
            self.ir, blink=True, overall_soundscape="Quiet room tone."
        )[0]

    def test_h3_mode_keeps_the_format_and_finds_nothing_to_remove(self):
        text, report, count, clean = NODES["ImageIRGroundingGuard"]().run(
            self.ir, self.document, prompt_format="minimax_h3"
        )
        self.assertTrue(clean, report)
        self.assertEqual(count, 0)
        self.assertIn("non_diegetic_music:", text)

    def test_plain_mode_would_shred_the_same_document(self):
        _text, _report, count, clean = NODES["ImageIRGroundingGuard"]().run(
            self.ir, self.document, prompt_format="plain"
        )
        self.assertFalse(clean)
        self.assertGreater(count, 0)

    def test_h3_mode_still_catches_a_claim_injected_into_the_description(self):
        tampered = self.document.replace("medium shot", "medium shot, holding a red umbrella")
        text, _report, count, clean = NODES["ImageIRGroundingGuard"]().run(
            self.ir, tampered, prompt_format="minimax_h3"
        )
        self.assertFalse(clean)
        self.assertEqual(count, 1)
        self.assertNotIn("umbrella", text)
        self.assertIn("integrated_multimodal_description:", text)
