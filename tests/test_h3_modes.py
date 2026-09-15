import unittest

import harness
from imageir.dsl import parse_dsl
from imageir.intent import parse_intent
from imageir.references import Reference, ReferencePack, parse_reference_pack
from imageir.h3_modes import compose_mode, route_mode
from imageir.provenance import guard_document
from test_intent import intent_data


class ModeTests(unittest.TestCase):
    def setUp(self):
        self.first = parse_dsl("subject.identity = a woman\nsubject.gaze = toward camera\nsubject.pose = seated\nwardrobe.top = pale blouse")
        self.last = parse_dsl("subject.identity = a woman\nsubject.pose = standing\nscene.location = window")
        self.intent = parse_intent(intent_data())
        self.pack = ReferencePack((Reference("Picture 1", "image", "identity", "1", self.first),))

    def test_auto_routes(self):
        for kwargs, expected in [({}, "T2VA"), ({"first": self.first}, "I2VA"),
                                 ({"first": self.first, "last": self.last}, "FL2VA"),
                                 ({"last": self.last}, "L2VA"), ({"references": self.pack}, "Ref2VA")]:
            with self.subTest(expected=expected):
                self.assertEqual(route_mode(**kwargs), expected)

    def test_manual_mode_validates_inputs(self):
        self.assertEqual(route_mode("I2VA", first=self.first), "I2VA")
        for kwargs in [{"mode": "I2VA"}, {"mode": "unknown"}, {"mode": "T2VA", "first": self.first},
                       {"first": self.first, "references": self.pack}, {"mode": "Ref2VA"}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                route_mode(**kwargs)

    def test_t2va_and_timing(self):
        result = compose_mode(self.intent)
        self.assertEqual(result.mode, "T2VA")
        self.assertEqual(result.frames, 158)
        self.assertNotIn("Picture", result.render())
        self.assertIn("She waves.", result.render())
        self.assertTrue(guard_document(result).clean)

    def test_i2va_future_action_camera_audio_and_trace(self):
        data = intent_data()
        for key, text in [("requested_camera", "The camera slowly pulls back."),
                          ("requested_sound", "Bells ring."), ("requested_music", "N/A")]:
            data["inputs"][key] = text
            data[key] = [{"text": text, "input": key, "evidence": text}]
        result = compose_mode(parse_intent(data), first=self.first)
        self.assertTrue(guard_document(result).clean)
        for text in ["She waves.", "Bells ring.", "pulls back", "pale blouse", "0.00 seconds"]:
            self.assertIn(text, result.render())
        for source in ["IMAGE_IR", "USER_INTENT", "H3_RULE"]:
            self.assertIn(source, result.trace())

    def test_first_last_and_final_only_alignment(self):
        for first, mode, picture in [(self.first, "FL2VA", "Picture 2"), (None, "L2VA", "<Picture 1>")]:
            result = compose_mode(self.intent, first=first, last=self.last)
            self.assertEqual(result.mode, mode)
            self.assertIn(picture, result.render())
            self.assertIn("6.58-second", result.render())
            self.assertIn("standing", result.render())
            self.assertIn("window", result.render())

    def test_ref_six_sections_and_scoped_roles(self):
        other = parse_dsl("subject.identity = a man\nwardrobe.top = red coat\nscene.location = forest")
        pack = ReferencePack((Reference("Picture 1", "image", "identity", "1", self.first),
                              Reference("Picture 2", "image", "wardrobe", "1", other),
                              Reference("Video 1", "video", "motion", "1"),
                              Reference("Audio 1", "audio", "sound", "1")))
        result = compose_mode(self.intent, references=pack)
        self.assertEqual(tuple(name for name, _ in result.sections), (
            "subject_definitions", "summary", "retention_analysis", "detailed_description",
            "overall_soundscape", "non_diegetic_music"))
        self.assertIn("red coat", result.render())
        self.assertNotIn("a man", result.render())
        self.assertNotIn("forest", result.render())
        retention = dict(result.sections)["retention_analysis"]
        self.assertIn("<Subject 1>", retention)
        self.assertNotIn("<Picture 1>", retention)
        self.assertIn("<Video 1>", retention)
        self.assertIn("<Audio 1>", retention)

    def test_reference_roundtrip(self):
        self.assertEqual(parse_reference_pack(self.pack.to_json()), self.pack)

    def test_reference_validation(self):
        for args in [("Picture 0", "image", "identity", "1"), ("Video 1", "video", "wardrobe", "1"),
                     ("Audio 1", "audio", "identity", "1")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                Reference(*args)
        with self.assertRaises(ValueError):
            ReferencePack(self.pack.references * 2)

    def test_untraced_object_and_forged_prior_state_rejected(self):
        result = compose_mode(self.intent, first=self.first)
        for extra in [" A handbag appears.", " She turns from looking away and faces the camera."]:
            audit = guard_document(result, result.render() + extra)
            self.assertFalse(audit.clean)
            self.assertEqual(audit.filtered_prompt, result.render())

    def test_requested_prior_state_conflict_rejected(self):
        data = intent_data("She turns from looking away and then faces the camera.")
        data["requested_actions"][0]["text"] = data["raw_request"]
        with self.assertRaisesRegex(ValueError, "prior|start"):
            compose_mode(parse_intent(data), first=self.first)

    def test_multishot(self):
        data = intent_data()
        data["shot_count_hint"] = 3
        result = compose_mode(parse_intent(data))
        self.assertIn("[Shot 2] At 00:", result.render())
        self.assertIn("[Shot 3] At 00:", result.render())

    def test_uncertainty_is_not_sharpened(self):
        ir = parse_dsl("subject.identity = a woman\nwardrobe.material ~ smooth | satin, silk")
        text = compose_mode(self.intent, first=ir).render()
        self.assertIn("smooth", text)
        self.assertNotIn("satin", text)
        self.assertNotIn("silk", text)

    def test_ordinary_future_sequencing_is_allowed(self):
        for text in ["They slowly raise one hand in greeting.", "They walk forward, then wave."]:
            data = intent_data(text)
            data["requested_actions"][0]["text"] = text
            self.assertIn(text, compose_mode(parse_intent(data), first=self.first).render())

    def test_boundaries_in_all_visual_fields(self):
        for name, text, kwargs in [
            ("requested_actions", "At the end, they are seated.", {"last": self.last}),
            ("requested_camera", "Initially the subject is standing.", {"first": self.first}),
            ("requested_scene_progression", "Finally she sits down.", {"first": self.first, "last": self.last}),
        ]:
            data = intent_data(text)
            data[name] = [{"text": text, "input": "request", "evidence": text}]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "state|boundary"):
                compose_mode(parse_intent(data), **kwargs)

    def test_unanchored_boundary_is_rendered(self):
        data = intent_data("Ends seated.")
        data["requested_actions"] = []
        data["final_states"] = [{"text": "seated", "input": "request", "evidence": "seated", "path": "subject.pose"}]
        for kwargs in ({}, {"first": self.first}):
            result = compose_mode(parse_intent(data), **kwargs)
            self.assertIn("At the final frame, seated", result.render())
            self.assertIn("final_states[0]", result.trace())

    def test_body_role_does_not_compete_with_identity(self):
        identity = parse_dsl("subject.identity = a woman\nsubject.body = slender")
        body = parse_dsl("subject.body = muscular")
        pack = ReferencePack((Reference("Picture 1", "image", "identity", "1", identity),
                              Reference("Picture 2", "image", "body", "1", body)))
        text = compose_mode(self.intent, references=pack).render()
        self.assertIn("muscular", text)
        self.assertNotIn("slender", text)

    def test_typed_boundary_validation(self):
        for path, text in [("subject.pose", "seated"), ("subject.pose", "standing"), ("scene.missing", "garden")]:
            data = intent_data(text)
            data["requested_actions"] = []
            data["start_states"] = [{"text": text, "input": "request", "evidence": text, "path": path}]
            if text == "seated":
                self.assertIn("seated", compose_mode(parse_intent(data), first=self.first).render())
            else:
                with self.assertRaises(ValueError):
                    compose_mode(parse_intent(data), first=self.first)

    def test_summary_only_is_preserved_in_all_modes(self):
        data = intent_data("A second person enters carrying a suitcase.")
        data["requested_actions"] = []
        data["summary"] = [{"text": data["raw_request"], "input": "request", "evidence": data["raw_request"]}]
        for kwargs in ({}, {"first": self.first}, {"last": self.last}, {"first": self.first, "last": self.last}, {"references": self.pack}):
            self.assertIn(data["raw_request"], compose_mode(parse_intent(data), **kwargs).render())

    def test_no_unrequested_metadata_loss(self):
        data = intent_data("안녕하세요")
        data["requested_actions"] = []
        data["dialogue"] = [{"text": "안녕하세요", "input": "request", "evidence": "안녕하세요"}]
        result = compose_mode(parse_intent(data))
        self.assertIn("<d>[Korean] 안녕하세요</d>", result.render())
        self.assertIn("H3_RULE dialogue", result.trace())

    def test_grid_limits_and_forged_sections(self):
        from dataclasses import replace
        from imageir.h3_modes import grid_frames
        for value in (0, -1, 16, float("inf"), True):
            with self.assertRaises(ValueError):
                grid_frames(value)
        self.assertEqual(grid_frames(362 / 24), 362)
        self.assertEqual(grid_frames(0.1), 5)
        result = compose_mode(self.intent)
        forged = replace(result, sections=(("integrated_multimodal_description", "a handbag"),))
        self.assertFalse(guard_document(forged).clean)

    def test_role_variants_and_ambiguous_role_assignment(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.pose = seated\nsubject.body = slender\nwardrobe.top = coat\nscene.location = forest\ncamera.shot = wide")
        for role, expected in [("body", "slender"), ("pose", "seated"), ("composition", "wide"),
                               ("environment", "forest"), ("clothing", "coat")]:
            ref = Reference("Picture 1", "image", role, "1", ir)
            self.assertIn(expected, str([f.value for f in ref.scoped_ir().facts]))
        with self.assertRaises(ValueError):
            ReferencePack((Reference("Picture 1", "image", "identity"), Reference("Picture 2", "image", "identity")))
        result = compose_mode(self.intent, references=ReferencePack((Reference("Picture 1", "image", "style"),)))
        self.assertTrue(any("no analyzed" in w for w in result.warnings))
