"""The MiniMax H3 I2VA format, and the grounding rules it still has to obey."""

import unittest

import harness  # noqa: F401

from imageir.dsl import parse_dsl
from imageir.guard import audit
from imageir.h3 import (
    CAMERA_MOTIONS,
    FIELD_ORDER,
    REFERENCE_LINE,
    audit_h3_document,
    compose_h3,
)

EXAMPLE = parse_dsl(
    """
    @frame viewer
    @laterality unconfirmed

    subject.identity      = a woman
    subject.pose          = seated
    subject.gaze          = toward camera
    subject.hands         = resting on the lap
    subject.hair          = shoulder-length
    wardrobe.top          = blouse
    wardrobe.top_material ~ smooth | satin, silk
    scene.setting         = a plain studio backdrop
    lighting.key          = soft light from the viewer-left
    camera.shot           = medium shot
    """
)


class FormatTests(unittest.TestCase):
    """The parts MiniMax's guide fixes. A rename here is a different prompt."""

    def test_the_reference_sentence_is_the_official_one(self):
        self.assertEqual(
            REFERENCE_LINE,
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.",
        )

    def test_the_rendered_prompt_opens_with_it(self):
        self.assertTrue(compose_h3(EXAMPLE).render().startswith(REFERENCE_LINE))

    def test_the_three_fields_appear_in_the_official_order(self):
        rendered = compose_h3(EXAMPLE).render()
        positions = [rendered.index(f"{name}:") for name in FIELD_ORDER]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(
            FIELD_ORDER,
            ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"),
        )

    def test_fields_are_label_colon_space(self):
        rendered = compose_h3(EXAMPLE).render()
        for name in FIELD_ORDER:
            with self.subTest(field=name):
                self.assertIn(f"{name}: ", rendered)

    def test_blocks_are_separated_by_a_blank_line(self):
        rendered = compose_h3(EXAMPLE).render()
        self.assertIn("\n\nintegrated_multimodal_description:", rendered)
        self.assertIn("\n\noverall_soundscape:", rendered)
        self.assertIn("\n\nnon_diegetic_music:", rendered)

    def test_the_shot_label_is_bracketed(self):
        self.assertIn("[Shot 1]", compose_h3(EXAMPLE).integrated_multimodal_description)

    def test_the_description_binds_itself_to_the_supplied_picture(self):
        self.assertIn("<Picture 1>", compose_h3(EXAMPLE).integrated_multimodal_description)

    def test_empty_audio_fields_become_not_applicable(self):
        prompt = compose_h3(EXAMPLE)
        self.assertEqual(prompt.overall_soundscape, "N/A")
        self.assertEqual(prompt.non_diegetic_music, "N/A")

    def test_authored_audio_is_kept_verbatim(self):
        prompt = compose_h3(EXAMPLE, overall_soundscape="Rain on glass.", non_diegetic_music="Low strings.")
        self.assertEqual(prompt.overall_soundscape, "Rain on glass.")
        self.assertIn("non_diegetic_music: Low strings.", prompt.render())


class MotionSeparationTests(unittest.TestCase):
    def test_subject_motion_and_camera_motion_are_separate_fields(self):
        prompt = compose_h3(EXAMPLE, motions=["blink"], camera_motion="slow_push_in")
        self.assertIn("blink", prompt.subject_motion)
        self.assertNotIn("blink", prompt.camera_motion)
        self.assertIn("camera", prompt.camera_motion.lower())
        self.assertNotIn("camera pushes", prompt.subject_motion)

    def test_camera_motion_is_written_as_an_action_not_a_trailing_label(self):
        # The guide asks for lens movement as natural English inside the shot.
        prompt = compose_h3(EXAMPLE, camera_motion="slow_pan_left")
        self.assertTrue(prompt.camera_motion.startswith("The camera"))
        self.assertTrue(prompt.camera_motion.endswith("."))

    def test_camera_motion_stays_viewer_relative(self):
        for key in ("slow_pan_left", "slow_pan_right"):
            with self.subTest(key=key):
                self.assertIn("viewer-", CAMERA_MOTIONS[key])

    def test_static_is_the_default(self):
        self.assertIn("locked-off", compose_h3(EXAMPLE).camera_motion)

    def test_an_unknown_camera_motion_is_rejected(self):
        with self.assertRaises(ValueError):
            compose_h3(EXAMPLE, camera_motion="crash_zoom")

    def test_only_allowlisted_subject_motion_reaches_the_prompt(self):
        prompt = compose_h3(EXAMPLE, motions=["blink", "backflip"])
        self.assertIn("blink", prompt.integrated_multimodal_description)
        self.assertEqual([r.key for r in prompt.rejected_motions], ["backflip"])
        self.assertNotIn("backflip", prompt.render())


class GroundingTests(unittest.TestCase):
    """The nine rules, restated against the H3 composer."""

    def test_the_composed_description_passes_its_own_audit(self):
        for motions in ([], ["blink"], ["blink", "breathing", "hair_movement"]):
            for camera in CAMERA_MOTIONS:
                with self.subTest(motions=motions, camera=camera):
                    self.assertTrue(compose_h3(EXAMPLE, motions=motions, camera_motion=camera).clean)

    def test_an_uncertain_material_is_never_sharpened(self):
        rendered = compose_h3(EXAMPLE).render()
        self.assertIn("smooth blouse", rendered)
        self.assertNotIn("satin", rendered)
        self.assertNotIn("silk", rendered)

    def test_viewer_relative_geometry_survives(self):
        self.assertIn("viewer-left", compose_h3(EXAMPLE).integrated_multimodal_description)

    def test_subject_relative_laterality_is_never_introduced(self):
        description = compose_h3(EXAMPLE, motions=["blink"]).integrated_multimodal_description.lower()
        for phrase in ("her left", "her right", "his left", "his right"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, description)

    def test_an_observed_gaze_is_held_not_arrived_at(self):
        # The rules' own example: toward camera must never become a turn.
        description = compose_h3(EXAMPLE, motions=["blink"]).integrated_multimodal_description
        self.assertIn("gaze stays toward camera", description)
        for phrase in ("turns from", "looking away", "begins to"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, description)

    def test_an_absent_attribute_is_never_described(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.jewellery !")
        self.assertNotIn("jewellery", compose_h3(ir).integrated_multimodal_description)

    def test_a_style_directive_that_describes_the_image_is_caught(self):
        prompt = compose_h3(EXAMPLE, style="holding a red umbrella")
        self.assertNotIn("umbrella", prompt.render())
        self.assertFalse(prompt.clean)

    def test_error_mode_stops_on_an_ungrounded_style(self):
        with self.assertRaises(ValueError):
            compose_h3(EXAMPLE, style="on a rooftop in Tokyo", on_violation="error")

    def test_report_only_keeps_the_text_and_still_reports(self):
        prompt = compose_h3(EXAMPLE, style="on a rooftop in Tokyo", on_violation="report_only")
        self.assertIn("rooftop", prompt.render())
        self.assertFalse(prompt.clean)

    def test_a_micro_motion_with_no_anchor_is_refused(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.pose = seated")
        prompt = compose_h3(ir, motions=["hair_movement"])
        self.assertNotIn("hair", prompt.integrated_multimodal_description)
        self.assertIn("hair_movement", prompt.trace())


class DocumentAuditTests(unittest.TestCase):
    """Rule 9 over a structured document, without shredding the structure."""

    def setUp(self):
        self.document = compose_h3(
            EXAMPLE, motions=["blink"], camera_motion="slow_push_in",
            overall_soundscape="Rain taps the glass; a chair creaks.",
            non_diegetic_music="Sparse piano.",
        ).render()

    def test_the_plain_guard_would_destroy_the_format(self):
        # Establishes why the H3-aware path exists at all.
        self.assertFalse(audit(EXAMPLE, self.document).clean)

    def test_the_h3_aware_audit_passes_a_clean_document(self):
        result, _filtered = audit_h3_document(EXAMPLE, self.document)
        self.assertTrue(result.clean, result.report())

    def test_every_field_name_survives_the_audit(self):
        _result, filtered = audit_h3_document(EXAMPLE, self.document)
        for name in FIELD_ORDER:
            with self.subTest(field=name):
                self.assertIn(f"{name}:", filtered)

    def test_the_reference_sentence_survives(self):
        _result, filtered = audit_h3_document(EXAMPLE, self.document)
        self.assertIn(REFERENCE_LINE, filtered)

    def test_markers_survive(self):
        _result, filtered = audit_h3_document(EXAMPLE, self.document)
        self.assertIn("[Shot 1]", filtered)
        self.assertIn("<Picture 1>", filtered)

    def test_audio_is_not_judged_as_visual_content(self):
        # A still records no sound, so there is nothing to ground a soundscape
        # against; auditing it visually would reject every honest one.
        _result, filtered = audit_h3_document(EXAMPLE, self.document)
        self.assertIn("Rain taps the glass", filtered)
        self.assertIn("Sparse piano", filtered)

    def test_an_ungrounded_claim_injected_into_the_description_is_still_caught(self):
        tampered = self.document.replace("medium shot", "medium shot, holding a red umbrella")
        result, filtered = audit_h3_document(EXAMPLE, tampered)
        self.assertFalse(result.clean)
        self.assertNotIn("umbrella", filtered)
        self.assertIn("integrated_multimodal_description:", filtered)

    def test_a_sharpened_material_injected_into_the_description_is_caught(self):
        tampered = self.document.replace("smooth blouse", "satin blouse")
        result, _filtered = audit_h3_document(EXAMPLE, tampered)
        self.assertEqual([f.rule for f in result.findings], [3])

    def test_a_subject_relative_side_injected_into_the_description_is_caught(self):
        tampered = self.document.replace("hands resting", "her left hand resting")
        result, _filtered = audit_h3_document(EXAMPLE, tampered)
        self.assertIn(4, [f.rule for f in result.findings])

    def test_an_invented_prior_state_injected_into_the_description_is_caught(self):
        tampered = self.document.replace("gaze stays toward camera", "she turns from looking away toward the camera")
        result, _filtered = audit_h3_document(EXAMPLE, tampered)
        self.assertIn(5, [f.rule for f in result.findings])


class TraceTests(unittest.TestCase):
    def test_the_trace_separates_grounded_content_from_authored_content(self):
        trace = compose_h3(EXAMPLE, motions=["blink"], camera_motion="slow_push_in").trace()
        self.assertIn("grounded in IMAGE_IR", trace)
        self.assertIn("authored, not derived from the image", trace)

    def test_the_trace_names_the_ir_facts_used(self):
        self.assertIn("subject.gaze", compose_h3(EXAMPLE).trace())

    def test_unused_facts_are_listed(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.jewellery !")
        self.assertIn("subject.jewellery", compose_h3(ir).trace())


if __name__ == "__main__":
    unittest.main()
