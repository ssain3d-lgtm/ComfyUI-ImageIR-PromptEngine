import unittest

import harness  # noqa: F401

from imageir.compose import compose
from imageir.dsl import parse_dsl
from imageir.guard import audit

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
    camera.height         = eye level
    """
)


class TierTests(unittest.TestCase):
    def test_core_holds_the_core_and_nothing_else(self):
        core = compose(EXAMPLE).core
        self.assertIn("a woman", core)
        self.assertIn("seated", core)
        self.assertIn("toward camera", core)
        self.assertNotIn("blouse", core)
        self.assertNotIn("backdrop", core)

    def test_detail_holds_the_attributes_the_core_left_out(self):
        detail = compose(EXAMPLE).detail
        self.assertIn("smooth blouse", detail)
        self.assertIn("a plain studio backdrop", detail)
        self.assertIn("medium shot", detail)
        self.assertNotIn("a woman", detail)

    def test_final_contains_both_tiers(self):
        result = compose(EXAMPLE)
        self.assertIn("a woman", result.final)
        self.assertIn("smooth blouse", result.final)
        self.assertTrue(result.final.endswith("."))

    def test_an_empty_ir_composes_to_nothing_rather_than_something(self):
        result = compose(parse_dsl(""))
        self.assertEqual(result.final, "")


class GroundingTests(unittest.TestCase):
    def test_the_composer_passes_its_own_guard(self):
        # The invariant the whole design rests on: composition is subtractive,
        # so its output cannot contain a statement the audit would reject.
        for motions in ([], ["blink"], ["blink", "breathing", "hair_movement", "weight_shift"]):
            for style in ("sentence", "tags"):
                for continuity in ("auto", "always", "never"):
                    with self.subTest(motions=motions, style=style, continuity=continuity):
                        result = compose(EXAMPLE, motions=motions, style=style, continuity=continuity)
                        self.assertTrue(result.self_audit.clean, result.self_audit.report())

    def test_an_uncertain_material_is_hedged_not_resolved(self):
        final_prompt = compose(EXAMPLE).final
        self.assertIn("smooth blouse", final_prompt)
        self.assertNotIn("satin", final_prompt)
        self.assertNotIn("silk", final_prompt)

    def test_an_uncertain_attribute_with_no_hedge_is_left_out_entirely(self):
        ir = parse_dsl("subject.identity = a woman\nwardrobe.top = blouse\nwardrobe.top_material ~ | satin")
        final_prompt = compose(ir).final
        self.assertIn("blouse", final_prompt)
        self.assertNotIn("satin", final_prompt)

    def test_an_absent_attribute_is_never_composed(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.jewellery !")
        self.assertNotIn("jewellery", compose(ir).final)

    def test_notes_never_reach_the_prompt(self):
        ir = parse_dsl("@note shot on a phone at dusk\nsubject.identity = a woman")
        final_prompt = compose(ir).final
        self.assertNotIn("phone", final_prompt)
        self.assertNotIn("dusk", final_prompt)

    def test_viewer_relative_wording_survives_composition(self):
        self.assertIn("viewer-left", compose(EXAMPLE).final)


class QualifierTests(unittest.TestCase):
    def test_a_qualifier_folds_into_the_attribute_it_qualifies(self):
        result = compose(EXAMPLE)
        clause = next(c for c in result.clauses if "blouse" in c.text)
        self.assertEqual(clause.text, "smooth blouse")
        self.assertEqual(set(clause.paths), {"wardrobe.top", "wardrobe.top_material"})

    def test_qualifiers_stack_in_reading_order(self):
        ir = parse_dsl(
            "wardrobe.top = blouse\nwardrobe.top_material ~ smooth\nwardrobe.top_color = pale blue\n"
        )
        self.assertIn("pale blue smooth blouse", compose(ir).final)

    def test_an_orphaned_qualifier_is_left_out_and_reported(self):
        # "smooth" on its own describes nothing, so it is not floated into the
        # prompt alone — it shows up as an unused fact instead.
        ir = parse_dsl("subject.identity = a woman\nwardrobe.top_material ~ smooth")
        result = compose(ir)
        self.assertNotIn("smooth", result.final)
        self.assertIn("wardrobe.top_material", result.unused_paths)


class MotionAndContinuityTests(unittest.TestCase):
    def test_requested_motions_appear(self):
        final_prompt = compose(EXAMPLE, motions=["blink", "breathing"]).final
        self.assertIn("blink", final_prompt)
        self.assertIn("breathing", final_prompt)

    def test_a_rejected_motion_is_absent_from_the_prompt_but_named_in_the_trace(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.pose = seated")
        result = compose(ir, motions=["hair_movement"])
        self.assertNotIn("hair", result.final)
        self.assertIn("hair_movement", result.trace())

    def test_auto_continuity_waits_for_motion(self):
        self.assertNotIn("stays", compose(EXAMPLE).final)
        self.assertIn("stays", compose(EXAMPLE, motions=["blink"]).final)

    def test_always_and_never_override_auto(self):
        self.assertIn("stays", compose(EXAMPLE, continuity="always").final)
        self.assertNotIn("stays", compose(EXAMPLE, motions=["blink"], continuity="never").final)

    def test_continuity_restates_the_observed_value_verbatim(self):
        final_prompt = compose(EXAMPLE, continuity="always").final
        self.assertIn("gaze stays toward camera", final_prompt)
        self.assertIn("hands stay resting on the lap", final_prompt)

    def test_continuity_covers_only_attributes_a_motion_prompt_drifts_on(self):
        final_prompt = compose(EXAMPLE, continuity="always").final
        self.assertNotIn("setting stays", final_prompt)


class DirectiveTests(unittest.TestCase):
    def test_directives_are_kept_verbatim_and_marked_as_such(self):
        result = compose(EXAMPLE, directives="5 seconds, static camera, 16:9")
        self.assertIn("5 seconds", result.final)
        clause = next(c for c in result.clauses if c.kind == "directive")
        self.assertEqual(clause.paths, ())
        self.assertIn("directive", result.trace())

    def test_a_directive_that_describes_the_image_is_caught_by_the_audit(self):
        result = compose(EXAMPLE, directives="holding a red umbrella")
        self.assertFalse(result.self_audit.clean)
        self.assertNotIn("umbrella", result.self_audit.filtered_prompt)

    def test_an_empty_directive_adds_nothing(self):
        self.assertEqual(compose(EXAMPLE, directives="  ,  ").final, compose(EXAMPLE).final)


class NegativeTests(unittest.TestCase):
    def test_the_negative_holds_the_poles_the_ir_ruled_out(self):
        negative = compose(EXAMPLE).negative
        self.assertIn("away from camera", negative)
        self.assertIn("standing", negative)

    def test_the_negative_never_holds_what_the_ir_observed(self):
        self.assertNotIn("seated", compose(EXAMPLE).negative)

    def test_uncertain_candidates_never_become_negatives(self):
        # "we could not tell whether it is satin" is not "it is not satin".
        # Putting a weighed-but-unconfirmed reading into a negative prompt
        # invents an exclusion the image never supported, and steers generation
        # away from what may be the right answer.
        negative = compose(EXAMPLE).negative
        self.assertNotIn("satin", negative)
        self.assertNotIn("silk", negative)

    def test_the_candidates_are_still_kept_for_the_guard(self):
        # Not negating them is not the same as forgetting them: the guard still
        # needs them to stop a prompt from asserting one.
        self.assertEqual(EXAMPLE.get("wardrobe.top_material").forbidden_specifics(), ("satin", "silk"))

    def test_absent_attributes_become_negatives(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.jewellery !")
        self.assertIn("jewellery", compose(ir).negative)


class TraceTests(unittest.TestCase):
    def test_every_description_clause_names_its_source(self):
        for clause in compose(EXAMPLE, motions=["blink"]).clauses:
            if clause.kind in ("description", "continuity", "motion"):
                self.assertTrue(clause.paths, clause.text)

    def test_unused_facts_are_listed(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.jewellery !")
        self.assertIn("subject.jewellery", compose(ir).unused_paths)

    def test_the_trace_mentions_each_clause(self):
        result = compose(EXAMPLE)
        trace = result.trace()
        for clause in result.clauses:
            self.assertIn(clause.text, trace)


class ArgumentTests(unittest.TestCase):
    def test_a_bad_style_is_rejected(self):
        with self.assertRaises(ValueError):
            compose(EXAMPLE, style="freeform")

    def test_a_bad_continuity_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            compose(EXAMPLE, continuity="sometimes")

    def test_tags_style_drops_the_connectives(self):
        tags = compose(EXAMPLE, style="tags").final
        self.assertNotIn("wearing", tags)
        self.assertIn("blouse", tags)
        self.assertTrue(audit(EXAMPLE, tags).clean)


if __name__ == "__main__":
    unittest.main()
