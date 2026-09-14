"""The nine grounding rules, each exercised on the wording it exists to reject."""

import unittest

import harness  # noqa: F401

from imageir.dsl import parse_dsl
from imageir.guard import RULE_TEXT, audit
from imageir.schema import parse_ir

# The IR from the rules' own example, plus enough context to write a prompt from.
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
    lighting.key          = soft light
    camera.shot           = medium shot
    """
)


def rules_broken(ir, prompt):
    return sorted({f.rule for f in audit(ir, prompt).findings})


class Rule1NoNewFactsTests(unittest.TestCase):
    def test_an_invented_object_is_rejected(self):
        self.assertIn(1, rules_broken(EXAMPLE, "a woman seated, holding a red umbrella"))

    def test_an_invented_setting_is_rejected(self):
        self.assertIn(1, rules_broken(EXAMPLE, "a woman seated on a rooftop in Tokyo"))

    def test_an_attribute_recorded_as_absent_cannot_be_described(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.jewellery !")
        findings = audit(ir, "a woman wearing jewellery").findings
        self.assertEqual([f.rule for f in findings], [1])
        self.assertIn("absent", findings[0].detail)

    def test_saying_an_absent_attribute_is_absent_agrees_with_the_ir(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.jewellery !")
        self.assertTrue(audit(ir, "a woman, no jewellery").clean)

    def test_a_gendered_pronoun_needs_the_ir_to_record_it(self):
        ir = parse_dsl("subject.identity = a person\nsubject.gaze = toward camera")
        findings = audit(ir, "she holds her position").findings
        self.assertEqual(findings[0].rule, 1)
        self.assertIn("presents", findings[0].detail)

    def test_a_gendered_pronoun_is_fine_once_the_ir_records_it(self):
        self.assertTrue(audit(EXAMPLE, "she is seated").clean)

    def test_the_pronoun_check_can_be_switched_off(self):
        ir = parse_dsl("subject.identity = a person\nsubject.pose = seated")
        self.assertTrue(audit(ir, "she is seated", check_pronouns=False).clean)

    def test_a_grounded_description_passes(self):
        self.assertTrue(audit(EXAMPLE, "a woman, seated, gaze toward camera, medium shot").clean)


class Rule2NoChangedAttributesTests(unittest.TestCase):
    def test_contradicting_an_observed_gaze_is_rejected(self):
        findings = audit(EXAMPLE, "a woman seated, looking away").findings
        self.assertEqual(findings[0].rule, 2)
        self.assertIn("gaze direction", findings[0].detail)

    def test_contradicting_an_observed_pose_is_rejected(self):
        self.assertIn(2, rules_broken(EXAMPLE, "a woman standing, gaze toward camera"))

    def test_contradicting_an_observed_shot_size_is_rejected(self):
        self.assertIn(2, rules_broken(EXAMPLE, "a woman seated, close-up"))

    def test_the_suggestion_names_the_observed_value(self):
        finding = audit(EXAMPLE, "a woman, eyes closed").findings[0]
        self.assertIn("toward camera", finding.suggestion)

    def test_restating_the_observed_pole_is_fine(self):
        self.assertTrue(audit(EXAMPLE, "a woman seated, gaze toward camera").clean)


class Rule3And7UncertaintyTests(unittest.TestCase):
    def test_the_rules_own_example_satin_blouse_is_rejected(self):
        findings = audit(EXAMPLE, "a woman in a satin blouse").findings
        self.assertEqual(findings[0].rule, 3)
        self.assertIn("satin", findings[0].detail)

    def test_the_suggested_fix_is_the_hedge(self):
        self.assertIn("smooth", audit(EXAMPLE, "a woman in a satin blouse").findings[0].suggestion)

    def test_the_rules_own_example_smooth_blouse_is_accepted(self):
        self.assertTrue(audit(EXAMPLE, "a woman, smooth blouse").clean)

    def test_omitting_the_material_entirely_is_accepted(self):
        self.assertTrue(audit(EXAMPLE, "a woman, blouse").clean)

    def test_a_specific_the_ir_never_listed_is_still_rejected(self):
        # "velvet" is not among the candidates; the slot-kind table is the
        # backstop for readings the IR author did not happen to write down.
        findings = audit(EXAMPLE, "a woman in a velvet blouse").findings
        self.assertEqual(findings[0].rule, 7)

    def test_an_observed_material_may_be_named(self):
        ir = parse_dsl("subject.identity = a woman\nwardrobe.top = blouse\nwardrobe.top_material = satin")
        self.assertTrue(audit(ir, "a woman in a satin blouse").clean)


class Rule4And8LateralityTests(unittest.TestCase):
    def test_subject_relative_sides_are_rejected_when_unconfirmed(self):
        findings = audit(EXAMPLE, "a woman, her left hand resting on the lap").findings
        self.assertEqual(findings[0].rule, 4)

    def test_a_bare_body_part_side_is_rejected_too(self):
        self.assertIn(4, rules_broken(EXAMPLE, "a woman, left hand resting on the lap"))

    def test_the_fix_offers_viewer_relative_wording(self):
        self.assertIn("viewer-left", audit(EXAMPLE, "a woman, her left hand visible").findings[0].suggestion)

    def test_viewer_relative_wording_passes(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.hand = on the viewer-left")
        self.assertTrue(audit(ir, "a woman, hand on the viewer-left").clean)

    def test_confirmed_laterality_permits_subject_relative_wording(self):
        ir = parse_dsl("@laterality confirmed\nsubject.identity = a woman\nsubject.hands = resting on the lap")
        self.assertTrue(audit(ir, "a woman, her hands resting on the lap").clean)


class Rule5NoInventedInitialStateTests(unittest.TestCase):
    def test_the_rules_own_example_is_rejected(self):
        findings = audit(EXAMPLE, "she turns from looking away toward the camera").findings
        self.assertEqual(findings[0].rule, 5)

    def test_the_rules_own_replacement_is_accepted(self):
        self.assertTrue(audit(EXAMPLE, "she maintains gentle eye contact with the camera").clean)

    def test_a_transition_is_caught_even_around_a_legal_motion(self):
        # "blink" is allowed; "begins to" is not, and the order of the checks
        # is what stops the legal half from excusing the illegal half.
        findings = audit(EXAMPLE, "she begins to blink").findings
        self.assertEqual(findings[0].rule, 5)

    def test_history_words_are_caught(self):
        for prompt in ("at first she is seated", "previously seated", "then she is seated"):
            with self.subTest(prompt=prompt):
                self.assertIn(5, rules_broken(EXAMPLE, prompt))

    def test_a_continuing_state_is_accepted(self):
        self.assertTrue(audit(EXAMPLE, "gaze stays toward camera").clean)


class Rule6MotionTests(unittest.TestCase):
    def test_every_allowed_micro_motion_passes(self):
        for prompt in (
            "a natural unhurried blink",
            "quiet breathing that barely moves the shoulders",
            "the faintest movement in the fingers",
            "a few strands of hair drifting",
            "a barely perceptible settling of weight",
        ):
            with self.subTest(prompt=prompt):
                self.assertTrue(audit(EXAMPLE, prompt).clean, prompt)

    def test_changing_gaze_direction_is_rejected(self):
        self.assertIn(6, rules_broken(EXAMPLE, "she shifts her gaze to the window"))

    def test_changing_pose_category_is_rejected(self):
        # No observed pose to contradict, so the ban on pose changes is what
        # has to catch this on its own.
        ir = parse_dsl("subject.identity = a woman\nsubject.gaze = toward camera")
        self.assertIn(6, rules_broken(ir, "she stands up"))

    def test_a_pose_change_that_contradicts_the_ir_is_caught_as_a_changed_attribute(self):
        # Both rules apply; the more specific diagnosis is the useful one,
        # because it can name the value the prompt should have kept.
        self.assertEqual(rules_broken(EXAMPLE, "she stands up"), [2])

    def test_raising_a_limb_is_rejected(self):
        self.assertIn(6, rules_broken(EXAMPLE, "she raises her arm"))

    def test_touching_a_new_body_part_is_rejected(self):
        self.assertIn(6, rules_broken(EXAMPLE, "she touches her hair"))

    def test_moving_to_another_location_is_rejected(self):
        self.assertIn(6, rules_broken(EXAMPLE, "she walks toward the backdrop"))

    def test_manipulating_an_object_is_rejected(self):
        self.assertIn(6, rules_broken(EXAMPLE, "she picks up a cup"))

    def test_a_micro_motion_that_contradicts_the_ir_is_rejected(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.eyes = eyes closed")
        findings = audit(ir, "a natural unhurried blink").findings
        self.assertEqual(findings[0].rule, 6)
        self.assertIn("contradicts", findings[0].detail)

    def test_a_micro_motion_with_no_anchor_is_rejected_under_the_strict_reading(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.pose = seated")
        findings = audit(ir, "a few strands of hair drifting").findings
        self.assertEqual(findings[0].rule, 6)
        self.assertIn("anchor", findings[0].detail)

    def test_the_literal_reading_of_rule_6_allows_it(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.pose = seated")
        self.assertTrue(audit(ir, "a few strands of hair drifting", require_anchor=False).clean)

    def test_an_observed_pose_is_a_description_not_a_movement(self):
        # "resting on the lap" matches a touching pattern, but the IR recorded
        # it, so it describes the frame rather than asking for a hand to move.
        self.assertTrue(audit(EXAMPLE, "hands resting on the lap").clean)


class Rule9FilterTests(unittest.TestCase):
    def test_only_the_offending_clause_is_removed(self):
        result = audit(EXAMPLE, "a woman, seated, in a satin blouse, medium shot")
        self.assertNotIn("satin", result.filtered_prompt)
        self.assertIn("a woman", result.filtered_prompt)
        self.assertIn("medium shot", result.filtered_prompt)

    def test_the_filtered_prompt_is_itself_clean(self):
        messy = (
            "she turns from looking away toward the camera, in a satin blouse, "
            "her left hand raised, holding a red umbrella, a natural unhurried blink, "
            "seated, medium shot"
        )
        once = audit(EXAMPLE, messy).filtered_prompt
        self.assertTrue(audit(EXAMPLE, once).clean, audit(EXAMPLE, once).report())

    def test_filtering_everything_yields_an_empty_prompt(self):
        self.assertEqual(audit(EXAMPLE, "holding a red umbrella on a rooftop").filtered_prompt, "")

    def test_punctuation_is_left_tidy(self):
        result = audit(EXAMPLE, "a woman, holding a red umbrella, seated")
        self.assertNotIn(",,", result.filtered_prompt)
        self.assertTrue(result.filtered_prompt.endswith("."))

    def test_an_empty_prompt_is_clean(self):
        self.assertTrue(audit(EXAMPLE, "").clean)

    def test_technical_directives_need_no_ir_trace(self):
        result = audit(EXAMPLE, "a woman, seated, 5 seconds, 16:9, static camera")
        self.assertTrue(result.clean, result.report())

    def test_style_wording_can_be_refused(self):
        prompt = "a woman, seated, cinematic film grain"
        self.assertTrue(audit(EXAMPLE, prompt).clean)
        self.assertFalse(audit(EXAMPLE, prompt, allow_style=False).clean)

    def test_the_report_names_the_rule_it_enforced(self):
        report = audit(EXAMPLE, "in a satin blouse").report()
        self.assertIn("rule 3", report)
        self.assertIn(RULE_TEXT[3], report)

    def test_the_report_lists_kept_clauses_with_their_sources(self):
        report = audit(EXAMPLE, "a woman, seated").report()
        self.assertIn("subject.identity", report)
        self.assertIn("clean", report)

    def test_ir_paths_used_are_reported(self):
        result = audit(EXAMPLE, "a woman, seated, medium shot")
        self.assertIn("subject.identity", result.ir_paths_used)
        self.assertIn("camera.shot", result.ir_paths_used)


class EquivalenceTests(unittest.TestCase):
    def test_a_paraphrase_of_an_observed_wording_traces_to_that_fact(self):
        result = audit(EXAMPLE, "holding eye contact with the camera")
        self.assertTrue(result.clean, result.report())
        self.assertIn("subject.gaze", result.ir_paths_used)

    def test_a_paraphrase_is_not_opened_by_an_uncertain_fact(self):
        ir = parse_ir({"sections": {"subject": {"gaze": {"certainty": "uncertain", "hedge": "forward"}}}})
        self.assertFalse(audit(ir, "eye contact with the camera").clean)


if __name__ == "__main__":
    unittest.main()
