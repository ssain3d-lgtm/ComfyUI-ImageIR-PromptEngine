import unittest

import harness  # noqa: F401

from imageir.dsl import parse_dsl
from imageir.lexicon import MICRO_MOTIONS
from imageir.motion import available_motions, plan_motion

FULL = parse_dsl(
    "subject.identity = a woman\n"
    "subject.pose     = seated\n"
    "subject.gaze     = toward camera\n"
    "subject.hands    = resting on the lap\n"
    "subject.hair     = shoulder-length\n"
)


class AllowlistTests(unittest.TestCase):
    def test_the_allowlist_is_exactly_the_five_from_the_rules(self):
        self.assertEqual(
            {str(m["key"]) for m in MICRO_MOTIONS},
            {"blink", "breathing", "finger_movement", "hair_movement", "weight_shift"},
        )

    def test_all_five_are_available_on_a_fully_described_subject(self):
        self.assertEqual(len(available_motions(FULL)), 5)

    def test_anything_outside_the_allowlist_is_rejected_by_name(self):
        accepted, rejected = plan_motion(FULL, ["raise_arm"])
        self.assertEqual(accepted, ())
        self.assertEqual(rejected[0].key, "raise_arm")
        self.assertIn("allowlist", rejected[0].reason)

    def test_a_rejection_is_reported_rather_than_dropped(self):
        _accepted, rejected = plan_motion(FULL, ["blink", "walk_away"])
        self.assertEqual([r.key for r in rejected], ["walk_away"])

    def test_duplicates_collapse(self):
        accepted, _ = plan_motion(FULL, ["blink", "blink"])
        self.assertEqual(len(accepted), 1)

    def test_blank_requests_are_ignored(self):
        accepted, rejected = plan_motion(FULL, ["", "  "])
        self.assertEqual((accepted, rejected), ((), ()))


class ContradictionTests(unittest.TestCase):
    def test_closed_eyes_rule_out_a_blink(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.eyes = eyes closed")
        _accepted, rejected = plan_motion(ir, ["blink"])
        self.assertIn("contradicts subject.eyes", rejected[0].reason)

    def test_hands_out_of_frame_rule_out_finger_movement(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.hands = hands out of frame")
        _accepted, rejected = plan_motion(ir, ["finger_movement"])
        self.assertIn("contradicts", rejected[0].reason)

    def test_a_shaved_head_rules_out_hair_movement(self):
        ir = parse_dsl("subject.identity = a person\nsubject.hair = shaved head")
        _accepted, rejected = plan_motion(ir, ["hair_movement"])
        self.assertIn("contradicts", rejected[0].reason)

    def test_lying_down_rules_out_a_weight_shift(self):
        ir = parse_dsl("subject.identity = a person\nsubject.pose = lying down")
        _accepted, rejected = plan_motion(ir, ["weight_shift"])
        self.assertIn("contradicts", rejected[0].reason)

    def test_an_uncertain_wording_can_still_contradict(self):
        # The hedge is what a prompt would say, so it is what a motion has to
        # be consistent with.
        ir = parse_dsl("subject.identity = a person\nsubject.hair ~ shaved head")
        _accepted, rejected = plan_motion(ir, ["hair_movement"])
        self.assertIn("contradicts", rejected[0].reason)


class AnchorTests(unittest.TestCase):
    def test_a_missing_anchor_blocks_the_motion_under_the_strict_reading(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.pose = seated")
        _accepted, rejected = plan_motion(ir, ["hair_movement"])
        self.assertIn("no IMAGE_IR anchor", rejected[0].reason)
        self.assertIn("rule 1", rejected[0].reason)

    def test_the_literal_reading_of_rule_6_needs_no_anchor(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.pose = seated")
        accepted, rejected = plan_motion(ir, ["hair_movement"], require_anchor=False)
        self.assertEqual(rejected, ())
        self.assertIsNone(accepted[0].anchor)
        self.assertIn("rule 6", accepted[0].trace)

    def test_an_anchor_recorded_as_absent_blocks_the_motion_either_way(self):
        # "looked for and not there" is a stronger statement than "not
        # mentioned", so the loose reading cannot override it.
        ir = parse_dsl("subject.identity = a person\nsubject.hair !")
        for strict in (True, False):
            with self.subTest(require_anchor=strict):
                _accepted, rejected = plan_motion(ir, ["hair_movement"], require_anchor=strict)
                self.assertIn("absent", rejected[0].reason)

    def test_an_accepted_motion_names_the_fact_that_licensed_it(self):
        accepted, _ = plan_motion(FULL, ["hair_movement"])
        self.assertEqual(accepted[0].anchor, "subject.hair")
        self.assertEqual(accepted[0].trace, "subject.hair")

    def test_available_motions_narrows_with_the_ir(self):
        ir = parse_dsl("subject.identity = a woman\nsubject.pose = seated")
        self.assertEqual(set(available_motions(ir)), {"breathing", "weight_shift"})


if __name__ == "__main__":
    unittest.main()
