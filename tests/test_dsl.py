import unittest

import harness  # noqa: F401

from imageir.dsl import parse_dsl
from imageir.schema import ABSENT, OBSERVED, UNCERTAIN, IRError


class DslTests(unittest.TestCase):
    def test_the_three_markers_map_to_the_three_certainties(self):
        ir = parse_dsl(
            "subject.gaze = toward camera\n"
            "wardrobe.top_material ~ smooth | satin, silk\n"
            "subject.jewellery !\n"
        )
        self.assertEqual(ir.get("subject.gaze").certainty, OBSERVED)
        self.assertEqual(ir.get("wardrobe.top_material").certainty, UNCERTAIN)
        self.assertEqual(ir.get("wardrobe.top_material").candidates, ("satin", "silk"))
        self.assertEqual(ir.get("subject.jewellery").certainty, ABSENT)

    def test_uncertain_without_candidates_is_allowed(self):
        ir = parse_dsl("wardrobe.top_material ~ smooth")
        self.assertEqual(ir.get("wardrobe.top_material").emitted_text(), "smooth")
        self.assertEqual(ir.get("wardrobe.top_material").candidates, ())

    def test_uncertain_with_no_hedge_licenses_nothing(self):
        ir = parse_dsl("wardrobe.top_material ~")
        self.assertIsNone(ir.get("wardrobe.top_material").emitted_text())

    def test_directives_set_geometry_and_notes(self):
        ir = parse_dsl("@frame subject\n@laterality confirmed\n@note phone capture\nsubject.a = x")
        self.assertEqual(ir.frame, "subject")
        self.assertTrue(ir.subject_laterality_confirmed)
        self.assertEqual(ir.notes, ("phone capture",))

    def test_laterality_defaults_to_unconfirmed(self):
        # The safe default has to be the one you get by saying nothing.
        self.assertFalse(parse_dsl("subject.a = x").subject_laterality_confirmed)

    def test_per_fact_frame_override(self):
        ir = parse_dsl("subject.hand_position = near the edge @subject")
        self.assertEqual(ir.get("subject.hand_position").frame, "subject")
        self.assertEqual(ir.get("subject.hand_position").value, "near the edge")

    def test_comments_and_blank_lines_are_ignored(self):
        ir = parse_dsl("# a comment\n\nsubject.a = x  # trailing\n")
        self.assertEqual(len(ir.facts), 1)
        self.assertEqual(ir.get("subject.a").value, "x")

    def test_a_later_line_replaces_an_earlier_one(self):
        ir = parse_dsl("subject.gaze = toward camera\nsubject.gaze = away from camera")
        self.assertEqual(len(ir.facts), 1)
        self.assertEqual(ir.get("subject.gaze").value, "away from camera")

    def test_values_may_contain_commas_and_punctuation(self):
        ir = parse_dsl("subject.pose = seated, torso upright, weight on the left hip")
        self.assertEqual(ir.get("subject.pose").value, "seated, torso upright, weight on the left hip")

    def test_the_line_number_is_reported(self):
        with self.assertRaises(IRError) as caught:
            parse_dsl("subject.a = x\nnonsense here\n")
        self.assertIn("line 2", str(caught.exception))

    def test_a_path_must_be_section_dot_attribute(self):
        for bad in ("gaze = toward camera", "subject = x", "sub ject.a = x"):
            with self.assertRaises(IRError):
                parse_dsl(bad)

    def test_observed_with_no_value_is_rejected_by_name(self):
        with self.assertRaises(IRError) as caught:
            parse_dsl("subject.gaze =")
        self.assertIn("absent", str(caught.exception))

    def test_unknown_directive_is_rejected(self):
        with self.assertRaises(IRError):
            parse_dsl("@laterallity confirmed")

    def test_bad_directive_values_are_rejected(self):
        for bad in ("@frame sideways", "@laterality maybe"):
            with self.assertRaises(IRError):
                parse_dsl(bad)

    def test_json_and_lines_can_describe_the_same_document(self):
        from imageir.schema import parse_ir

        lines = parse_dsl("@frame viewer\nsubject.gaze = toward camera\nwardrobe.m ~ smooth | satin")
        self.assertEqual(parse_ir(lines.to_json()).to_dict(), lines.to_dict())


if __name__ == "__main__":
    unittest.main()
