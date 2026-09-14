import unittest

import harness  # noqa: F401  (puts the repository root on sys.path)

from imageir.schema import (
    ABSENT,
    OBSERVED,
    UNCERTAIN,
    Fact,
    IRError,
    ImageIR,
    MergeConflict,
    merge_ir,
    parse_ir,
    tokenize,
)


class ParseShorthandTests(unittest.TestCase):
    def test_bare_string_is_an_observation(self):
        ir = parse_ir({"sections": {"subject": {"gaze": "toward camera"}}})
        fact = ir.get("subject.gaze")
        self.assertEqual(fact.certainty, OBSERVED)
        self.assertEqual(fact.value, "toward camera")

    def test_tilde_prefix_marks_uncertainty_and_keeps_only_the_hedge(self):
        ir = parse_ir({"sections": {"wardrobe": {"top_material": "~smooth"}}})
        fact = ir.get("wardrobe.top_material")
        self.assertEqual(fact.certainty, UNCERTAIN)
        self.assertIsNone(fact.value)
        self.assertEqual(fact.emitted_text(), "smooth")

    def test_bare_tilde_licenses_no_wording_at_all(self):
        ir = parse_ir({"sections": {"wardrobe": {"top_material": "~"}}})
        self.assertIsNone(ir.get("wardrobe.top_material").emitted_text())

    def test_null_is_absence(self):
        ir = parse_ir({"sections": {"subject": {"jewellery": None}}})
        self.assertEqual(ir.get("subject.jewellery").certainty, ABSENT)
        self.assertIsNone(ir.get("subject.jewellery").emitted_text())

    def test_certainty_is_inferred_from_the_object_shape(self):
        ir = parse_ir({"sections": {"w": {"a": {"value": "blue"}, "b": {"hedge": "pale"}}}})
        self.assertEqual(ir.get("w.a").certainty, OBSERVED)
        self.assertEqual(ir.get("w.b").certainty, UNCERTAIN)

    def test_json_text_and_mapping_agree(self):
        data = {"sections": {"subject": {"gaze": "toward camera"}}}
        self.assertEqual(parse_ir(data).to_dict(), parse_ir(parse_ir(data).to_json()).to_dict())

    def test_round_trip_preserves_candidates_and_frame(self):
        source = {
            "geometry": {"frame": "viewer", "subject_laterality_confirmed": True},
            "sections": {"w": {"m": {"certainty": "uncertain", "hedge": "smooth", "candidates": ["satin"], "frame": "subject"}}},
            "notes": ["handheld"],
        }
        again = parse_ir(parse_ir(source).to_json())
        self.assertEqual(again.get("w.m").candidates, ("satin",))
        self.assertEqual(again.get("w.m").frame, "subject")
        self.assertTrue(again.subject_laterality_confirmed)
        self.assertEqual(again.notes, ("handheld",))


class ParseErrorTests(unittest.TestCase):
    def test_unknown_top_level_key_is_rejected(self):
        # A typo in subject_laterality_confirmed would otherwise leave the safe
        # default in place while the author believes laterality was confirmed.
        with self.assertRaises(IRError):
            parse_ir({"sections": {}, "geomety": {}})

    def test_unknown_geometry_key_is_rejected(self):
        with self.assertRaises(IRError):
            parse_ir({"geometry": {"subject_laterality": True}})

    def test_unknown_fact_key_is_rejected(self):
        with self.assertRaises(IRError) as caught:
            parse_ir({"sections": {"subject": {"gaze": {"value": "x", "certanty": "observed"}}}})
        self.assertIn("subject.gaze", str(caught.exception))

    def test_observed_needs_a_value(self):
        with self.assertRaises(IRError):
            parse_ir({"sections": {"subject": {"gaze": {"certainty": "observed"}}}})

    def test_absent_cannot_carry_a_value(self):
        with self.assertRaises(IRError):
            parse_ir({"sections": {"subject": {"gaze": {"certainty": "absent", "value": "x"}}}})

    def test_bad_frame_is_rejected(self):
        with self.assertRaises(IRError):
            parse_ir({"geometry": {"frame": "camera"}})

    def test_bad_certainty_is_rejected(self):
        with self.assertRaises(IRError):
            parse_ir({"sections": {"s": {"a": {"value": "x", "certainty": "probably"}}}})

    def test_empty_text_and_bad_json_report_clearly(self):
        for bad in ("", "   ", "{nope}"):
            with self.assertRaises(IRError):
                parse_ir(bad)

    def test_non_object_is_rejected(self):
        with self.assertRaises(IRError):
            parse_ir([1, 2, 3])


class FactTests(unittest.TestCase):
    def test_uncertain_never_emits_its_candidates(self):
        fact = Fact("w", "top_material", None, UNCERTAIN, hedge="smooth", candidates=("satin", "silk"))
        self.assertEqual(fact.emitted_text(), "smooth")
        self.assertEqual(fact.forbidden_specifics(), ("satin", "silk"))

    def test_an_uncertain_value_is_itself_forbidden_wording(self):
        # A value on an uncertain fact is the reading that could not be
        # confirmed, so it is a candidate, not something to emit.
        fact = Fact("w", "top_material", "satin", UNCERTAIN, hedge="smooth")
        self.assertEqual(fact.emitted_text(), "smooth")
        self.assertIn("satin", fact.forbidden_specifics())

    def test_observed_facts_have_no_forbidden_specifics(self):
        self.assertEqual(Fact("s", "gaze", "toward camera", OBSERVED).forbidden_specifics(), ())


class VocabularyTests(unittest.TestCase):
    def setUp(self):
        self.ir = parse_ir({
            "sections": {
                "subject": {"gaze": "toward camera"},
                "wardrobe": {"top_material": {"certainty": "uncertain", "hedge": "smooth", "candidates": ["satin"]}},
            }
        })

    def test_vocabulary_holds_emittable_wording_and_slot_names(self):
        self.assertLessEqual({"toward", "camera", "gaze", "smooth", "material", "top"}, set(self.ir.vocabulary()))

    def test_vocabulary_never_holds_a_candidate(self):
        self.assertNotIn("satin", self.ir.vocabulary())

    def test_trace_index_points_back_at_the_fact(self):
        self.assertEqual(self.ir.trace_index()["toward"], ("subject.gaze",))

    def test_function_words_drop_but_claims_survive(self):
        # "she" stays: it asserts how the subject presents, and the guard checks
        # it. Directional words stay too — half of a spatial observation.
        self.assertEqual(tokenize("she is in the room"), ("she", "room"))
        self.assertEqual(tokenize("gaze toward the camera"), ("gaze", "toward", "camera"))

    def test_sections_order_known_first_then_the_rest(self):
        ir = parse_ir({"sections": {"zebra": {"a": "x"}, "camera": {"b": "y"}, "subject": {"c": "z"}}})
        self.assertEqual(ir.sections(), ("subject", "camera", "zebra"))


class MergeTests(unittest.TestCase):
    def observed(self, value):
        return parse_ir({"sections": {"subject": {"gaze": value}}})

    def test_an_observation_fills_a_gap(self):
        base = parse_ir({"sections": {"subject": {"pose": "seated"}}})
        merged = merge_ir(base, self.observed("toward camera"))
        self.assertEqual(merged.get("subject.gaze").value, "toward camera")
        self.assertEqual(merged.get("subject.pose").value, "seated")

    def test_an_observation_sharpens_an_uncertain_attribute(self):
        base = parse_ir({"sections": {"subject": {"gaze": "~roughly forward"}}})
        merged = merge_ir(base, self.observed("toward camera"))
        self.assertEqual(merged.get("subject.gaze").certainty, OBSERVED)

    def test_an_observation_is_never_weakened(self):
        merged = merge_ir(self.observed("toward camera"), parse_ir({"sections": {"subject": {"gaze": "~forward"}}}))
        self.assertEqual(merged.get("subject.gaze").value, "toward camera")

    def test_conflicting_observations_raise_by_default(self):
        with self.assertRaises(MergeConflict):
            merge_ir(self.observed("toward camera"), self.observed("away from camera"))

    def test_keep_base_ignores_the_incoming_reading(self):
        merged = merge_ir(self.observed("toward camera"), self.observed("away from camera"), on_conflict="keep_base")
        self.assertEqual(merged.get("subject.gaze").value, "toward camera")

    def test_uncertain_policy_records_both_readings_as_candidates(self):
        merged = merge_ir(self.observed("toward camera"), self.observed("away from camera"), on_conflict="uncertain")
        fact = merged.get("subject.gaze")
        self.assertEqual(fact.certainty, UNCERTAIN)
        self.assertEqual(set(fact.forbidden_specifics()), {"toward camera", "away from camera"})
        self.assertIsNone(fact.emitted_text())

    def test_laterality_survives_only_if_both_documents_confirm_it(self):
        confirmed = parse_ir({"geometry": {"subject_laterality_confirmed": True}, "sections": {}})
        plain = parse_ir({"sections": {}})
        self.assertTrue(merge_ir(confirmed, confirmed).subject_laterality_confirmed)
        self.assertFalse(merge_ir(confirmed, plain).subject_laterality_confirmed)

    def test_disagreeing_frames_fall_back_to_viewer(self):
        viewer = parse_ir({"geometry": {"frame": "viewer"}, "sections": {}})
        subject = parse_ir({"geometry": {"frame": "subject"}, "sections": {}})
        self.assertEqual(merge_ir(subject, viewer).frame, "viewer")

    def test_bad_policy_is_rejected(self):
        with self.assertRaises(IRError):
            merge_ir(ImageIR(), ImageIR(), on_conflict="whatever")


if __name__ == "__main__":
    unittest.main()
