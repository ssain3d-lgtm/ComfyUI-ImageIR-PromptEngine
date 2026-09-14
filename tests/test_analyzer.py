"""Reading a vision model's reply — the stage where a wrong repair invents facts."""

import unittest

import harness  # noqa: F401

from imageir.analyzer import (
    AnalyzerError,
    analyze,
    build_user_prompt,
    extract_json,
    find_json_object,
    flag_low_confidence,
    load_system_prompt,
)
from imageir.backend import BackendConfig, Secret
from imageir.backend.base import ImagePayload
from imageir.backend.openai_compatible import OpenAICompatibleBackend
from imageir.schema import IRError, parse_ir

TOKEN = "sk-test-fake"
IMAGE = ImagePayload(b"\x89PNG\r\n\x1a\nfake", "image/png")

GOOD_IR = """{
  "sections": {
    "subject": {"identity": {"value": "a woman", "certainty": "observed", "confidence": 0.95},
                "gaze": {"value": "toward camera", "certainty": "observed", "confidence": 0.9}},
    "wardrobe": {"top": {"value": "blouse", "certainty": "observed", "confidence": 0.88},
                 "top_material": {"value": null, "certainty": "uncertain", "hedge": "smooth",
                                  "candidates": ["satin", "silk"], "confidence": 0.4,
                                  "evidence": "even sheen, no visible weave"}}
  }
}"""


def backend_returning(text, config=None):
    config = config or BackendConfig(model_name="m", api_token=Secret(TOKEN))
    return OpenAICompatibleBackend(
        config,
        lambda url, payload, headers, timeout: {"choices": [{"message": {"content": text}}], "model": "m"},
    )


class ExtractionTests(unittest.TestCase):
    def test_clean_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_markdown_fenced_json(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_an_unlabelled_fence(self):
        self.assertEqual(extract_json('```\n{"a": 1}\n```'), {"a": 1})

    def test_reasoning_before_the_json(self):
        self.assertEqual(extract_json('Let me look at this image.\n\n{"a": 1}'), {"a": 1})

    def test_commentary_after_the_json(self):
        self.assertEqual(extract_json('{"a": 1}\n\nLet me know if you need more.'), {"a": 1})

    def test_prose_on_both_sides_of_a_fence(self):
        self.assertEqual(extract_json('First:\n```json\n{"a": 1}\n```\nDone.'), {"a": 1})

    def test_a_trailing_comma_is_repaired(self):
        self.assertEqual(extract_json('{"a": 1,}'), {"a": 1})

    def test_braces_inside_strings_do_not_end_the_scan(self):
        self.assertEqual(extract_json('{"a": "a { brace } inside"}'), {"a": "a { brace } inside"})

    def test_escaped_quotes_do_not_end_the_scan(self):
        self.assertEqual(extract_json(r'{"a": "he said \"hi\" {"}'), {"a": 'he said "hi" {'})

    def test_nested_objects_survive(self):
        self.assertEqual(extract_json('x {"a": {"b": {"c": 1}}} y'), {"a": {"b": {"c": 1}}})

    def test_no_json_at_all_is_a_useful_error(self):
        with self.assertRaises(AnalyzerError) as caught:
            extract_json("I am unable to analyse this image.")
        self.assertIn("unable to analyse", str(caught.exception))

    def test_an_empty_reply_is_a_useful_error(self):
        with self.assertRaises(AnalyzerError) as caught:
            extract_json("   ")
        self.assertIn("empty", str(caught.exception))

    def test_malformed_json_reports_where_it_broke(self):
        with self.assertRaises(AnalyzerError) as caught:
            extract_json('{"a": [1, 2}')
        message = str(caught.exception)
        self.assertIn("not valid JSON", message)
        self.assertIn("position", message)

    def test_a_json_array_is_rejected_rather_than_coerced(self):
        with self.assertRaises(AnalyzerError):
            extract_json("[1, 2, 3]")

    def test_an_error_excerpt_never_carries_a_secret(self):
        reply = f"my key is {TOKEN} and I cannot answer"
        with self.assertRaises(AnalyzerError) as caught:
            extract_json(reply, secrets=(Secret(TOKEN),))
        self.assertNotIn(TOKEN, str(caught.exception))

    def test_find_json_object_returns_none_when_there_is_none(self):
        self.assertIsNone(find_json_object("no braces here"))


class ForbiddenRepairTests(unittest.TestCase):
    """The repairs stop at syntax. These would be inventions."""

    def test_a_missing_value_is_not_supplied(self):
        with self.assertRaises(AnalyzerError):
            extract_json('{"value": }')

    def test_unquoted_prose_is_not_guessed_into_a_string(self):
        with self.assertRaises(AnalyzerError):
            extract_json("{value: a woman seated}")


class ConfidenceTests(unittest.TestCase):
    def test_confidence_is_validated_on_the_way_in(self):
        for bad in (1.5, -0.1, "high", True):
            with self.subTest(bad=bad), self.assertRaises(IRError):
                parse_ir({"sections": {"s": {"a": {"value": "x", "certainty": "observed", "confidence": bad}}}})

    def test_the_bounds_are_inclusive(self):
        for good in (0.0, 1.0, 0.5):
            with self.subTest(good=good):
                ir = parse_ir({"sections": {"s": {"a": {"value": "x", "certainty": "observed", "confidence": good}}}})
                self.assertEqual(ir.get("s.a").confidence, good)

    def test_verification_required_round_trips(self):
        source = {"sections": {"s": {"a": {"value": "x", "certainty": "observed", "verification_required": True,
                                           "evidence": "a note", "confidence": 0.3}}}}
        again = parse_ir(parse_ir(source).to_json()).get("s.a")
        self.assertTrue(again.verification_required)
        self.assertEqual(again.evidence, "a note")
        self.assertEqual(again.confidence, 0.3)

    def test_an_old_document_without_confidence_still_parses(self):
        ir = parse_ir({"sections": {"subject": {"gaze": "toward camera"}}})
        fact = ir.get("subject.gaze")
        self.assertIsNone(fact.confidence)
        self.assertFalse(fact.verification_required)

    def test_an_unstated_confidence_is_not_invented_on_the_way_out(self):
        # Emitting 1.0 for "nobody said" would be fabricated precision.
        self.assertNotIn("confidence", parse_ir({"sections": {"s": {"a": "x"}}}).to_json())

    def test_an_unstated_confidence_is_not_treated_as_low(self):
        self.assertFalse(parse_ir({"sections": {"s": {"a": "x"}}}).get("s.a").is_low_confidence)

    def test_flagging_marks_only_what_is_below_the_threshold(self):
        ir = parse_ir({"sections": {"s": {
            "high": {"value": "x", "certainty": "observed", "confidence": 0.9},
            "low": {"value": "y", "certainty": "observed", "confidence": 0.3},
            "silent": "z",
        }}})
        updated, flagged = flag_low_confidence(ir, 0.6)
        self.assertEqual([f.path for f in flagged], ["s.low"])
        self.assertTrue(updated.get("s.low").verification_required)
        self.assertFalse(updated.get("s.high").verification_required)
        self.assertFalse(updated.get("s.silent").verification_required)

    def test_flagging_never_changes_a_value_or_a_certainty(self):
        ir = parse_ir({"sections": {"s": {"a": {"value": "x", "certainty": "observed", "confidence": 0.1}}}})
        updated, _ = flag_low_confidence(ir, 0.6)
        self.assertEqual(updated.get("s.a").value, "x")
        self.assertEqual(updated.get("s.a").certainty, "observed")

    def test_an_explicit_request_is_honoured_whatever_the_score(self):
        ir = parse_ir({"sections": {"s": {"a": {"value": "x", "certainty": "observed",
                                                "confidence": 0.99, "verification_required": True}}}})
        _updated, flagged = flag_low_confidence(ir, 0.6)
        self.assertEqual([f.path for f in flagged], ["s.a"])


class AnalyzeTests(unittest.TestCase):
    def test_a_full_pass_produces_a_valid_ir(self):
        result = analyze(backend_returning(GOOD_IR), IMAGE)
        self.assertEqual(result.ir.get("subject.gaze").value, "toward camera")
        self.assertEqual(result.ir.get("wardrobe.top_material").hedge, "smooth")

    def test_uncertainty_survives_the_round_trip(self):
        result = analyze(backend_returning(GOOD_IR), IMAGE)
        material = result.ir.get("wardrobe.top_material")
        self.assertTrue(material.is_uncertain)
        self.assertIsNone(material.value)
        self.assertEqual(material.candidates, ("satin", "silk"))

    def test_low_confidence_attributes_are_reported(self):
        result = analyze(backend_returning(GOOD_IR), IMAGE, confidence_threshold=0.6)
        self.assertEqual([f.path for f in result.flagged], ["wardrobe.top_material"])
        report = result.low_confidence_report(0.6)
        self.assertIn("wardrobe.top_material", report)
        self.assertIn("0.40", report)
        self.assertIn("even sheen", report)

    def test_nothing_flagged_says_so_plainly(self):
        self.assertIn("no attribute", analyze(backend_returning(GOOD_IR), IMAGE, confidence_threshold=0.0)
                      .low_confidence_report(0.0))

    def test_the_compact_summary_lists_every_attribute(self):
        summary = analyze(backend_returning(GOOD_IR), IMAGE).compact_summary()
        self.assertIn("identity=a woman", summary)
        self.assertIn("top_material~smooth", summary)

    def test_a_reply_that_is_json_but_not_ir_names_the_problem(self):
        with self.assertRaises(AnalyzerError) as caught:
            analyze(backend_returning('{"sections": {"s": {"a": {"certainty": "observed"}}}}'), IMAGE)
        self.assertIn("s.a", str(caught.exception))

    def test_a_reply_with_no_attributes_is_refused(self):
        with self.assertRaises(AnalyzerError) as caught:
            analyze(backend_returning('{"sections": {}}'), IMAGE)
        self.assertIn("no attributes", str(caught.exception))

    def test_a_malformed_reply_gives_a_useful_error(self):
        with self.assertRaises(AnalyzerError) as caught:
            analyze(backend_returning("I see a woman. She looks happy."), IMAGE)
        self.assertIn("no JSON object", str(caught.exception))

    def test_the_raw_reply_is_masked(self):
        result = analyze(backend_returning(GOOD_IR + f"\n<!-- {TOKEN} -->"), IMAGE)
        self.assertNotIn(TOKEN, result.raw_reply)

    def test_the_request_summary_is_masked(self):
        self.assertNotIn(TOKEN, analyze(backend_returning(GOOD_IR), IMAGE).request_summary)

    def test_an_out_of_range_threshold_is_rejected(self):
        with self.assertRaises(AnalyzerError):
            analyze(backend_returning(GOOD_IR), IMAGE, confidence_threshold=1.5)

    def test_a_custom_system_prompt_replaces_the_shipped_one(self):
        captured = {}

        def sender(url, payload, headers, timeout):
            captured["system"] = payload["messages"][0]["content"]
            return {"choices": [{"message": {"content": GOOD_IR}}]}

        backend = OpenAICompatibleBackend(BackendConfig(model_name="m"), sender)
        analyze(backend, IMAGE, system_prompt="JUST DO IT")
        self.assertEqual(captured["system"], "JUST DO IT")


class PromptTests(unittest.TestCase):
    def test_the_extractor_specification_loads_from_disk(self):
        text = load_system_prompt()
        self.assertIn("IMAGE_IR", text)
        self.assertIn("viewer", text.lower())

    def test_it_forbids_the_things_it_must_forbid(self):
        text = load_system_prompt().lower()
        for phrase in ("not writing a prompt", "do not invent", "cinematic", "uncertain"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_it_names_every_risk_area(self):
        text = load_system_prompt().lower()
        for risk in ("viewer-left", "hosiery", "colour cast", "crossed-leg", "camera angle", "material"):
            with self.subTest(risk=risk):
                self.assertIn(risk, text)

    def test_each_detail_level_asks_for_something_different(self):
        prompts = {level: build_user_prompt(level) for level in ("fast", "balanced", "detailed")}
        self.assertEqual(len(set(prompts.values())), 3)

    def test_an_unknown_detail_level_is_rejected(self):
        with self.assertRaises(AnalyzerError):
            build_user_prompt("exhaustive")


if __name__ == "__main__":
    unittest.main()
