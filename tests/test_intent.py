import json
import unittest

import harness
from imageir.backend import BackendConfig, get_backend
from imageir.intent import UserIntent, author_intent, parse_intent


def intent_data(request="She waves."):
    return {
        "schema": "USER_INTENT_v1", "raw_request": request,
        "inputs": {"request": request, "camera": "", "sound": "", "music": "", "style": "", "dialogue": ""}, "duration_hint": 6.0,
        "mode_hint": "AUTO", "shot_count_hint": 1,
        "requested_actions": [{"text": "She waves.", "input": "request", "evidence": request}],
    }


def approved_review(data):
    from imageir.intent import CLAUSE_FIELDS
    return {"clauses": [{"id": f"{name}[{i}]", "supported": True}
                        for name in CLAUSE_FIELDS for i, _ in enumerate(data.get(name, []))],
            "missing_requirements": []}


def model_reply(data):
    return {"choices": [{"message": {"content": json.dumps(data)}}]}


class IntentTests(unittest.TestCase):
    def test_roundtrip(self):
        intent = parse_intent(intent_data())
        self.assertIsInstance(intent, UserIntent)
        self.assertEqual(parse_intent(intent.to_json()), intent)

    def test_invalid_evidence(self):
        data = intent_data()
        data["requested_actions"][0]["evidence"] = "a handbag"
        with self.assertRaisesRegex(ValueError, "evidence"):
            parse_intent(data)

    def test_reject_invalid_types_and_unknown_fields(self):
        for key, value in [("schema", "v2"), ("duration_hint", float("nan")),
                           ("duration_hint", 16), ("shot_count_hint", True),
                           ("mode_hint", "guess"), ("requested_actions", "wave"),
                           ("invented", []), ("raw_request", 3)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_intent({**intent_data(), key: value})

    def test_korean_authoring_uses_existing_text_backend(self):
        request = "천천히 손을 들어 인사한다."
        data = intent_data(request)
        data["requested_actions"][0]["text"] = "Slowly raises one hand in greeting."
        calls = []

        def sender(url, payload, headers, timeout):
            calls.append(payload)
            return model_reply(data if len(calls) == 1 else approved_review(data))

        intent = author_intent(get_backend(BackendConfig(), sender), request, duration=6)
        self.assertIn("raises one hand", intent.requested_actions[0].text)
        self.assertEqual(intent.raw_request, request)
        self.assertIn(request, json.dumps(calls, ensure_ascii=False))
        self.assertNotIn("image_url", json.dumps(calls))

    def test_model_cannot_replace_input_or_duration(self):
        data = intent_data("forged")
        backend = get_backend(BackendConfig(), lambda *a: {"choices": [{"message": {"content": json.dumps(data)}}]})
        with self.assertRaisesRegex(ValueError, "input"):
            author_intent(backend, "She waves.", duration=6)

    def test_authoring_error_masks_token(self):
        token = "fake-private-token"
        backend = get_backend(BackendConfig().with_token(token), lambda *a: {"choices": [{"message": {"content": token}}]})
        with self.assertRaises(ValueError) as caught:
            author_intent(backend, "wave")
        self.assertNotIn(token, str(caught.exception))

    def test_invalid_clause_shapes(self):
        for change in [{"text": 42}, {"text": ""}, {"shot": 0}, {"shot": 3},
                       {"text": "<Picture 8>"}, {"input": "unknown"}, {"path": None}]:
            data = intent_data()
            data["requested_actions"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_intent(data)
        for change in [{"inputs": []}, {"requested_actions": ["wave"]}, {"raw_request": "different"}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_intent({**intent_data(), **change})

    def test_boundary_requires_path(self):
        data = intent_data()
        data["start_states"] = data["requested_actions"]
        with self.assertRaisesRegex(ValueError, "path"):
            parse_intent(data)

    def test_nonempty_optional_input_cannot_disappear(self):
        data = intent_data("")
        data["requested_actions"] = []
        data["inputs"]["camera"] = "The camera pans left."
        backend = get_backend(BackendConfig(), lambda *a: {"choices": [{"message": {"content": json.dumps(data)}}]})
        with self.assertRaisesRegex(ValueError, "coverage|no structured"):
            author_intent(backend, "", camera="The camera pans left.")

    def test_korean_full_author_to_composer(self):
        from imageir.dsl import parse_dsl
        from imageir.h3_modes import compose_mode
        from imageir.provenance import guard_document
        request = "여성이 카운터에서 일어나 앞으로 걸어오며 머리카락을 귀 뒤로 넘긴다. 카메라는 천천히 뒤로 이동한다."
        data = intent_data(request)
        data["requested_actions"] = [
            {"text": "She stands from the counter and walks forward, then tucks her hair behind her ear.",
             "input": "request", "evidence": "여성이 카운터에서 일어나 앞으로 걸어오며 머리카락을 귀 뒤로 넘긴다."}]
        data["requested_camera"] = [{"text": "The camera slowly pulls back.", "input": "request",
                                     "evidence": "카메라는 천천히 뒤로 이동한다."}]
        replies = iter((model_reply(data), model_reply(approved_review(data))))
        backend = get_backend(BackendConfig(), lambda *a: next(replies))
        intent = author_intent(backend, request)
        ir = parse_dsl("subject.identity = a woman\nsubject.pose = leaning against a counter\nsubject.gaze = toward camera\nwardrobe.top = pale blouse")
        result = compose_mode(intent, first=ir)
        self.assertTrue(guard_document(result).clean)
        for text in ("walks forward", "tucks her hair", "pulls back", "pale blouse"):
            self.assertIn(text, result.render())
        self.assertNotIn("handbag", result.render())

    def test_semantic_review_rejects_invented_object_despite_valid_quote(self):
        data = intent_data("손을 흔든다.")
        data["requested_actions"][0]["text"] = "She waves a handbag."
        review = approved_review(data)
        review["clauses"][0]["supported"] = False
        replies = iter((model_reply(data), model_reply(review)))
        backend = get_backend(BackendConfig(), lambda *a: next(replies))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            author_intent(backend, "손을 흔든다.")

    def test_semantic_review_requires_complete_typed_verdicts(self):
        data = intent_data()
        for review in [{"clauses": [], "missing_requirements": []},
                       {"clauses": [{"id": "requested_actions[0]", "supported": "true"}], "missing_requirements": []},
                       {**approved_review(data), "missing_requirements": ["camera movement"]},
                       {"clauses": [{"id": "unknown", "supported": True}], "missing_requirements": []}]:
            replies = iter((model_reply(data), model_reply(review)))
            backend = get_backend(BackendConfig(), lambda *a, replies=replies: next(replies))
            with self.subTest(review=review), self.assertRaisesRegex(ValueError, "review"):
                author_intent(backend, "She waves.")


if __name__ == "__main__":
    unittest.main()
