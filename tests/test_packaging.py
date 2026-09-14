"""What ships, and whether the repository describes itself accurately.

The example workflows are executed rather than only parsed: a workflow whose
Note promises a verdict the nodes no longer give is worse than no example, and
this is the only place that mismatch would be caught.
"""

import ast
import json
import re
import unittest
from pathlib import Path

import harness
from harness import read_text

ROOT = harness.ROOT
nodes = harness.load_node_package()
NODES = nodes.NODE_CLASS_MAPPINGS


class ShippedFilesTests(unittest.TestCase):
    def test_the_files_comfyui_and_the_registry_read_are_present(self):
        for name in ("__init__.py", "image_ir_nodes.py", "pyproject.toml", "requirements.txt", "README.md", "LICENSE"):
            with self.subTest(name=name):
                self.assertTrue((ROOT / name).is_file(), name)

    def test_the_engine_is_a_package_of_its_own(self):
        for name in ("__init__", "schema", "lexicon", "motion", "compose", "guard", "dsl"):
            with self.subTest(name=name):
                self.assertTrue((ROOT / "imageir" / f"{name}.py").is_file())

    def test_the_engine_imports_nothing_from_comfyui(self):
        # The rules must be runnable, and therefore testable, without ComfyUI.
        for path in (ROOT / "imageir").glob("*.py"):
            source = read_text(path)
            with self.subTest(module=path.name):
                self.assertNotIn("import folder_paths", source)
                self.assertNotIn("import comfy", source)

    def test_importing_the_engine_needs_no_third_party_packages(self):
        # requirements.txt is deliberately empty, and this is what keeps it
        # honest. The rule is about IMPORT time: imaging.py reaches for numpy
        # and Pillow inside a function, guarded, and falls back to the standard
        # library — which is exactly why these tests can run at all here. A
        # top-level import of the same names would make the plugin unloadable
        # on a machine that lacks them, so only those are rejected.
        third_party = {"torch", "numpy", "PIL", "aiohttp", "av", "scipy", "cv2", "requests", "httpx"}
        for path in sorted((ROOT / "imageir").rglob("*.py")):
            tree = ast.parse(read_text(path), filename=str(path))
            for node in tree.body:  # top level only
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module.split(".")[0]]
                for name in names:
                    with self.subTest(module=path.name, imported=name):
                        self.assertNotIn(name, third_party)

    def test_the_whole_engine_imports_on_a_bare_interpreter(self):
        # The end this is all for: every module loads with nothing installed.
        import importlib

        for name in ("schema", "lexicon", "motion", "compose", "guard", "dsl", "h3", "analyzer", "imaging"):
            with self.subTest(module=name):
                importlib.import_module(f"imageir.{name}")
        for name in ("base", "openai_compatible", "gemini", "llama_cpp_launcher"):
            with self.subTest(module=name):
                importlib.import_module(f"imageir.backend.{name}")


class EncodingTests(unittest.TestCase):
    """Windows decides file encoding by locale, so the tests have to be explicit."""

    def test_every_shipped_text_file_is_valid_utf8(self):
        for pattern in ("*.py", "*.md", "*.toml", "*.txt", "imageir/*.py", "tests/*.py",
                        "example_workflows/*.json", ".github/workflows/*.yml"):
            for path in ROOT.glob(pattern):
                with self.subTest(path=path.name):
                    path.read_bytes().decode("utf-8")

    def test_no_test_reads_a_file_without_naming_the_encoding(self):
        # The bug that only ever shows up on a Windows runner: a bare
        # read_text() or open() decodes with the machine's locale, so a Korean
        # README or an em dash blows up there and nowhere else. Caught here
        # rather than three minutes into CI on someone else's pull request.
        #
        # Read through the AST rather than by pattern: prose about read_text()
        # — this comment included — is not a call to it, and a checker that
        # cannot tell the difference gets deleted the first time it is wrong.
        for path in sorted((ROOT / "tests").glob("*.py")):
            tree = ast.parse(read_text(path), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                named = {keyword.arg for keyword in node.keywords}
                function = node.func
                is_read_text = isinstance(function, ast.Attribute) and function.attr == "read_text"
                is_open = isinstance(function, ast.Name) and function.id == "open"
                if (is_read_text or is_open) and "encoding" not in named:
                    with self.subTest(path=path.name, line=node.lineno):
                        self.fail(
                            f"{path.name}:{node.lineno} reads a file without an encoding; "
                            f"use harness.read_text()"
                        )


class SecretLeakTests(unittest.TestCase):
    """Nothing in the repository may look like a working credential.

    The patterns match real key SHAPES and lengths — an OpenAI key is 40+
    characters, a Google API key is 39 — so the deliberately short fakes the
    tests use ("sk-test-fake") cannot match, and a pasted real one cannot not.
    """

    PATTERNS = (
        ("OpenAI-style", re.compile(r"\bsk-[A-Za-z0-9_-]{24,}")),
        ("Google API key", re.compile(r"\bAIza[A-Za-z0-9_-]{30,}")),
        ("bearer blob", re.compile(r"\bBearer\s+[A-Za-z0-9._-]{30,}")),
        ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
        ("private key block", re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")),
    )

    def test_no_file_carries_anything_shaped_like_a_credential(self):
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                content = read_text(path)
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable: nothing to read a key out of
            for label, pattern in self.PATTERNS:
                match = pattern.search(content)
                with self.subTest(path=path.name, pattern=label):
                    self.assertIsNone(match, f"{path}: possible {label} credential")

    def test_the_test_fakes_are_too_short_to_be_real(self):
        # If a fake ever grew long enough to match, the scan above would start
        # failing on our own fixtures and get weakened to compensate.
        for label, pattern in self.PATTERNS:
            with self.subTest(pattern=label):
                self.assertIsNone(pattern.search("sk-test-fake AIza-fake"))


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.pyproject = read_text(ROOT / "pyproject.toml")

    def test_the_registry_metadata_is_filled_in(self):
        for key in ("name =", "description =", "version =", "PublisherId", "DisplayName"):
            with self.subTest(key=key):
                self.assertIn(key, self.pyproject)

    def test_the_python_floor_matches_the_lint_target(self):
        self.assertIn('requires-python = ">=3.10"', self.pyproject)
        self.assertIn('target-version = "py310"', read_text(ROOT / "ruff.toml"))

    def test_the_repository_url_points_at_this_project(self):
        self.assertIn("ComfyUI-ImageIR-PromptEngine", self.pyproject)

    def test_requirements_declares_no_runtime_dependency(self):
        lines = [
            line.strip()
            for line in read_text(ROOT / "requirements.txt").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertEqual(lines, [])


class ReadmeTests(unittest.TestCase):
    def setUp(self):
        self.readme = read_text(ROOT / "README.md")

    def test_every_node_is_documented_by_its_display_name(self):
        for name in nodes.NODE_DISPLAY_NAME_MAPPINGS.values():
            with self.subTest(node=name):
                self.assertIn(name, self.readme)

    def test_all_nine_rules_are_stated(self):
        from imageir.guard import RULE_TEXT

        for number in RULE_TEXT:
            with self.subTest(rule=number):
                self.assertRegex(self.readme, rf"\b{number}\.")

    def test_both_languages_are_present(self):
        self.assertIn("한국어", self.readme)
        self.assertIn("English", self.readme)

    def test_the_example_workflows_are_mentioned(self):
        for path in sorted((ROOT / "example_workflows").glob("*.json")):
            with self.subTest(workflow=path.name):
                self.assertIn(path.name, self.readme)


class ExampleWorkflowTests(unittest.TestCase):
    def workflows(self):
        paths = sorted((ROOT / "example_workflows").glob("*.json"))
        self.assertTrue(paths, "no example workflows shipped")
        for path in paths:
            yield path, json.loads(read_text(path))

    def test_they_are_valid_workflow_documents(self):
        for path, data in self.workflows():
            with self.subTest(workflow=path.name):
                self.assertEqual(set(data) >= {"nodes", "links", "version"}, True)
                self.assertTrue(data["nodes"])

    # Node types ComfyUI itself provides. An example graph that starts from a
    # real Load Image is worth more than one that cannot be run.
    CORE_TYPES = {"Note", "LoadImage", "PreviewImage", "SaveImage"}

    def test_every_node_type_exists(self):
        for path, data in self.workflows():
            for node in data["nodes"]:
                if node["type"] in self.CORE_TYPES:
                    continue
                with self.subTest(workflow=path.name, node=node["type"]):
                    self.assertIn(node["type"], NODES)

    def test_widget_values_line_up_with_the_current_widgets(self):
        # A widget added in the middle of INPUT_TYPES silently shifts every
        # saved value after it; this is what catches that before a user hits it.
        for path, data in self.workflows():
            for node in data["nodes"]:
                if node["type"] in self.CORE_TYPES:
                    continue
                spec = NODES[node["type"]].INPUT_TYPES()
                widgets = [
                    name
                    for group in ("required", "optional")
                    for name, definition in spec.get(group, {}).items()
                    if definition[0] not in ("IMAGE_IR", "IMAGEIR_BACKEND", "IMAGE")
                ]
                with self.subTest(workflow=path.name, node=node["type"]):
                    self.assertEqual(len(node["widgets_values"]), len(widgets))

    def test_every_link_connects_declared_slots(self):
        for path, data in self.workflows():
            ids = {node["id"] for node in data["nodes"]}
            for link in data["links"]:
                _link_id, origin, _origin_slot, target, _target_slot, _type = link
                with self.subTest(workflow=path.name, link=link):
                    self.assertIn(origin, ids)
                    self.assertIn(target, ids)

    def widget_kwargs(self, node, exclude=()):
        spec = NODES[node["type"]].INPUT_TYPES()
        names = [
            name
            for group in ("required", "optional")
            for name, definition in spec.get(group, {}).items()
            if definition[0] not in ("IMAGE_IR", "IMAGEIR_BACKEND", "IMAGE")
        ]
        pairs = dict(zip(names, node["widgets_values"], strict=True))
        return {k: v for k, v in pairs.items() if k not in exclude}

    def test_the_manual_workflow_produces_a_clean_prompt(self):
        data = json.loads(read_text(ROOT / "example_workflows" / "image-ir-manual-prompt.json"))
        by_type = {node["type"]: node for node in data["nodes"]}
        ir = NODES["ImageIRFromText"]().build(by_type["ImageIRFromText"]["widgets_values"][0])[0]
        prompt, *_rest = NODES["ImageIRPromptEngine"]().run(ir, **self.widget_kwargs(by_type["ImageIRPromptEngine"]))
        _text, report, count, clean = NODES["ImageIRGroundingGuard"]().run(ir, prompt)
        self.assertTrue(clean, report)
        self.assertEqual(count, 0)
        self.assertIn("smooth blouse", prompt)
        self.assertNotIn("satin", prompt)

    def test_the_analyzer_workflow_composes_a_clean_h3_prompt(self):
        # Executed with a hand-written IR standing in for the vision model: the
        # graph's shape and its composer settings are what this pins, and those
        # are exactly what silently rot when a widget is added.
        from imageir.dsl import parse_dsl

        data = json.loads(read_text(ROOT / "example_workflows" / "image-ir-analyze-to-h3.json"))
        by_type = {node["type"]: node for node in data["nodes"]}
        self.assertIn("ImageIRAnalyzer", by_type)
        self.assertIn("ImageIRBackendConfig", by_type)

        ir = parse_dsl(
            "subject.identity = a woman\nsubject.pose = seated\nsubject.gaze = toward camera\n"
            "subject.hair = shoulder-length\ncamera.shot = medium shot"
        )
        composer = NODES["MiniMaxH3PromptComposer"]()
        prompt, *_rest = composer.run(ir, **self.widget_kwargs(by_type["MiniMaxH3PromptComposer"]))
        guard_kwargs = self.widget_kwargs(by_type["ImageIRGroundingGuard"], exclude=("prompt",))
        _text, report, count, clean = NODES["ImageIRGroundingGuard"]().run(ir, prompt, **guard_kwargs)
        self.assertTrue(clean, report)
        self.assertEqual(count, 0)
        for field in ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"):
            self.assertIn(f"{field}:", prompt)

    def test_the_analyzer_workflow_guard_is_in_h3_mode(self):
        # A plain-mode guard on an H3 document deletes the format; the example
        # must not teach that.
        data = json.loads(read_text(ROOT / "example_workflows" / "image-ir-analyze-to-h3.json"))
        guard = next(n for n in data["nodes"] if n["type"] == "ImageIRGroundingGuard")
        self.assertEqual(self.widget_kwargs(guard)["prompt_format"], "minimax_h3")

    def test_the_audit_workflow_reports_what_its_note_promises(self):
        data = json.loads(read_text(ROOT / "example_workflows" / "image-ir-audit-existing-prompt.json"))
        by_type = {node["type"]: node for node in data["nodes"]}
        note = next(node for node in data["nodes"] if node["type"] == "Note")["widgets_values"][0]
        ir = NODES["ImageIRFromText"]().build(by_type["ImageIRFromText"]["widgets_values"][0])[0]
        prompt = by_type["ImageIRGroundingGuard"]["widgets_values"][0]
        text, report, count, clean = NODES["ImageIRGroundingGuard"]().run(ir, prompt)

        self.assertFalse(clean)
        self.assertEqual(count, 6, report)
        # The Note's table claims one violation of each rule; the audit has to agree.
        self.assertEqual(sorted(re.findall(r'rule (\d)\s+"', note)), ["1", "2", "3", "4", "5", "6"])
        for number in range(1, 7):
            self.assertIn(f"rule {number} ·", report)
        # ...and the filtered prompt it prints has to be the one produced.
        self.assertIn(text, note)


if __name__ == "__main__":
    unittest.main()
