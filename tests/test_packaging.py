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

    def test_the_engine_needs_no_third_party_packages(self):
        # requirements.txt is deliberately empty; this is what keeps it honest.
        stdlib_only = re.compile(r"^\s*(?:from|import)\s+(\w+)", re.MULTILINE)
        third_party = {"torch", "numpy", "PIL", "aiohttp", "av", "scipy", "cv2"}
        for path in (ROOT / "imageir").glob("*.py"):
            for module in stdlib_only.findall(read_text(path)):
                with self.subTest(module=path.name):
                    self.assertNotIn(module, third_party)


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

    def test_every_node_type_exists(self):
        for path, data in self.workflows():
            for node in data["nodes"]:
                if node["type"] == "Note":
                    continue
                with self.subTest(workflow=path.name, node=node["type"]):
                    self.assertIn(node["type"], NODES)

    def test_widget_values_line_up_with_the_current_widgets(self):
        # A widget added in the middle of INPUT_TYPES silently shifts every
        # saved value after it; this is what catches that before a user hits it.
        for path, data in self.workflows():
            for node in data["nodes"]:
                if node["type"] == "Note":
                    continue
                spec = NODES[node["type"]].INPUT_TYPES()
                widgets = [
                    name
                    for group in ("required", "optional")
                    for name, definition in spec.get(group, {}).items()
                    if definition[0] != "IMAGE_IR"
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

    def test_the_h3_workflow_produces_a_clean_prompt(self):
        data = json.loads(read_text(ROOT / "example_workflows" / "image-ir-h3-prompt.json"))
        by_type = {node["type"]: node for node in data["nodes"]}
        ir = NODES["ImageIRFromText"]().build(by_type["ImageIRFromText"]["widgets_values"][0])[0]
        engine = NODES["ImageIRPromptEngine"]()
        spec = engine.INPUT_TYPES()["required"]
        widgets = [name for name in spec if spec[name][0] != "IMAGE_IR"]
        kwargs = dict(zip(widgets, by_type["ImageIRPromptEngine"]["widgets_values"], strict=True))
        prompt, *_rest = engine.run(ir, **kwargs)
        _text, report, count, clean = NODES["ImageIRGroundingGuard"]().run(ir, prompt)
        self.assertTrue(clean, report)
        self.assertEqual(count, 0)
        self.assertIn("smooth blouse", prompt)
        self.assertNotIn("satin", prompt)

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
