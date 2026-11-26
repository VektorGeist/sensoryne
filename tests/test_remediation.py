"""tests/test_remediation.py - in-scope remediation tests.

Covers the code renderer path tests and the controller tests
for completion with a revision, structure currency after a
revision, and no-revision with complete set false.
"""

import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import code_renderer
from stage2 import controller
from stage2.apps.code import app as code_app
from stage2.store import ArtifactStore
from stage2.tools import ToolRegistry


def _code_artifact(name, text="print('hi')\n"):
    return {
        "files": [{"name": name, "text": text}],
        "purpose": "print hi",
        "tests": [{"name": "t", "input": "", "expected": "hi\n"}],
    }


class CodeRendererPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.render_root = self.root / "render"
        self.render_root.mkdir()
        self.outside = self.root / "outside"
        self.outside.mkdir()

    def _render(self, name):
        return code_renderer.renderer(_code_artifact(name), str(self.render_root), [])

    def _outside_files(self):
        return [p for p in self.outside.rglob("*") if p.is_file()]

    def test_normal_filename(self):
        record = self._render("main.py")
        self.assertEqual(record["files"], ["main.py"])
        self.assertTrue((self.render_root / "main.py").is_file())
        self.assertEqual(self._outside_files(), [])

    def test_nested_permitted_filename(self):
        record = self._render("sub/dir/main.py")
        self.assertEqual(record["files"], ["sub/dir/main.py"])
        self.assertTrue((self.render_root / "sub" / "dir" / "main.py").is_file())
        self.assertEqual(self._outside_files(), [])

    def test_absolute_path_rejected(self):
        target = self.outside / "evil.py"
        with self.assertRaises(ValueError):
            self._render(str(target))
        self.assertFalse(target.exists())
        self.assertEqual(self._outside_files(), [])

    def test_parent_traversal_rejected(self):
        for name in ("../evil.py", "sub/../../evil.py", "..\\evil.py"):
            with self.assertRaises(ValueError, msg=name):
                self._render(name)
        self.assertEqual(self._outside_files(), [])
        self.assertFalse((self.root / "evil.py").exists())

    def test_windows_drive_path_rejected(self):
        for name in ("C:\\evil.py", "C:/evil.py", "D:evil.py"):
            with self.assertRaises(ValueError, msg=name):
                self._render(name)
        self.assertEqual(self._outside_files(), [])

    def test_unc_path_rejected(self):
        for name in ("\\\\server\\share\\evil.py", "//server/share/evil.py"):
            with self.assertRaises(ValueError, msg=name):
                self._render(name)
        self.assertEqual(self._outside_files(), [])

    def test_mixed_separator_traversal_rejected(self):
        for name in ("a/..\\evil.py", "a\\../evil.py", "sub/..\\..\\evil.py"):
            with self.assertRaises(ValueError, msg=name):
                self._render(name)
        self.assertEqual(self._outside_files(), [])

    def test_symlink_directory_escape_rejected(self):
        link = self.render_root / "link"
        link.symlink_to(self.outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self._render("link/evil.py")
        self.assertFalse((self.outside / "evil.py").exists())
        self.assertEqual(self._outside_files(), [])

    def test_symlink_file_escape_rejected(self):
        target = self.outside / "evil.py"
        target.write_text("outside\n", encoding="utf-8")
        link = self.render_root / "evil.py"
        link.symlink_to(target)
        with self.assertRaises(ValueError):
            self._render("evil.py")
        self.assertEqual(target.read_text(encoding="utf-8"), "outside\n")

    def test_revision_generated_unsafe_filename_rejected(self):
        artifact = _code_artifact("main.py")
        before = copy.deepcopy(artifact)
        for name in ("../evil.py", "/tmp/evil.py", "C:\\evil.py", "a/..\\evil.py"):
            with self.assertRaises(ValueError, msg=name):
                code_app.apply_revision(
                    artifact, {"op": "add_file", "name": name, "text": "x = 1\n"}
                )
            self.assertEqual(artifact, before)
        self.assertEqual(self._outside_files(), [])

    def test_no_write_outside_render_root(self):
        attempts = [
            "../evil.py",
            str(self.outside / "evil.py"),
            "C:\\evil.py",
            "\\\\server\\share\\evil.py",
            "a/..\\evil.py",
        ]
        for name in attempts:
            with self.assertRaises(ValueError, msg=name):
                self._render(name)
        written_outside = [
            p for p in self.root.rglob("*")
            if p.is_file() and self.render_root not in p.parents and p.parent != self.render_root
        ]
        self.assertEqual(written_outside, [])


class StubApp:
    NAME = "stub"

    def validate(self, artifact):
        if not isinstance(artifact, dict) or "value" not in artifact:
            return ["artifact must hold a value"]
        value = artifact["value"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return ["value must be a non-negative integer"]
        return []

    def composition_prompt(self, objective):
        return "compose a stub artifact"

    def composition_retry_prompt(self, objective, errors, rejected):
        return "retry: %s" % "; ".join(errors)

    def control_prompt(self, objective, artifact):
        return "control prompt"

    def perceptual_prompt(self, objective, evidence, input_summary, structure_record):
        return "perceptual prompt"

    def input_summary(self, artifact):
        return copy.deepcopy(artifact)

    def structure_record(self, artifact):
        return {"value": artifact.get("value"), "kind": "stub"}

    def generate_ground_truth(self, faults):
        return []

    def plant_faults(self, artifact):
        return []

    def perceive(self, render_record):
        return {"summary": "stub evidence", "measurements": {}, "full": {}}

    def apply_revision(self, artifact, revision):
        if not isinstance(revision, dict) or revision.get("op") != "set_value":
            raise ValueError("unknown revision op: %r" % (revision.get("op") if isinstance(revision, dict) else revision,))
        value = revision.get("value")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("value must be a non-negative integer")
        revised = copy.deepcopy(artifact)
        revised["value"] = value
        errors = self.validate(revised)
        if errors:
            raise ValueError("; ".join(errors))
        return revised

    def apply_revisions(self, artifact, revisions):
        current = copy.deepcopy(artifact)
        for revision in revisions or []:
            current = self.apply_revision(current, revision)
        return current


def _stub_renderer(artifact, out_path, faults):
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("value=%s\n" % artifact.get("value"))
    return {"path": str(out_path), "kind": "text"}


def _open_stub_session(tmpdir, condition="perceptual"):
    store = ArtifactStore(os.path.join(tmpdir, "artifacts"))
    registry = ToolRegistry()
    registry._session_counter = 0
    registry.pending_context = {
        "store": store,
        "app": StubApp(),
        "renderer": _stub_renderer,
        "session_root": os.path.join(tmpdir, "sessions"),
        "session_id": "stub-session",
        "artifact_id": "stub-artifact",
        "objective": {},
        "condition": condition,
        "test": False,
        "faults": [],
        "max_iterations": 4,
        "ledger_path": None,
    }
    controller.open_session(
        registry,
        {"application": "stub", "objective": {}, "condition": condition, "test": False},
    )
    context = registry.get("stub-session")
    controller.submit_work(context, {"value": 1})
    return registry, context


class ControllerTerminationTests(unittest.TestCase):
    def test_completion_with_revision_is_rendered_before_ending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context = _open_stub_session(tmpdir)
            self.assertEqual(context["iteration"], 1)
            result = controller.submit_decision(
                context,
                {
                    "faults": [],
                    "revisions": [{"op": "set_value", "value": 42}],
                    "complete": True,
                    "assessment": "done after revision",
                },
            )
            self.assertEqual(result["ended_by"], "model_complete")
            self.assertEqual(context["ended_by"], "model_complete")
            self.assertEqual(context["iteration"], 2)
            versions = context["store"].load("stub-artifact")["versions"]
            self.assertEqual(len(versions), 2)
            latest = context["store"].load_version_source("stub-artifact", versions[-1]["version_id"])
            self.assertEqual(latest["value"], 42)
            self.assertEqual(context["current_source"]["value"], 42)

    def test_structure_record_is_current_after_revision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context = _open_stub_session(tmpdir)
            result = controller.submit_decision(
                context,
                {
                    "faults": [],
                    "revisions": [{"op": "set_value", "value": 7}],
                    "complete": False,
                    "assessment": "needs another look",
                },
            )
            self.assertEqual(result["structure_record"]["value"], 7)
            self.assertNotEqual(result["structure_record"]["value"], 1)
            perception = [
                e for e in context["history"].entries if e.get("event") == "perception"
            ]
            self.assertEqual(perception[-1]["structure_record"]["value"], 7)
            self.assertIsNone(context["ended_by"])

    def test_no_revision_with_complete_false_is_not_successful_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context = _open_stub_session(tmpdir)
            result = controller.submit_decision(
                context,
                {
                    "faults": [],
                    "revisions": [],
                    "complete": False,
                    "assessment": "not yet complete",
                },
            )
            self.assertEqual(result["ended_by"], "model_no_revisions")
            self.assertEqual(context["ended_by"], "model_no_revisions")
            self.assertNotEqual(context["ended_by"], "model_complete")
            self.assertEqual(context["iteration"], 1)


if __name__ == "__main__":
    unittest.main()
