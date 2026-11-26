"""tests/test_delta.py - delta evidence tests.

Exercises the pure comparison on known document pairs and a
controller session driven by a stub application. No model
calls are made.

Version naming: versions are identified everywhere by the
store's version identifier (v1, v2, ...). A delta's
from_version and to_version both use that identifier.
"""

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2 import controller
from stage2.delta import compute_delta
from stage2.store import ArtifactStore
from stage2.tools import ToolRegistry


class ComputeDeltaTests(unittest.TestCase):
    def test_known_document_pair(self):
        before = {
            "a": 1,
            "b": "x",
            "c": [1, 2],
            "d": {"x": True},
            "gone": 3,
        }
        after = {
            "a": 2,
            "b": "x",
            "c": [1, 3, 4],
            "d": {"x": False},
            "new": 9,
        }
        delta = compute_delta(
            "v1",
            "v2",
            before,
            after,
            {"parts": [{"notes": [1, 2]}]},
            {"parts": [{"notes": [1, 3, 4]}]},
        )
        self.assertEqual(delta["from_version"], "v1")
        self.assertEqual(delta["to_version"], "v2")

        paths = [entry["path"] for entry in delta["changed_measurements"]]
        self.assertEqual(paths, sorted(paths))
        self.assertIn("a", paths)
        self.assertIn("c.1", paths)
        self.assertIn("d.x", paths)
        self.assertNotIn("c.2", paths)

        by_path = {entry["path"]: entry for entry in delta["changed_measurements"]}
        self.assertEqual(by_path["a"]["before"], 1)
        self.assertEqual(by_path["a"]["after"], 2)
        self.assertEqual(by_path["a"]["absolute_delta"], 1.0)
        self.assertEqual(by_path["a"]["relative_delta"], 1.0)
        self.assertEqual(by_path["c.1"]["absolute_delta"], 1.0)
        self.assertEqual(by_path["d.x"]["before"], True)
        self.assertEqual(by_path["d.x"]["after"], False)
        self.assertNotIn("absolute_delta", by_path["d.x"])
        self.assertNotIn("relative_delta", by_path["d.x"])

        self.assertEqual(delta["added_measurements"], ["c.2", "new"])
        self.assertEqual(delta["removed_measurements"], ["gone"])
        self.assertEqual(delta["unchanged_keys"], ["b"])
        self.assertEqual(
            delta["structural_diff"],
            {
                "added": ["parts.0.notes.2"],
                "removed": [],
                "modified": ["parts.0.notes.1"],
            },
        )

    def test_non_numeric_and_zero_before_changes(self):
        delta = compute_delta(
            "v1",
            "v2",
            {"count": 0, "label": "before", "flag": False},
            {"count": 3, "label": "after", "flag": True},
            {},
            {},
        )
        by_path = {entry["path"]: entry for entry in delta["changed_measurements"]}
        self.assertEqual(by_path["count"]["absolute_delta"], 3.0)
        self.assertNotIn("relative_delta", by_path["count"])
        self.assertEqual(by_path["label"]["before"], "before")
        self.assertEqual(by_path["label"]["after"], "after")
        self.assertNotIn("absolute_delta", by_path["label"])
        self.assertEqual(by_path["flag"]["before"], False)
        self.assertEqual(by_path["flag"]["after"], True)
        self.assertNotIn("absolute_delta", by_path["flag"])

    def test_determinism(self):
        before = {"z": 1, "a": [3, 2]}
        after = {"z": 2, "a": [3, 4, 5], "m": "new"}
        first = compute_delta("v1", "v2", before, after, before, after)
        second = compute_delta("v1", "v2", before, after, before, after)
        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )
        self.assertEqual(
            [entry["path"] for entry in first["changed_measurements"]],
            sorted(entry["path"] for entry in first["changed_measurements"]),
        )


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
        artifact = render_record.get("_artifact") or {}
        value = artifact.get("value", 0)
        return {
            "summary": "value %s" % value,
            "measurements": {
                "value": value,
                "label": "stub",
                "double": value * 2,
            },
            "full": {},
        }

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


def _open_session(tmpdir):
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
        "condition": "perceptual",
        "test": False,
        "faults": [],
        "max_iterations": 4,
        "ledger_path": None,
    }
    controller.open_session(
        registry,
        {
            "application": "stub",
            "objective": {},
            "condition": "perceptual",
            "test": False,
        },
    )
    context = registry.get("stub-session")
    result = controller.submit_work(context, {"value": 1})
    return registry, context, result


class ControllerDeltaTests(unittest.TestCase):
    def test_first_version_has_no_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context, result = _open_session(tmpdir)
            self.assertNotIn("delta", result["evidence"])
            self.assertEqual(context["current_version_id"], "v1")
            record = context["store"].load("stub-artifact")
            self.assertEqual(len(record["versions"]), 1)
            self.assertNotIn("delta", record["versions"][0]["render_info"])

    def test_revision_evidence_carries_correct_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context, _first = _open_session(tmpdir)
            result = controller.submit_decision(
                context,
                {
                    "faults": [],
                    "revisions": [{"op": "set_value", "value": 2}],
                    "complete": False,
                    "assessment": "continue",
                },
            )
            self.assertNotIn("ended_by", result)
            delta = result["evidence"]["delta"]
            self.assertEqual(delta["from_version"], "v1")
            self.assertEqual(delta["to_version"], "v2")
            by_path = {
                entry["path"]: entry
                for entry in delta["changed_measurements"]
            }
            self.assertEqual(by_path["measurements.value"]["before"], 1)
            self.assertEqual(by_path["measurements.value"]["after"], 2)
            self.assertEqual(by_path["measurements.value"]["absolute_delta"], 1.0)
            self.assertEqual(by_path["measurements.value"]["relative_delta"], 1.0)
            self.assertEqual(by_path["measurements.double"]["absolute_delta"], 2.0)
            self.assertIn("measurements.label", delta["unchanged_keys"])
            self.assertEqual(delta["added_measurements"], [])
            self.assertEqual(delta["removed_measurements"], [])
            self.assertEqual(delta["structural_diff"]["modified"], ["value"])
            self.assertEqual(delta["structural_diff"]["added"], [])
            self.assertEqual(delta["structural_diff"]["removed"], [])

            record = context["store"].load("stub-artifact")
            self.assertEqual(len(record["versions"]), 2)
            self.assertEqual(record["versions"][1]["render_info"]["delta"], delta)

    def test_rejected_revision_produces_no_version_and_no_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context, first = _open_session(tmpdir)
            result = controller.submit_decision(
                context,
                {
                    "faults": [],
                    "revisions": [{"op": "set_value", "value": -1}],
                    "complete": False,
                    "assessment": "invalid revision",
                },
            )
            self.assertEqual(result["applied"], [])
            self.assertEqual(len(result["rejected"]), 1)
            self.assertIn("non-negative", result["rejected"][0]["error"])
            self.assertEqual(context["current_version_id"], "v1")
            self.assertEqual(context["iteration"], 1)
            record = context["store"].load("stub-artifact")
            self.assertEqual(len(record["versions"]), 1)
            self.assertNotIn("delta", record["versions"][0]["render_info"])
            self.assertNotIn("delta", result["evidence"])
            self.assertEqual(result["evidence"], first["evidence"])

    def test_noop_revision_produces_no_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context, _first = _open_session(tmpdir)
            result = controller.submit_decision(
                context,
                {
                    "faults": [],
                    "revisions": [{"op": "set_value", "value": 1}],
                    "complete": False,
                    "assessment": "no change",
                },
            )
            self.assertEqual(result["applied"], [])
            self.assertEqual(len(result["rejected"]), 1)
            self.assertIn("no-op", result["rejected"][0]["error"])
            record = context["store"].load("stub-artifact")
            self.assertEqual(len(record["versions"]), 1)


if __name__ == "__main__":
    unittest.main()
