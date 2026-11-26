"""tests/test_probe_recall.py - probe and recall tool tests.

A probe returns marked hypothetical evidence with a delta and
creates no version. An invalid probe changes nothing. Recall
returns only model_complete sessions' pairs for the named
application. No model calls are made.

Version naming: a session delta names both ends by the store's
version identifier. A probe delta names the hypothetical end
as the next store identifier with a -probe suffix, such as
v2-probe.
"""

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2 import controller
from stage2.store import ArtifactStore
from stage2.tools import ToolRegistry, dispatch


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
            "measurements": {"value": value, "double": value * 2},
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


def _open_session(tmpdir, session_id="stub-session", application="stub"):
    store = ArtifactStore(os.path.join(tmpdir, "artifacts"))
    registry = ToolRegistry()
    registry._session_counter = 0
    registry.pending_context = {
        "store": store,
        "app": StubApp(),
        "renderer": _stub_renderer,
        "session_root": os.path.join(tmpdir, "sessions"),
        "session_id": session_id,
        "artifact_id": "stub-artifact",
        "objective": {"goal": "value matters"},
        "condition": "perceptual",
        "test": False,
        "faults": [],
        "max_iterations": 4,
        "ledger_path": None,
    }
    controller.open_session(
        registry,
        {
            "application": application,
            "objective": {"goal": "value matters"},
            "condition": "perceptual",
            "test": False,
        },
    )
    return registry, registry.get(session_id)


def _history_events(context, name):
    return [e for e in context["history"].entries if e.get("event") == name]


class ProbeTests(unittest.TestCase):
    def test_probe_returns_marked_evidence_with_delta_and_no_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry, context = _open_session(tmpdir)
            controller.submit_work(context, {"value": 1})
            before_versions = copy.deepcopy(
                context["store"].load("stub-artifact")["versions"]
            )
            before_source = copy.deepcopy(context["current_source"])
            before_iteration = context["iteration"]

            result = dispatch(
                registry,
                "sensoryne_probe",
                {
                    "session_id": context["session_id"],
                    "decision": {
                        "faults": [],
                        "revisions": [{"op": "set_value", "value": 5}],
                        "complete": False,
                        "assessment": "hypothetical",
                    },
                },
            )

            self.assertTrue(result["probe"])
            self.assertEqual(result["measurements"]["value"], 5)
            self.assertEqual(result["delta"]["from_version"], "v1")
            self.assertEqual(result["delta"]["to_version"], "v2-probe")
            by_path = {
                entry["path"]: entry
                for entry in result["delta"]["changed_measurements"]
            }
            self.assertEqual(
                by_path["measurements.value"]["absolute_delta"], 4.0
            )
            self.assertEqual(result["structure_record"]["value"], 5)
            self.assertEqual(result["hypothetical_artifact"]["value"], 5)

            record = context["store"].load("stub-artifact")
            self.assertEqual(record["versions"], before_versions)
            self.assertEqual(len(record["versions"]), 1)
            self.assertEqual(context["current_source"], before_source)
            self.assertEqual(context["current_source"]["value"], 1)
            self.assertEqual(context["iteration"], before_iteration)
            self.assertIsNone(context["ended_by"])
            self.assertEqual(_history_events(context, "session_end"), [])

    def test_invalid_probe_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry, context = _open_session(tmpdir)
            controller.submit_work(context, {"value": 2})
            before_versions = copy.deepcopy(
                context["store"].load("stub-artifact")["versions"]
            )
            before_source = copy.deepcopy(context["current_source"])

            result = dispatch(
                registry,
                "sensoryne_probe",
                {
                    "session_id": context["session_id"],
                    "decision": {
                        "faults": [],
                        "revisions": [{"op": "set_value", "value": -3}],
                        "complete": True,
                        "assessment": "invalid hypothetical",
                    },
                },
            )

            self.assertTrue(result["probe"])
            self.assertFalse(result["valid"])
            self.assertEqual(len(result["errors"]), 1)
            self.assertIn("non-negative", result["errors"][0])
            self.assertNotIn("measurements", result)
            self.assertNotIn("delta", result)

            record = context["store"].load("stub-artifact")
            self.assertEqual(record["versions"], before_versions)
            self.assertEqual(context["current_source"], before_source)
            self.assertEqual(context["current_source"]["value"], 2)
            self.assertIsNone(context["ended_by"])

    def test_probe_cannot_end_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry, context = _open_session(tmpdir)
            controller.submit_work(context, {"value": 1})
            result = dispatch(
                registry,
                "sensoryne_probe",
                {
                    "session_id": context["session_id"],
                    "decision": {
                        "faults": [],
                        "revisions": [{"op": "set_value", "value": 9}],
                        "complete": True,
                        "assessment": "hypothetical completion",
                    },
                },
            )
            self.assertTrue(result["probe"])
            self.assertNotIn("ended_by", result)
            self.assertIsNone(context["ended_by"])
            self.assertEqual(_history_events(context, "session_end"), [])
            controller.submit_decision(
                context,
                {
                    "faults": [],
                    "revisions": [],
                    "complete": True,
                    "assessment": "actually done",
                },
            )
            self.assertEqual(context["ended_by"], "model_complete")

    def test_control_probe_returns_control_shape_without_evidence_or_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(os.path.join(tmpdir, "artifacts"))
            registry = ToolRegistry()
            registry._session_counter = 0
            registry.pending_context = {
                "store": store,
                "app": StubApp(),
                "renderer": _stub_renderer,
                "session_root": os.path.join(tmpdir, "sessions"),
                "session_id": "control-session",
                "artifact_id": "stub-artifact",
                "objective": {},
                "condition": "control",
                "test": False,
                "faults": [],
                "max_iterations": 4,
                "ledger_path": None,
            }
            controller.open_session(
                registry,
                {"application": "stub", "objective": {}, "condition": "control", "test": False},
            )
            context = registry.get("control-session")
            controller.submit_work(context, {"value": 1})
            result = dispatch(
                registry,
                "sensoryne_probe",
                {
                    "session_id": "control-session",
                    "decision": {
                        "faults": [],
                        "revisions": [{"op": "set_value", "value": 4}],
                        "complete": False,
                        "assessment": "hypothetical",
                    },
                },
            )
            self.assertTrue(result["probe"])
            self.assertEqual(result["perception"], "perception_skipped")
            self.assertNotIn("evidence", result)
            self.assertNotIn("delta", result)
            self.assertNotIn("delta_stored", result)
            self.assertEqual(result["artifact"]["value"], 4)
            self.assertEqual(result["structure_record"]["value"], 4)
            self.assertEqual(context["current_source"]["value"], 1)


class RecallTests(unittest.TestCase):
    def _complete_session(self, tmpdir, session_id, application, value):
        registry, context = _open_session(
            tmpdir, session_id=session_id, application=application
        )
        controller.submit_work(context, {"value": value})
        controller.submit_decision(
            context,
            {
                "faults": [],
                "revisions": [],
                "complete": True,
                "assessment": "complete %s" % value,
            },
        )
        self.assertEqual(context["ended_by"], "model_complete")
        return registry

    def test_recall_returns_only_model_complete_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = self._complete_session(tmpdir, "session-a", "stub", 1)

            registry2, context2 = _open_session(
                tmpdir, session_id="session-b", application="stub"
            )
            controller.submit_work(context2, {"value": 99})
            controller.submit_decision(
                context2,
                {
                    "faults": [],
                    "revisions": [],
                    "complete": False,
                    "assessment": "not complete",
                },
            )
            self.assertEqual(context2["ended_by"], "model_no_revisions")

            self._complete_session(tmpdir, "session-c", "stub", 3)

            result = dispatch(
                registry,
                "sensoryne_recall",
                {"application": "stub", "limit": 10},
            )
            self.assertEqual(result["application"], "stub")
            entries = result["entries"]
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["structure_record"]["value"], 3)
            self.assertEqual(entries[1]["structure_record"]["value"], 1)
            for entry in entries:
                self.assertEqual(entry["termination"], "model_complete")
                self.assertEqual(entry["ended_by"], "model_complete")
                self.assertEqual(entry["objective"], {"goal": "value matters"})
                self.assertIn("evidence_summary", entry)
                self.assertIn("summary", entry["evidence_summary"])
            values = [entry["structure_record"]["value"] for entry in entries]
            self.assertNotIn(99, values)

    def test_recall_limit_and_unknown_application(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = self._complete_session(tmpdir, "session-a", "stub", 1)
            self._complete_session(tmpdir, "session-b", "stub", 2)
            limited = dispatch(
                registry, "sensoryne_recall", {"application": "stub", "limit": 1}
            )
            self.assertEqual(len(limited["entries"]), 1)
            self.assertEqual(
                limited["entries"][0]["structure_record"]["value"], 2
            )
            empty = dispatch(
                registry, "sensoryne_recall", {"application": "other", "limit": 5}
            )
            self.assertEqual(empty["entries"], [])

    def test_recall_survives_new_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._complete_session(tmpdir, "session-a", "stub", 7)
            fresh = ToolRegistry()
            fresh._session_counter = 0
            fresh.pending_context = {
                "store": ArtifactStore(os.path.join(tmpdir, "artifacts")),
                "app": StubApp(),
                "renderer": _stub_renderer,
                "session_root": os.path.join(tmpdir, "sessions"),
            }
            result = dispatch(
                fresh, "sensoryne_recall", {"application": "stub", "limit": 5}
            )
            self.assertEqual(len(result["entries"]), 1)
            self.assertEqual(
                result["entries"][0]["structure_record"]["value"], 7
            )


if __name__ == "__main__":
    unittest.main()
