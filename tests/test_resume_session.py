"""tests/test_resume_session.py - session resume and shape-field delivery.

Covers the tenth tool, sensoryne_resume_session: a session whose
process died (no session_end) and a session that ended by an error
(decision_unparseable) both continue in a fresh registry over the
same on-disk record, producing the next version in the same
session record; a session that ended as model_complete and an
unknown id are refused with no change to the record; resuming an
already-live session returns its state undisturbed (no duplicate
registration and no session_resumed event logged for it).

Also covers the revision-vocabulary delivery: submit_work,
submit_decision and probe results carry decision_shape_text and,
when the application defines one, revision_shape_text as their
own top-level fields, and both texts survive shaping whole in an
over-budget result.
"""

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2 import controller
from stage2.decision import DECISION_SHAPE_TEXT
from stage2.shape import EVIDENCE_BUDGET, _size, shape_document
from stage2.store import ArtifactStore
from stage2.tools import TOOL_NAMES, ToolRegistry, dispatch


class StubApp:
    NAME = "stub"
    REVISION_SHAPE_TEXT = "Revisions: {\"op\": \"set_value\", \"value\": 7}."

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


def _fresh_registry(tmpdir):
    registry = ToolRegistry()
    registry._session_counter = 0
    registry.pending_context = {
        "store": ArtifactStore(os.path.join(tmpdir, "artifacts")),
        "app": StubApp(),
        "renderer": _stub_renderer,
        "session_root": os.path.join(tmpdir, "sessions"),
        "objective": {"goal": "value matters"},
        "faults": [],
        "max_iterations": 4,
        "ledger_path": None,
    }
    return registry


def _open(registry, condition="perceptual"):
    return dispatch(
        registry,
        "sensoryne_open_session",
        {"application": "stub", "objective": {"goal": "value matters"}, "condition": condition, "test": False},
    )


def _history_entries(ctx_or_path):
    if isinstance(ctx_or_path, str):
        path = ctx_or_path if os.path.basename(ctx_or_path) == "history.jsonl" else os.path.join(ctx_or_path, "history.jsonl")
    else:
        path = os.path.join(ctx_or_path["session_dir"], "history.jsonl")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _history_bytes(ctx):
    with open(os.path.join(ctx["session_dir"], "history.jsonl"), "rb") as handle:
        return handle.read()


class ResumeProcessDeathTests(unittest.TestCase):
    def test_process_death_resume_continues_same_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg1 = _fresh_registry(tmpdir)
            opened = _open(reg1)
            sid = opened["session_id"]
            ctx1 = reg1.get(sid)
            controller.submit_work(ctx1, {"value": 1})
            self.assertEqual(ctx1["current_version_id"], "v1")

            # The process dies: the live registry is dropped and
            # only the on-disk record remains.
            del reg1
            reg2 = _fresh_registry(tmpdir)
            result = dispatch(reg2, "sensoryne_resume_session", {"session_id": sid})

            self.assertEqual(result["session_id"], sid)
            self.assertEqual(result["application"], "stub")
            self.assertEqual(result["version_resumed_at"], "v1")
            self.assertIn("Session so far:", result["session_so_far"])
            self.assertEqual(result["decision_shape_text"], DECISION_SHAPE_TEXT)
            self.assertEqual(result["revision_shape_text"], StubApp.REVISION_SHAPE_TEXT)

            entries = _history_entries(os.path.join(tmpdir, "sessions", sid, "history.jsonl"))
            resumed = [e for e in entries if e.get("event") == "session_resumed"]
            self.assertEqual(len(resumed), 1)
            self.assertEqual(resumed[0]["version_resumed_at"], "v1")
            self.assertIsNone(resumed[0]["prior_ended_by"])

            ctx2 = reg2.get(sid)
            self.assertEqual(ctx2["current_source"], {"value": 1})
            self.assertEqual(ctx2["iteration"], 1)
            self.assertTrue(ctx2["_work_submitted"])
            self.assertIsNone(ctx2["ended_by"])

            applied = controller.submit_decision(
                ctx2,
                {"faults": [], "revisions": [{"op": "set_value", "value": 5}], "complete": False, "assessment": "raised"},
            )
            self.assertEqual(applied["applied"], [{"op": "set_value", "value": 5}])
            self.assertEqual(ctx2["current_source"], {"value": 5})
            self.assertEqual(ctx2["current_version_id"], "v2")
            record = ctx2["store"].load(ctx2["artifact_id"])
            self.assertEqual([v["version_id"] for v in record["versions"]], ["v1", "v2"])
            entries_after = _history_entries(ctx2["session_dir"])
            renders = [e for e in entries_after if e.get("event") == "render"]
            self.assertEqual([e["version_id"] for e in renders], ["v1", "v2"])
            self.assertEqual(ctx2["session_dir"], os.path.join(tmpdir, "sessions", sid))
            # The decision result carries the shape texts too.
            self.assertEqual(applied["decision_shape_text"], DECISION_SHAPE_TEXT)
            self.assertEqual(applied["revision_shape_text"], StubApp.REVISION_SHAPE_TEXT)

    def test_error_ending_resume_continues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg1 = _fresh_registry(tmpdir)
            sid = _open(reg1)["session_id"]
            ctx1 = reg1.get(sid)
            controller.submit_work(ctx1, {"value": 1})
            bad = controller.submit_decision(ctx1, "this is not a decision at all {{{")
            self.assertEqual(bad.get("ended_by"), "decision_unparseable")
            self.assertEqual(ctx1["ended_by"], "decision_unparseable")

            del reg1
            reg2 = _fresh_registry(tmpdir)
            result = dispatch(reg2, "sensoryne_resume_session", {"session_id": sid})
            self.assertEqual(result["prior_ended_by"], "decision_unparseable")
            self.assertEqual(result["version_resumed_at"], "v1")

            entries = _history_entries(os.path.join(tmpdir, "sessions", sid, "history.jsonl"))
            resumed = [e for e in entries if e.get("event") == "session_resumed"]
            self.assertEqual(len(resumed), 1)
            self.assertEqual(resumed[0]["prior_ended_by"], "decision_unparseable")

            ctx2 = reg2.get(sid)
            self.assertIsNone(ctx2["ended_by"])
            decision_result = controller.submit_decision(
                ctx2,
                {"faults": [], "revisions": [{"op": "set_value", "value": 9}], "complete": False, "assessment": "after resume"},
            )
            self.assertEqual(decision_result["applied"], [{"op": "set_value", "value": 9}])
            self.assertEqual(ctx2["current_version_id"], "v2")
            record = ctx2["store"].load(ctx2["artifact_id"])
            self.assertEqual([v["version_id"] for v in record["versions"]], ["v1", "v2"])


class ResumeRefusalTests(unittest.TestCase):
    def test_model_complete_refused_and_record_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg1 = _fresh_registry(tmpdir)
            sid = _open(reg1)["session_id"]
            ctx1 = reg1.get(sid)
            controller.submit_work(ctx1, {"value": 1})
            controller.submit_decision(
                ctx1,
                {"faults": [], "revisions": [], "complete": True, "assessment": "done"},
            )
            self.assertEqual(ctx1["ended_by"], "model_complete")
            before = _history_bytes(ctx1)
            versions_before = copy.deepcopy(ctx1["store"].load(ctx1["artifact_id"])["versions"])

            del reg1
            reg2 = _fresh_registry(tmpdir)
            with self.assertRaises(ValueError) as cm:
                dispatch(reg2, "sensoryne_resume_session", {"session_id": sid})
            self.assertIn("model_complete", str(cm.exception))
            self.assertEqual(_history_bytes(ctx1), before)
            self.assertEqual(ctx1["store"].load(ctx1["artifact_id"])["versions"], versions_before)

    def test_live_model_complete_refused_in_same_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = _fresh_registry(tmpdir)
            sid = _open(reg)["session_id"]
            ctx = reg.get(sid)
            controller.submit_work(ctx, {"value": 1})
            controller.submit_decision(
                ctx,
                {"faults": [], "revisions": [], "complete": True, "assessment": "done"},
            )
            before = _history_bytes(ctx)
            with self.assertRaises(ValueError) as cm:
                dispatch(reg, "sensoryne_resume_session", {"session_id": sid})
            self.assertIn("model_complete", str(cm.exception))
            self.assertEqual(_history_bytes(ctx), before)

    def test_unknown_session_errors_cleanly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = _fresh_registry(tmpdir)
            with self.assertRaises(ValueError) as cm:
                dispatch(reg, "sensoryne_resume_session", {"session_id": "nope-perceptual-1"})
            self.assertIn("unknown session_id", str(cm.exception))
            sessions_root = os.path.join(tmpdir, "sessions")
            leftover = os.listdir(sessions_root) if os.path.isdir(sessions_root) else []
            self.assertEqual(leftover, [])


class ResumeLiveTests(unittest.TestCase):
    def test_live_resume_returns_state_undisturbed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = _fresh_registry(tmpdir)
            sid = _open(reg)["session_id"]
            ctx = reg.get(sid)
            controller.submit_work(ctx, {"value": 1})
            result = dispatch(reg, "sensoryne_resume_session", {"session_id": sid})
            self.assertEqual(result["session_id"], sid)
            self.assertEqual(result["version_resumed_at"], "v1")
            # Same live object: no duplicate registration, and
            # (this run's chosen behaviour) no session_resumed
            # event is logged for an undisturbed live session.
            self.assertIs(reg.get(sid), ctx)
            entries = _history_entries(ctx["session_dir"])
            self.assertEqual([e for e in entries if e.get("event") == "session_resumed"], [])
            decision_result = controller.submit_decision(
                ctx,
                {"faults": [], "revisions": [{"op": "set_value", "value": 3}], "complete": False, "assessment": "continued"},
            )
            self.assertEqual(decision_result["applied"], [{"op": "set_value", "value": 3}])
            self.assertEqual(ctx["current_version_id"], "v2")


class ShapeTextDeliveryTests(unittest.TestCase):
    def test_tool_registered_and_counts_corrected(self):
        self.assertIn("sensoryne_resume_session", TOOL_NAMES)
        self.assertEqual(len(TOOL_NAMES), 10)
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp_server.py"), encoding="utf-8") as handle:
            mcp_text = handle.read()
        self.assertIn("ten tools", mcp_text)
        self.assertIn("sensoryne_resume_session", mcp_text)

    def test_controller_results_carry_shape_texts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = _fresh_registry(tmpdir)
            sid = _open(reg)["session_id"]
            ctx = reg.get(sid)
            work = controller.submit_work(ctx, {"value": 1})
            self.assertEqual(work["decision_shape_text"], DECISION_SHAPE_TEXT)
            self.assertEqual(work["revision_shape_text"], StubApp.REVISION_SHAPE_TEXT)
            probe_result = dispatch(
                reg,
                "sensoryne_probe",
                {"session_id": sid, "decision": {"faults": [], "revisions": [{"op": "set_value", "value": 4}], "complete": False, "assessment": "hypo"}},
            )
            self.assertEqual(probe_result["decision_shape_text"], DECISION_SHAPE_TEXT)
            self.assertEqual(probe_result["revision_shape_text"], StubApp.REVISION_SHAPE_TEXT)

    def test_shape_texts_arrive_whole_when_over_budget(self):
        big_evidence = {
            "summary": "Heard a tonal sound.",
            "measurements": {"value": 3, "pitch_hz": 440.0, "values": list(range(30000)), "waveform": list(range(20000))},
            "full": {"blob": list(range(30000))},
        }
        revision_text = StubApp.REVISION_SHAPE_TEXT
        top_level = {
            "evidence": copy.deepcopy(big_evidence),
            "structure_record": {"value": 1, "kind": "stub"},
            "decision_shape_text": DECISION_SHAPE_TEXT,
            "revision_shape_text": revision_text,
            "presentation": "x" * 80000,
        }
        self.assertGreater(_size(top_level), EVIDENCE_BUDGET)
        shaped = shape_document(copy.deepcopy(top_level))
        self.assertEqual(shaped["decision_shape_text"], DECISION_SHAPE_TEXT)
        self.assertEqual(shaped["revision_shape_text"], revision_text)

        nested = {
            "evidence": dict(
                copy.deepcopy(big_evidence),
                decision_shape_text=DECISION_SHAPE_TEXT,
                revision_shape_text=revision_text,
            )
        }
        self.assertGreater(_size(nested), EVIDENCE_BUDGET)
        shaped_nested = shape_document(copy.deepcopy(nested))
        self.assertEqual(shaped_nested["evidence"]["decision_shape_text"], DECISION_SHAPE_TEXT)
        self.assertEqual(shaped_nested["evidence"]["revision_shape_text"], revision_text)


if __name__ == "__main__":
    unittest.main()
