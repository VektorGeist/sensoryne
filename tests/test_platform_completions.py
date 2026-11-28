"""tests/test_platform_completions.py - three platform completions.

submit_work accepts a JSON-text artifact the same as an object
artifact; tool descriptions state the form of their arguments;
judge verdicts recorded on a session reach the working model
through build_session_so_far (the resume path's summary).
No model calls are made.
"""

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2 import controller, judge
from stage2.decision import build_session_so_far
from stage2.store import ArtifactStore
from stage2.tools import TOOLS, ToolRegistry, dispatch


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
            raise ValueError("unknown revision op")
        value = revision.get("value")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("value must be a non-negative integer")
        revised = copy.deepcopy(artifact)
        revised["value"] = value
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


def _fresh_registry(tmpdir, session_id=None):
    registry = ToolRegistry()
    registry._session_counter = 0
    pending = {
        "store": ArtifactStore(os.path.join(tmpdir, "artifacts")),
        "app": StubApp(),
        "renderer": _stub_renderer,
        "session_root": os.path.join(tmpdir, "sessions"),
        "objective": {"goal": "value matters"},
        "faults": [],
        "max_iterations": 4,
        "ledger_path": None,
    }
    if session_id:
        pending["session_id"] = session_id
        pending["artifact_id"] = session_id
    registry.pending_context = pending
    return registry


def _open(registry, condition="perceptual"):
    return dispatch(
        registry,
        "sensoryne_open_session",
        {"application": "stub", "objective": {"goal": "value matters"}, "condition": condition, "test": False},
    )


def _history_from_disk(tmpdir, sid):
    path = os.path.join(tmpdir, "sessions", sid, "history.jsonl")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class SubmitWorkStringFormTests(unittest.TestCase):
    def test_submit_work_as_json_text_stores_work_and_produces_perception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = _fresh_registry(tmpdir, session_id="stub-session")
            sid = _open(reg)["session_id"]
            ctx = reg.get(sid)
            artifact = {"value": 3}
            result = controller.submit_work(ctx, json.dumps(artifact))

            self.assertTrue(result["valid"])
            self.assertIn("evidence", result)
            self.assertEqual(result["evidence"]["measurements"]["value"], 3)
            self.assertEqual(ctx["current_source"], artifact)
            self.assertTrue(ctx["_work_submitted"])
            record = ctx["store"].load(ctx["artifact_id"])
            self.assertEqual(len(record["versions"]), 1)
            stored = ctx["store"].load_version_source(ctx["artifact_id"], "v1")
            self.assertEqual(stored, artifact)
            self.assertIsInstance(stored, dict)
            # The session record stores the parsed object, never the string.
            tool_calls = [e for e in ctx["history"].entries if e.get("event") == "tool_call" and e.get("tool") == "sensoryne_submit_work"]
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(tool_calls[0]["artifact"], artifact)
            self.assertIsInstance(tool_calls[0]["artifact"], dict)
            perceptions = [e for e in ctx["history"].entries if e.get("event") == "perception"]
            self.assertEqual(len(perceptions), 1)
            self.assertEqual(perceptions[0]["evidence"]["measurements"]["value"], 3)

    def test_submit_work_as_json_text_matches_object_form(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_obj = _fresh_registry(tmpdir, session_id="obj-session")
            sid_obj = _open(reg_obj)["session_id"]
            ctx_obj = reg_obj.get(sid_obj)
            result_obj = controller.submit_work(ctx_obj, {"value": 7})

            reg_str = _fresh_registry(tmpdir, session_id="str-session")
            sid_str = _open(reg_str)["session_id"]
            ctx_str = reg_str.get(sid_str)
            result_str = controller.submit_work(ctx_str, json.dumps({"value": 7}))

            self.assertTrue(result_obj["valid"])
            self.assertTrue(result_str["valid"])
            self.assertEqual(
                result_str["evidence"]["measurements"],
                result_obj["evidence"]["measurements"],
            )
            self.assertEqual(ctx_str["current_source"], ctx_obj["current_source"])
            self.assertEqual(
                ctx_str["store"].load_version_source(ctx_str["artifact_id"], "v1"),
                ctx_obj["store"].load_version_source(ctx_obj["artifact_id"], "v1"),
            )

    def test_submit_work_unparseable_string_still_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = _fresh_registry(tmpdir, session_id="bad-session")
            sid = _open(reg)["session_id"]
            ctx = reg.get(sid)
            result = controller.submit_work(ctx, "this is not json at all {{{")
            self.assertFalse(result["valid"])
            self.assertTrue(result["errors"])
            self.assertFalse(ctx["_work_submitted"])
            self.assertIsNone(ctx["current_source"])
            record = ctx["store"].load(ctx["artifact_id"])
            self.assertEqual(record.get("versions", []), [])


class JudgeVerdictReachesModelTests(unittest.TestCase):
    def test_recorded_verdict_appears_in_session_so_far(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = _fresh_registry(tmpdir, session_id="judged-session")
            sid = _open(reg)["session_id"]
            ctx = reg.get(sid)
            controller.submit_work(ctx, {"value": 1})

            verdict = {
                "bottom_line": "mixed",
                "judgment": "UNIQUE_JUDGMENT_MARKER the work is uneven",
                "strengths": ["UNIQUE_STRENGTH_MARKER clear opening"],
                "weaknesses": ["UNIQUE_WEAKNESS_MARKER thin close"],
            }
            recorded = judge.record_verdict(reg, sid, verdict)
            self.assertEqual(recorded, {"recorded": True, "session_id": sid})

            entries = _history_from_disk(tmpdir, sid)
            judgments = [e for e in entries if e.get("event") == "judgment"]
            self.assertEqual(len(judgments), 1)
            self.assertEqual(judgments[0]["verdict"], verdict)

            sofar = build_session_so_far(entries, ctx.get("evidence_summaries") or [])
            self.assertIn("mixed", sofar)
            self.assertIn("UNIQUE_JUDGMENT_MARKER", sofar)
            self.assertIn("UNIQUE_STRENGTH_MARKER", sofar)
            self.assertIn("UNIQUE_WEAKNESS_MARKER", sofar)
            self.assertIn("bottom_line", sofar)
            self.assertIn("judgment", sofar)
            self.assertIn("strengths", sofar)
            self.assertIn("weaknesses", sofar)

            # The resume path carries the same judgments to the model.
            del reg
            fresh = _fresh_registry(tmpdir)
            resumed = dispatch(fresh, "sensoryne_resume_session", {"session_id": sid})
            resumed_sofar = resumed["session_so_far"]
            self.assertIn("mixed", resumed_sofar)
            self.assertIn("UNIQUE_JUDGMENT_MARKER", resumed_sofar)
            self.assertIn("UNIQUE_STRENGTH_MARKER", resumed_sofar)
            self.assertIn("UNIQUE_WEAKNESS_MARKER", resumed_sofar)

    def test_session_with_no_verdicts_has_no_judgments_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = _fresh_registry(tmpdir, session_id="plain-session")
            sid = _open(reg)["session_id"]
            ctx = reg.get(sid)
            controller.submit_work(ctx, {"value": 1})
            entries = _history_from_disk(tmpdir, sid)
            sofar = build_session_so_far(entries, ctx.get("evidence_summaries") or [])
            self.assertNotIn("Judgments:", sofar)
            self.assertIn("Session so far:", sofar)


class ToolDescriptionTests(unittest.TestCase):
    def test_submit_work_and_decision_descriptions_state_json_object_form(self):
        by_name = {t["name"]: t["description"] for t in TOOLS}
        work = by_name["sensoryne_submit_work"]
        decision = by_name["sensoryne_submit_decision"]
        self.assertIn("JSON object", work)
        self.assertIn("instrument", work)
        self.assertIn("parts", work)
        self.assertIn("velocity", work)
        self.assertIn("0 to 1", work)
        self.assertIn("JSON object", decision)
        self.assertIn("faults", decision)
        self.assertIn("revisions", decision)
        self.assertIn("complete", decision)
        self.assertIn("assessment", decision)


if __name__ == "__main__":
    unittest.main()
