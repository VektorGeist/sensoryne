"""tests/test_shaping.py - evidence return-shaping tests.

The stored session record keeps the complete evidence exactly
as perceived; only the copy handed back through a tool return
is shaped. A large stored evidence therefore returns with the
model token pack whole while the stored record stays complete,
and a small evidence passes through unshaped in substance (its
sections intact, only the shaping metadata added).

The model token pack is exempt from shaping in whole and in
part: shaping never replaces the pack, a segment, or any field
inside a segment, at any budget. Other sections take any
reduction, so the shaped copy's non-pack sections stay bounded
even though the pack itself is carried in full.

No model calls are made. The large case is the real music
session's stored evidence (sessions/music-perceptual-1),
read from its history record; if that record is not present
the large case is skipped rather than failing on a missing
fixture.
"""

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2 import controller
from stage2.shape import EVIDENCE_BUDGET, shape_document, shape_evidence, _size
from stage2.store import ArtifactStore
from stage2.tools import ToolRegistry, dispatch


def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _music_evidence():
    path = os.path.join(_root(), "sessions", "music-perceptual-1", "history.jsonl")
    if not os.path.isfile(path):
        return None, None
    start = None
    evidence = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("event") == "session_start":
                start = entry
            elif entry.get("event") == "perception":
                evidence = entry.get("evidence")
    return evidence, start


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
        # A large synthetic payload alongside the small stated
        # figures, so shaping has bulk to reference-replace.
        return {
            "summary": "value %s" % value,
            "measurements": {
                "value": value,
                "double": value * 2,
                "waveform": list(range(20000)),
            },
            "invariants": {
                "checked": [{"id": "value_ok", "held": True, "value": value}],
                "summary": {"total_checked": 1, "held": 1, "broken": 0},
            },
            "reliability": {"level": "high"},
            "objective_coverage": [{"goal": "value matters", "covered": True}],
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


def _open_stub(tmpdir):
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
            "application": "stub",
            "objective": {"goal": "value matters"},
            "condition": "perceptual",
            "test": False,
        },
    )
    return registry, registry.get("stub-session")


class ShapeFunctionTests(unittest.TestCase):
    def test_small_evidence_passes_through_unshaped_in_substance(self):
        evidence = {
            "summary": "a short heard summary",
            "measurements": {"value": 3, "double": 6},
            "invariants": {
                "checked": [{"id": "value_ok", "held": True, "value": 3}],
                "summary": {"total_checked": 1, "held": 1, "broken": 0},
            },
            "reliability": {"level": "high"},
            "objective_coverage": [{"goal": "g", "covered": True}],
            "delta": {"from_version": "v1", "to_version": "v2"},
        }
        original = copy.deepcopy(evidence)
        shaped = shape_evidence(evidence)
        # Input is not mutated and every original section is
        # intact in the shaped copy.
        self.assertEqual(evidence, original)
        for key in original:
            self.assertEqual(shaped[key], original[key])
        self.assertTrue(shaped["shaped_copy"])
        self.assertIn("shaped copy", shaped["shaping_note"])
        self.assertIn("session record", shaped["complete_evidence_location"])
        self.assertLess(_size(shaped), EVIDENCE_BUDGET)

    def test_shaping_never_mutates_its_input(self):
        evidence, _start = _music_evidence()
        if evidence is None:
            self.skipTest("music session record not present")
        before = _size(evidence)
        shaped = shape_evidence(evidence)
        self.assertEqual(_size(evidence), before)
        self.assertLess(_size(shaped), before)


class MusicEvidenceShapingTests(unittest.TestCase):
    def test_large_music_evidence_returns_bounded_while_stored_stays_complete(self):
        evidence, start = _music_evidence()
        if evidence is None:
            self.skipTest("music session record not present")
        original_size = _size(evidence)
        self.assertGreater(original_size, 10 * EVIDENCE_BUDGET)

        shaped = shape_evidence(evidence)

        # The stored evidence is untouched at full size and the
        # shaped copy states that it is a shaped copy.
        self.assertEqual(_size(evidence), original_size)
        self.assertTrue(shaped["shaped_copy"])
        self.assertIn("shaped copy", shaped["shaping_note"].lower())
        self.assertIn("session record", shaped["complete_evidence_location"])

        # The pack is exempt and whole: byte-identical to the
        # stored pack, every segment in full, no reference
        # anywhere inside it. Other sections take any reduction.
        shaped_pack = shaped["measurements"]["token_pack"]
        original_pack = evidence["measurements"]["token_pack"]
        self.assertNotIn("_reference", shaped_pack)
        self.assertEqual(
            json.dumps(shaped_pack, sort_keys=True, ensure_ascii=False, default=str),
            json.dumps(original_pack, sort_keys=True, ensure_ascii=False, default=str),
        )
        shaped_segments = shaped_pack["segments_data"]
        original_segments = original_pack["segments_data"]
        self.assertEqual(len(shaped_segments), len(original_segments))
        for shaped_seg, original_seg in zip(shaped_segments, original_segments):
            self.assertNotIn("_reference", shaped_seg)
            self.assertEqual(shaped_seg, original_seg)
        # No reference anywhere inside the pack subtree.
        self.assertNotIn("_reference", json.dumps(shaped_pack, default=str))

        # Every other required section survived shaping in full
        # or as a bounded reference-replaced subtree, and the
        # non-pack shaped copy stays bounded: the stored record's
        # non-pack bulk (the full repeat copies) is replaced.
        self.assertEqual(shaped["summary"], evidence["summary"])
        for key in ("pitch_hz", "loudness", "duration"):
            if key in evidence["measurements"]:
                self.assertEqual(shaped["measurements"][key], evidence["measurements"][key])
        self.assertEqual(
            [c["id"] for c in shaped["invariants"]["checked"]],
            [c["id"] for c in evidence["invariants"]["checked"]],
        )
        self.assertEqual(
            [c["held"] for c in shaped["invariants"]["checked"]],
            [c["held"] for c in evidence["invariants"]["checked"]],
        )
        self.assertEqual(shaped["reliability"], evidence["reliability"])
        self.assertEqual(shaped["objective_coverage"], evidence["objective_coverage"])
        # The repeat copies of the measurements are references;
        # the stored record keeps them complete.
        self.assertIn("_reference", shaped["full"]["measurements"])
        self.assertNotIn("_reference", evidence["full"]["measurements"])
        self.assertNotIn("shaped_copy", evidence)

        # The whole tool-result document shapes the same way:
        # the pack stays whole, structure and input kept in full.
        document = {
            "valid": True,
            "session_id": "music-perceptual-1",
            "iteration": 1,
            "evidence": copy.deepcopy(evidence),
            "structure_record": copy.deepcopy(start["structure_record"]),
            "input_summary": copy.deepcopy(start["input_summary"]),
        }
        shaped_doc = shape_document(document)
        self.assertTrue(shaped_doc["shaped_copy"])
        doc_pack = shaped_doc["evidence"]["measurements"]["token_pack"]
        self.assertEqual(
            json.dumps(doc_pack, sort_keys=True, ensure_ascii=False, default=str),
            json.dumps(original_pack, sort_keys=True, ensure_ascii=False, default=str),
        )
        self.assertEqual(shaped_doc["structure_record"], start["structure_record"])
        self.assertEqual(shaped_doc["input_summary"], start["input_summary"])


class ReturnBoundaryTests(unittest.TestCase):
    def test_submit_work_returns_shaped_copy_and_stores_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context = _open_stub(tmpdir)
            result = controller.submit_work(context, {"value": 2})
            self.assertTrue(result["shaped_copy"])
            self.assertIn("shaped copy", result["shaping_note"].lower())
            evidence = result["evidence"]
            self.assertEqual(evidence["summary"], "value 2")
            self.assertEqual(evidence["measurements"]["value"], 2)
            # Bulk waveform in the returned copy is a reference.
            self.assertIn("_reference", evidence["measurements"]["waveform"])
            # The stored record keeps the complete evidence.
            perception = [
                e for e in context["history"].entries if e.get("event") == "perception"
            ]
            stored = perception[-1]["evidence"]
            self.assertNotIn("shaped_copy", stored)
            self.assertEqual(stored["measurements"]["waveform"], list(range(20000)))

    def test_submit_decision_returns_shaped_copy_with_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context = _open_stub(tmpdir)
            controller.submit_work(context, {"value": 1})
            result = controller.submit_decision(
                context,
                {
                    "faults": [],
                    "revisions": [{"op": "set_value", "value": 4}],
                    "complete": False,
                    "assessment": "continue",
                },
            )
            self.assertTrue(result["shaped_copy"])
            self.assertEqual(result["evidence"]["delta"]["to_version"], "v2")
            self.assertEqual(result["evidence"]["measurements"]["value"], 4)
            perception = [
                e for e in context["history"].entries if e.get("event") == "perception"
            ]
            self.assertNotIn("shaped_copy", perception[-1]["evidence"])

    def test_probe_returns_shaped_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry, context = _open_stub(tmpdir)
            controller.submit_work(context, {"value": 1})
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
            self.assertTrue(result["shaped_copy"])
            self.assertEqual(result["measurements"]["value"], 5)
            self.assertEqual(result["delta"]["to_version"], "v2-probe")


if __name__ == "__main__":
    unittest.main()
