"""tests/test_music_invariants_controller.py - regression test for the
invariants hearing-count defect.

Assembles an evidence pack through the controller's own path
(open_session -> submit_work) for a Music & Sound score work
with a known comparison, and asserts the invariants block's
per-part heard figures equal the comparison block's in the
same pack, and the invariants' event realized figures equal
the comparison's event figures where the comparison carries
them. Also asserts the no-comparison (control) pack does not
state a heard count of zero or list written notes as not
heard.

A test that calls evaluate_invariants directly with a
hand-built record does not cover this defect and is not
sufficient on its own; every assertion here goes through
controller.submit_work, whose returned document is shaped
(see stage2/shape.py) but whose stored session record is
complete (see stage2/controller.py submit_work comment).
This test reads the complete stored evidence so the
comparison block itself is asserted on, not only its shaped
summary.
"""

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2 import controller
from stage2.store import ArtifactStore
from stage2.tools import ToolRegistry
from music_renderer import renderer as music_renderer


def _score():
    return {
        "instrument": "piano",
        "parts": [
            {
                "name": "melody",
                "gain": 1.0,
                "notes": [
                    {"start": 0.0, "midi": 72, "vel": 0.4, "dur": 0.9},
                    {"start": 1.0, "midi": 74, "vel": 0.4, "dur": 0.9},
                ],
            },
            {
                "name": "harmony",
                "gain": 1.0,
                "notes": [
                    {"start": 0.0, "midi": 60, "vel": 0.3, "dur": 0.9},
                    {"start": 1.0, "midi": 62, "vel": 0.3, "dur": 0.9},
                ],
            },
            {
                "name": "bass",
                "gain": 1.0,
                "notes": [{"start": 0.0, "midi": 36, "vel": 0.3, "dur": 0.9}],
            },
        ],
    }


def _open_music_session(tmpdir, session_id, condition):
    store = ArtifactStore(os.path.join(tmpdir, "artifacts"))
    registry = ToolRegistry()
    registry._session_counter = 0
    registry.pending_context = {
        "store": store,
        "app": None,
        "renderer": music_renderer,
        "session_root": os.path.join(tmpdir, "sessions"),
        "session_id": session_id,
        "artifact_id": session_id + "-artifact",
        "objective": {"goal": "a short piano piece"},
        "condition": condition,
        "test": False,
        "faults": [],
        "max_iterations": 4,
        "ledger_path": None,
    }
    controller.open_session(
        registry,
        {
            "application": "music",
            "objective": {"goal": "a short piano piece"},
            "condition": condition,
            "test": False,
        },
    )
    return registry, registry.get(session_id)


def _stored_evidence(context):
    for entry in reversed(context["history"].entries):
        if entry.get("event") == "perception":
            return entry.get("evidence")
    return None


def _checked_by_id(block):
    return {entry["id"]: entry for entry in block["checked"]}


def _per_part_heard(block):
    check = _checked_by_id(block)["notes_heard_of_written"]
    return {
        entry["part"]: entry["notes_heard"]
        for entry in check["value"]
    }


class ControllerEvidencePackTests(unittest.TestCase):
    def test_perceptual_pack_invariants_match_comparison(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context = _open_music_session(tmpdir, "music-pkg-1", "perceptual")
            result = controller.submit_work(context, _score())
            self.assertTrue(result["valid"])

            evidence = _stored_evidence(context)
            self.assertIsNotNone(evidence)

            comparison = evidence.get("comparison")
            self.assertIsNotNone(comparison)
            per_part = comparison.get("per_part")
            self.assertTrue(per_part)

            # Known clean render: every written note is heard.
            expected = {entry["part"]: entry["notes_heard"] for entry in per_part}
            self.assertEqual(expected, {"melody": 2, "harmony": 2, "bass": 1})

            invariants = evidence.get("invariants")
            self.assertIsNotNone(invariants)
            self.assertEqual(_per_part_heard(invariants), expected)

            # The values must be measured figures, not the
            # no-comparison zero branch.
            heard_check = _checked_by_id(invariants)["notes_heard_of_written"]
            self.assertTrue(heard_check["held"])
            self.assertEqual(heard_check["violations"], [])

            # The stored pack is the complete evidence the
            # platform holds for this version: record and
            # returned copy carry the same invariants.
            version = context["store"].load(context["artifact_id"])["versions"][-1]
            self.assertEqual(version["render_info"]["invariants"], invariants)

    def test_control_pack_states_no_heard_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context = _open_music_session(tmpdir, "music-pkg-2", "control")
            result = controller.submit_work(context, _score())
            self.assertTrue(result["valid"])
            self.assertEqual(result["perception"], "perception_skipped")

            version = context["store"].load(context["artifact_id"])["versions"][-1]
            inv = version["render_info"]["invariants"]
            heard_check = _checked_by_id(inv)["notes_heard_of_written"]
            self.assertFalse(heard_check["held"])
            self.assertEqual(
                heard_check["value"],
                [
                    {"part": "melody", "notes_written": 2, "notes_heard": None},
                    {"part": "harmony", "notes_written": 2, "notes_heard": None},
                    {"part": "bass", "notes_written": 1, "notes_heard": None},
                ],
            )
            self.assertEqual(heard_check["violations"], [])


class ControllerInvariantsMatchComparisonTests(unittest.TestCase):
    def test_invariants_per_part_equal_comparison_in_same_pack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context = _open_music_session(tmpdir, "music-pkg-3", "perceptual")
            controller.submit_work(context, _score())
            evidence = _stored_evidence(context)
            comparison = evidence["comparison"]
            invariants = evidence["invariants"]

            inv_heard = _per_part_heard(invariants)
            comp_heard = {
                entry["part"]: entry["notes_heard"]
                for entry in comparison["per_part"]
            }
            self.assertEqual(inv_heard, comp_heard)

            # The not-heard violations carry the comparison's
            # not-heard figures, not the full written count.
            heard_check = _checked_by_id(invariants)["notes_heard_of_written"]
            comp_not_heard = {
                entry["part"]: entry["notes_not_heard"]
                for entry in comparison["per_part"]
            }
            for entry in heard_check["violations"]:
                self.assertEqual(entry["notes_not_heard"], comp_not_heard[entry["part"]])
                self.assertEqual(entry["notes_heard"], comp_heard[entry["part"]])


if __name__ == "__main__":
    unittest.main()
