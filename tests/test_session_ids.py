"""tests/test_session_ids.py - session id minting from the store.

A session id minted by open_session must never collide with a
session that already exists in the store, even when the new
session is opened through a fresh registry standing in for a
fresh server process. The second session gets its own artifact
with no versions from the first, and the first session's
history file is unchanged.

A small music perceptual session opened after an earlier music
session exists in the same store hears the same notes the
music comparison computes directly on the same score and
render.
"""

import copy
import math
import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2 import controller
from stage2.store import ArtifactStore
from stage2.tools import ToolRegistry

SAMPLE_RATE = 44100


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


def _open(registry, application="stub", condition="perceptual"):
    result = controller.open_session(
        registry,
        {
            "application": application,
            "objective": {"goal": "value matters"},
            "condition": condition,
            "test": False,
        },
    )
    return result, registry.get(result["session_id"])


def _tone(midi, seconds, amplitude=0.4):
    count = int(round(seconds * SAMPLE_RATE))
    t = np.arange(count, dtype=np.float64) / float(SAMPLE_RATE)
    freq = 440.0 * (2.0 ** ((int(midi) - 69) / 12.0))
    envelope = np.ones(count, dtype=np.float64)
    fade = max(1, int(0.01 * SAMPLE_RATE))
    if count > fade * 2:
        envelope[:fade] = np.linspace(0.0, 1.0, fade)
        envelope[-fade:] = np.linspace(1.0, 0.0, fade)
    return amplitude * envelope * np.sin(2.0 * math.pi * freq * t)


def _music_renderer(artifact, out_path, faults):
    """A small sine renderer in the suite's comparison style."""
    total = 0.0
    for part in artifact.get("parts", []):
        for note in part.get("notes", []):
            dur = float(note["dur"]) if note.get("dur") is not None else 0.5
            total = max(total, float(note["start"]) + dur)
    buffer = np.zeros(int(math.ceil(total * SAMPLE_RATE)) + SAMPLE_RATE // 2)
    for part in artifact.get("parts", []):
        gain = float(part.get("gain", 1.0))
        for note in part.get("notes", []):
            start = float(note["start"])
            dur = float(note["dur"]) if note.get("dur") is not None else 0.5
            effective = float(note.get("vel", 0.0)) * gain
            sample = _tone(int(note["midi"]), dur, effective)
            begin = int(start * SAMPLE_RATE)
            end = min(len(buffer), begin + len(sample))
            buffer[begin:end] += sample[: end - begin]
    pcm = np.round(np.clip(buffer, -1.0, 1.0) * 32767.0).astype(np.int16)
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    wavfile.write(str(out_path), SAMPLE_RATE, pcm)
    return {"path": str(out_path)}


def _small_score():
    return {
        "instrument": "piano",
        "parts": [
            {
                "name": "solo",
                "gain": 1.0,
                "notes": [
                    {"start": float(i), "midi": 69, "vel": 0.4, "dur": 1.0}
                    for i in range(4)
                ],
            }
        ],
    }


class SessionIdCollisionTests(unittest.TestCase):
    def test_fresh_registry_mints_a_different_id_with_its_own_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_result, first = _open(_fresh_registry(tmpdir))
            controller.submit_work(first, {"value": 1})
            controller.submit_decision(
                first,
                {
                    "faults": [],
                    "revisions": [],
                    "complete": True,
                    "assessment": "first done",
                },
            )
            self.assertEqual(first["ended_by"], "model_complete")
            history_path = os.path.join(first["session_dir"], "history.jsonl")
            with open(history_path, "rb") as handle:
                before = handle.read()
            first_versions = copy.deepcopy(
                first["store"].load(first["artifact_id"])["versions"]
            )
            self.assertEqual(len(first_versions), 1)
            self.assertEqual(first_result["session_id"], "stub-perceptual-1")
            self.assertTrue(first_result["session_id"].startswith("stub-perceptual-"))

            second_result, second = _open(_fresh_registry(tmpdir))
            self.assertNotEqual(second_result["session_id"], first_result["session_id"])
            self.assertTrue(second_result["session_id"].startswith("stub-perceptual-"))
            self.assertEqual(second["artifact_id"], second_result["session_id"])
            self.assertNotEqual(second["artifact_id"], first["artifact_id"])
            controller.submit_work(second, {"value": 2})
            controller.submit_decision(
                second,
                {
                    "faults": [],
                    "revisions": [],
                    "complete": True,
                    "assessment": "second done",
                },
            )
            # The first session's history file is unchanged,
            # and its artifact has no versions from the second.
            with open(history_path, "rb") as handle:
                after = handle.read()
            self.assertEqual(before, after)
            self.assertEqual(
                first["store"].load(first["artifact_id"])["versions"],
                first_versions,
            )
            second_versions = second["store"].load(second["artifact_id"])["versions"]
            self.assertEqual(len(second_versions), 1)
            self.assertEqual(second_versions[0]["version_id"], "v1")
            self.assertNotIn(
                copy.deepcopy(second_versions),
                [copy.deepcopy(first_versions)],
            )

    def test_ids_advance_within_one_registry_and_ignore_other_conditions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = _fresh_registry(tmpdir)
            first, _ctx1 = _open(registry)
            second, _ctx2 = _open(registry)
            self.assertEqual(first["session_id"], "stub-perceptual-1")
            self.assertEqual(second["session_id"], "stub-perceptual-2")
            self.assertNotEqual(first["session_id"], second["session_id"])


class MusicAfterEarlierSessionTests(unittest.TestCase):
    def test_music_session_after_earlier_music_session_hears_the_same_notes(self):
        from stage2.apps.music.comparison import compute_comparison
        from stage2.apps.music.perceive import perceive_wav

        with tempfile.TemporaryDirectory() as tmpdir:
            score = _small_score()

            def _music_registry():
                registry = ToolRegistry()
                registry._session_counter = 0
                registry.pending_context = {
                    "store": ArtifactStore(os.path.join(tmpdir, "artifacts")),
                    "renderer": _music_renderer,
                    "session_root": os.path.join(tmpdir, "sessions"),
                    "objective": {"goal": "a short tune"},
                    "faults": [],
                    "max_iterations": 4,
                    "ledger_path": None,
                }
                return registry

            first_result, first = _open(_music_registry(), application="music")
            controller.submit_work(first, copy.deepcopy(score))
            controller.submit_decision(
                first,
                {
                    "faults": [],
                    "revisions": [],
                    "complete": True,
                    "assessment": "first tune done",
                },
            )
            self.assertEqual(first["ended_by"], "model_complete")
            first_history = os.path.join(first["session_dir"], "history.jsonl")
            with open(first_history, "rb") as handle:
                before = handle.read()
            first_versions = copy.deepcopy(
                first["store"].load(first["artifact_id"])["versions"]
            )

            second_result, second = _open(_music_registry(), application="music")
            self.assertNotEqual(
                second_result["session_id"], first_result["session_id"]
            )
            result = controller.submit_work(second, copy.deepcopy(score))
            self.assertTrue(result["valid"])
            self.assertEqual(second["artifact_id"], second_result["session_id"])
            self.assertNotEqual(second["artifact_id"], first["artifact_id"])

            perception = [
                e
                for e in second["history"].entries
                if e.get("event") == "perception"
            ]
            self.assertEqual(len(perception), 1)
            heard_per_part = perception[-1]["evidence"]["comparison"]["per_part"]
            self.assertEqual(
                sum(entry["notes_heard"] for entry in heard_per_part), 4
            )

            # The music comparison computed directly on the
            # same score and render gives the same counts.
            render_path = perception[-1]["evidence"]["render_path"]
            direct_evidence = perceive_wav(render_path, artifact=score)
            direct = compute_comparison(
                score, {"path": render_path}, direct_evidence
            )
            self.assertEqual(heard_per_part, direct["per_part"])
            self.assertEqual(
                sum(entry["notes_heard"] for entry in heard_per_part),
                sum(entry["notes_heard"] for entry in direct["per_part"]),
            )
            self.assertEqual(
                sum(entry["notes_written"] for entry in heard_per_part),
                sum(entry["notes_written"] for entry in direct["per_part"]),
            )

            # The earlier session stayed untouched.
            with open(first_history, "rb") as handle:
                self.assertEqual(before, handle.read())
            self.assertEqual(
                first["store"].load(first["artifact_id"])["versions"],
                first_versions,
            )
            self.assertEqual(
                len(second["store"].load(second["artifact_id"])["versions"]), 1
            )


if __name__ == "__main__":
    unittest.main()
