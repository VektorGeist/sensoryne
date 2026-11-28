"""tests/test_music_sound_comparison.py - merged Music & Sound tests.

The event comparison tests formerly for the separate event
application now run under the merged music application, in
substance (same planted faults, same finding kinds). Added:
an event-only work judged in event terms, a score-only work
judged as before, and a work with both judged in both.

No model calls are made.
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

from stage2.apps.music.app import perceive, validate
from stage2.apps.music.comparison import compute_comparison
from stage2.apps.music.invariants import objective_coverage
from music_renderer import renderer

SAMPLE_RATE = 44100


def _work():
    return {
        "duration": 6.0,
        "events": [
            {"id": "hit", "kind": "percussive hit", "start": 0.0, "duration": 0.5, "gain": 0.8, "filter_cutoff": 180, "bandwidth": 0},
            {"id": "wash", "kind": "filtered noise", "start": 1.0, "duration": 2.0, "gain": 0.4, "filter_cutoff": 1200, "bandwidth": 800},
            {"id": "tail", "kind": "noise burst", "start": 4.0, "duration": 0.5, "gain": 0.5, "filter_cutoff": 4000, "bandwidth": 1000},
        ],
    }


def _compare(artifact, tmpdir, delivered_artifact=None, faults=(), label="render"):
    source = copy.deepcopy(delivered_artifact) if delivered_artifact is not None else artifact
    path = os.path.join(tmpdir, label + ".wav")
    record = renderer(source, path, list(faults))
    record["_artifact"] = copy.deepcopy(artifact)
    evidence = perceive(record)
    return compute_comparison(artifact, record, evidence), record, evidence


def _findings(comparison, kind):
    return [finding for finding in comparison["findings"] if finding["kind"] == kind]


def _coverage_with_comparison(artifact, record, evidence, comparison):
    evidence_with_comparison = copy.deepcopy(evidence)
    record_with_comparison = copy.deepcopy(record)
    evidence_with_comparison["comparison"] = comparison
    record_with_comparison["comparison"] = comparison
    return objective_coverage(artifact, record_with_comparison, evidence_with_comparison)


class MusicSoundEventComparisonTests(unittest.TestCase):
    def test_missing_written_event_produces_event_not_realized_naming_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _work()
            comparison, _record, _evidence = _compare(artifact, tmpdir, faults=({"kind": "drop_event", "id": "wash"},), label="missing-event")
            matched = _findings(comparison, "event_not_realized")
            named = [f for f in matched if f["written"].get("id") == "wash" or f["written"].get("event") == "wash" or "'wash'" in f["statement"]]
            self.assertTrue(named, "expected event_not_realized naming wash, got %r" % (comparison["findings"],))
            finding = named[0]
            self.assertIn("wash", finding["statement"])
            self.assertEqual(finding["written"].get("id"), "wash")
            self.assertAlmostEqual(finding["written"].get("written_onset"), 1.0, places=6)
            self.assertIn("1.0", finding["statement"])
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertIn("event_not_realized", comparison["summary"]["counts"])
            self.assertEqual(comparison["tolerances"]["onset_tolerance_seconds"], 0.1)

    def test_event_half_a_second_late_produces_onset_displaced_with_both_times(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _work()
            comparison, _record, _evidence = _compare(artifact, tmpdir, faults=({"kind": "shift_event_start", "id": "wash", "from": 1.0, "new_start": 1.5},), label="displaced")
            matched = _findings(comparison, "onset_displaced")
            named = [f for f in matched if f["written"].get("id") == "wash" or f["written"].get("event") == "wash"]
            self.assertTrue(named, "expected onset_displaced naming wash, got %r" % (comparison["findings"],))
            finding = named[0]
            self.assertAlmostEqual(finding["written"].get("written_onset"), 1.0, places=6)
            delivered = finding["delivered"]
            realized = delivered.get("realized_onset")
            if realized is None:
                realized = delivered.get("onset_time")
            self.assertIsNotNone(realized)
            self.assertAlmostEqual(realized, 1.5, delta=0.1)
            displacement = delivered.get("displacement_seconds")
            if displacement is None:
                displacement = delivered.get("difference_seconds")
            if displacement is None and realized is not None:
                displacement = float(realized) - float(finding["written"].get("written_onset"))
            self.assertAlmostEqual(displacement, 0.5, delta=0.1)
            self.assertIn("wash", finding["statement"])
            self.assertIn(str(finding["written"].get("written_onset")), finding["statement"])
            self.assertIn(str(realized), finding["statement"])
            self.assertEqual(_findings(comparison, "event_not_realized"), [])

    def test_extra_rendered_sound_produces_onset_not_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _work()
            delivered = copy.deepcopy(artifact)
            delivered["events"].append({"id": "extra", "kind": "noise burst", "start": 5.0, "duration": 0.5, "gain": 0.5, "filter_cutoff": 4000, "bandwidth": 1000})
            comparison, _record, _evidence = _compare(artifact, tmpdir, delivered_artifact=delivered, label="extra-sound")
            matched = _findings(comparison, "onset_not_written")
            self.assertTrue(matched, "expected onset_not_written, got %r" % (comparison["findings"],))
            finding = matched[0]
            delivered_block = finding["delivered"]
            realized = delivered_block.get("realized_onset")
            if realized is None:
                realized = delivered_block.get("onset_time")
            if realized is None:
                realized = delivered_block.get("onset")
            self.assertIsNotNone(realized)
            self.assertAlmostEqual(realized, 5.0, delta=0.1)
            self.assertIn(str(realized), finding["statement"])
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertEqual(_findings(comparison, "event_not_realized"), [])
            self.assertEqual(_findings(comparison, "onset_displaced"), [])

    def test_clean_delivery_produces_zero_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _work()
            comparison, record, evidence = _compare(artifact, tmpdir)
            self.assertEqual(comparison["findings"], [])
            self.assertEqual(comparison["summary"]["total"], 0)
            self.assertEqual(comparison["summary"]["counts"], {})
            self.assertIn("No findings", comparison["summary"]["line"])
            self.assertEqual(comparison["events_written"], 3)
            self.assertEqual(comparison["events_realized_at_written_onsets"], 3)
            coverage = _coverage_with_comparison(artifact, record, evidence, comparison)
            self.assertEqual(coverage, [{"item": "events realized at their written onsets, 3 of 3", "covered": True}])

    def test_event_only_work_judged_in_event_terms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _work()
            self.assertEqual(validate(artifact), [])
            comparison, _record, _evidence = _compare(artifact, tmpdir, faults=({"kind": "drop_event", "id": "hit"},), label="event-only")
            kinds = {f["kind"] for f in comparison["findings"]}
            self.assertIn("event_not_realized", kinds)
            self.assertTrue(kinds <= {"event_not_realized", "onset_displaced", "onset_not_written"}, kinds)
            self.assertEqual(comparison["events_written"], 3)


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


def _render_score(score, path):
    total = 0.0
    for part in score.get("parts", []):
        for note in part.get("notes", []):
            total = max(total, float(note["start"]) + float(note["dur"]))
    buffer = np.zeros(int(math.ceil(total * SAMPLE_RATE)) + SAMPLE_RATE // 2)
    for part in score.get("parts", []):
        gain = float(part.get("gain", 1.0))
        for note in part.get("notes", []):
            sample = _tone(int(note["midi"]), float(note["dur"]), float(note.get("vel", 0.0)) * gain)
            begin = int(float(note["start"]) * SAMPLE_RATE)
            end = min(len(buffer), begin + len(sample))
            buffer[begin:end] += sample[: end - begin]
    pcm = np.round(np.clip(buffer, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, pcm)
    return {"path": path, "duration": total}


class MusicSoundMergedTests(unittest.TestCase):
    def test_score_only_work_judged_as_before(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            score = {"instrument": "piano", "parts": [{"name": "solo", "gain": 1.0, "notes": [{"start": float(i), "midi": 69, "vel": 0.4, "dur": 1.0} for i in range(5)]}]}
            self.assertEqual(validate(score), [])
            path = os.path.join(tmpdir, "score.wav")
            record = _render_score(score, path)
            record["_artifact"] = copy.deepcopy(score)
            evidence = perceive(record)
            comparison = compute_comparison(score, record, evidence)
            self.assertEqual(_findings(comparison, "note_not_heard"), [])
            self.assertEqual(_findings(comparison, "pitch_off"), [])
            self.assertEqual(comparison["per_part"], [{"part": "solo", "notes_written": 5, "notes_heard": 5, "notes_not_heard": 0}])
            self.assertNotIn("events_written", comparison)

    def test_work_with_both_judged_in_both(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = {
                "instrument": "piano",
                "duration": 6.0,
                "parts": [{"name": "solo", "gain": 1.0, "notes": [{"start": 0.0, "midi": 69, "vel": 0.4, "dur": 1.0}, {"start": 1.0, "midi": 72, "vel": 0.4, "dur": 1.0}]}],
                "events": _work()["events"],
            }
            self.assertEqual(validate(artifact), [])
            path = os.path.join(tmpdir, "both.wav")
            record = renderer(artifact, path, [])
            self.assertTrue(os.path.exists(path))
            self.assertEqual(record["event_count"], 3)
            record["_artifact"] = copy.deepcopy(artifact)
            evidence = perceive(record)
            comparison = compute_comparison(artifact, record, evidence)
            # score terms and event terms both present
            self.assertIn("per_part", comparison)
            self.assertEqual(comparison["events_written"], 3)
            self.assertIn("pitch_tolerance_cents", comparison["tolerances"])
            self.assertEqual(comparison["tolerances"]["onset_tolerance_seconds"], 0.1)
            # a dropped note and a dropped event each produce their own kind
            path2 = os.path.join(tmpdir, "both-faulted.wav")
            record2 = renderer(artifact, path2, [{"kind": "drop_event", "id": "wash"}])
            record2["_artifact"] = copy.deepcopy(artifact)
            evidence2 = perceive(record2)
            comparison2 = compute_comparison(artifact, record2, evidence2)
            kinds2 = {f["kind"] for f in comparison2["findings"]}
            self.assertIn("event_not_realized", kinds2)
            self.assertIn("per_part", comparison2)


if __name__ == "__main__":
    unittest.main()
