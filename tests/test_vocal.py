"""tests/test_vocal.py - vocal extension tests.

Generates a synthetic sung render with numpy inside the test,
writes it as a WAV, and measures it through the music
application's hearing. No model calls are made.
"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.apps.music.app import apply_revision, structure_record, validate
from stage2.apps.music.invariants import evaluate_invariants
from stage2.apps.music.perceive import perceive_wav


SAMPLE_RATE = 44100


def _tone(freq, seconds, amplitude=0.45):
    count = int(round(seconds * SAMPLE_RATE))
    t = np.arange(count, dtype=np.float64) / float(SAMPLE_RATE)
    envelope = np.ones(count, dtype=np.float64)
    fade = max(1, int(0.03 * SAMPLE_RATE))
    if count > fade * 2:
        envelope[:fade] = np.linspace(0.0, 1.0, fade)
        envelope[-fade:] = np.linspace(1.0, 0.0, fade)
    return amplitude * envelope * np.sin(2.0 * math.pi * freq * t)


def _write_wav(path, samples):
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, pcm)
    return path


def _sung_render(path):
    """Two phrases separated by a silence gap.

    Phrase one: two 440 Hz syllable tones.
    Phrase two: two 523.25 Hz syllable tones.
    """
    parts = [
        _tone(440.0, 0.45),
        np.zeros(int(0.08 * SAMPLE_RATE), dtype=np.float64),
        _tone(440.0, 0.45),
        np.zeros(int(0.40 * SAMPLE_RATE), dtype=np.float64),
        _tone(523.25, 0.45),
        np.zeros(int(0.08 * SAMPLE_RATE), dtype=np.float64),
        _tone(523.25, 0.45),
        np.zeros(int(0.20 * SAMPLE_RATE), dtype=np.float64),
    ]
    samples = np.concatenate(parts)
    return _write_wav(path, samples)


class VocalHearingTests(unittest.TestCase):
    def test_synthetic_sung_render_is_measured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _sung_render(os.path.join(tmpdir, "sung.wav"))
            lyrics = [["la", "la"], ["lu", "na"]]
            evidence = perceive_wav(path, lyrics=lyrics)
            measurements = evidence["measurements"]
            vocal = measurements["vocal"]

            self.assertIsNone(evidence["interpretation"])
            self.assertIn("summary", evidence)
            self.assertIn("measurements", evidence)
            self.assertIn("full", evidence)
            self.assertEqual(len(vocal["lines"]), 2)
            self.assertGreaterEqual(vocal["phrase_count"], 2)
            self.assertEqual(len(vocal["phrase_boundaries"]), vocal["phrase_count"])
            self.assertGreaterEqual(len(vocal["breath_gaps"]), 1)
            for gap in vocal["breath_gaps"]:
                self.assertGreater(gap["duration"], 0.0)

            first, second = vocal["lines"]
            self.assertTrue(first["detected_phrase"])
            self.assertTrue(second["detected_phrase"])
            self.assertIsNotNone(first["onset_time"])
            self.assertIsNotNone(second["onset_time"])
            self.assertGreaterEqual(first["syllable_events_detected"], 1)
            self.assertGreaterEqual(second["syllable_events_detected"], 1)
            self.assertEqual(first["stated_syllable_count"], 2)
            self.assertEqual(second["stated_syllable_count"], 2)

            self.assertAlmostEqual(first["median_pitch_hz"], 440.0, delta=12.0)
            self.assertAlmostEqual(second["median_pitch_hz"], 523.25, delta=12.0)
            self.assertEqual(first["note"], "A4")
            self.assertEqual(second["note"], "C5")
            self.assertIsNotNone(first["cents"])
            self.assertLessEqual(abs(first["cents"]), 50.0)
            self.assertIsNotNone(first["pitch_range_hz"])
            self.assertIn("rate_hz", first["vibrato"])
            self.assertIn("extent_cents", first["vibrato"])
            self.assertGreater(first["formants"]["f1"], 0.0)

    def test_vocal_artifact_validation_structure_and_revision(self):
        artifact = {
            "kind": "vocal",
            "lyrics": [["la", "la"], ["lu", "na"]],
        }
        self.assertEqual(validate(artifact), [])
        record = structure_record(artifact)
        self.assertEqual(record["kind"], "vocal")
        self.assertEqual(record["line_count"], 2)
        self.assertEqual(record["syllable_count"], 4)
        self.assertEqual(record["lines"][1]["syllables"], ["lu", "na"])

        revised = apply_revision(
            artifact,
            {"op": "replace_line", "line": 2, "text": ["mi", "ra", "do"]},
        )
        self.assertEqual(revised["lyrics"][1], ["mi", "ra", "do"])
        self.assertEqual(artifact["lyrics"][1], ["lu", "na"])
        self.assertEqual(validate(revised), [])

        with self.assertRaises(ValueError):
            apply_revision(artifact, {"op": "remove_line", "line": 9})
        with self.assertRaises(ValueError):
            apply_revision(
                artifact,
                {"op": "replace_line", "line": 1, "text": []},
            )

    def test_vocal_invariants_use_measured_phrases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _sung_render(os.path.join(tmpdir, "sung.wav"))
            artifact = {
                "kind": "vocal",
                "lyrics": [["la", "la"], ["lu", "na"]],
            }
            evidence = perceive_wav(path, lyrics=artifact["lyrics"])
            block = evaluate_invariants(artifact, {}, evidence)
            ids = [entry["id"] for entry in block["checked"]]
            self.assertIn("vocal_lines_detected", ids)
            self.assertIn("vocal_syllable_counts", ids)
            self.assertIn("vocal_pitch_present", ids)
            lines = [
                entry
                for entry in block["checked"]
                if entry["id"] == "vocal_lines_detected"
            ][0]
            self.assertTrue(lines["held"])
            self.assertEqual(lines["value"], 2)


if __name__ == "__main__":
    unittest.main()
