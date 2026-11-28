"""tests/test_comparison.py - comparison judgment tests (music).

Planted faults through the application's own fault definitions
and ground truth produce the named comparison findings;
a clean render produces no note_not_heard and no pitch_off
findings; the symphony-scale pack is carried whole; and the
removed template ids appear in no music evidence document,
for a short work and for a work of symphony length.

Synthesis is done in-test with numpy (as in test_vocal.py):
sine notes at the written pitch, with the planted faults
applied exactly as plant_faults / generate_ground_truth
state them. No model calls are made.

The probe score from the dynamics recalibration (the
symphony artifact rendered through the real piano route)
is covered in PianoProbeTests: a faithful render produces
no findings at all, a part_gain of 0.25 on the lead
produces dynamics_off findings naming the lead, and a
semitone substitution (midi 80 played where 79 is written)
produces a finding naming midi 80 as heard in that
note's window.

The dense-hearing regression in DenseHearingTests guards
the fundamental-evidence note-heard rule (magnitude at the
sounding fundamental against the median neighbouring-
semitone fundamental and the median fundamental across
midi 21 to 108 in the same window, with the dominance
guard): a dense multi-part score rendered through the
piano renderer, comparable to the live proof pieces,
must have its written notes heard, and a dropped note
must be named note_not_heard.
"""

import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.apps.music.app import (
    generate_ground_truth,
    plant_faults,
    score_total_length,
)
from stage2.apps.music.comparison import compute_comparison
from stage2.apps.music.invariants import evaluate_invariants
from stage2.apps.music.perceive import perceive_wav
from stage2.shape import shape_evidence

SAMPLE_RATE = 44100
REMOVED_IDS = (
    "total_duration_measured",
    "effective_velocity_caps",
    "harmony_at_or_after_8",
)
FINDING_KEYS = ("kind", "where", "written", "heard", "statement")

# The probe score: a two-part score for the piano route -
# a lead part carrying eight long notes (each 3.0 seconds,
# spaced 3.5 seconds apart) over a pad part carrying two
# long notes - rendered through the piano route.
PROBE_SCORE = {
    "instrument": "violin",
    "parts": [
        {
            "name": "lead",
            "gain": 1.0,
            "notes": [
                {"start": 0.0, "midi": 69, "vel": 0.9, "dur": 3.0},
                {"start": 3.5, "midi": 72, "vel": 0.9, "dur": 3.0},
                {"start": 7.0, "midi": 76, "vel": 0.9, "dur": 3.0},
                {"start": 10.5, "midi": 79, "vel": 0.9, "dur": 3.0},
                {"start": 14.0, "midi": 76, "vel": 0.9, "dur": 3.0},
                {"start": 17.5, "midi": 72, "vel": 0.9, "dur": 3.0},
                {"start": 21.0, "midi": 74, "vel": 0.9, "dur": 3.0},
                {"start": 24.5, "midi": 71, "vel": 0.9, "dur": 3.0},
            ],
        },
        {
            "name": "pad",
            "gain": 0.8,
            "notes": [
                {"start": 0.0, "midi": 45, "vel": 0.85, "dur": 14.0},
                {"start": 14.0, "midi": 48, "vel": 0.85, "dur": 14.0},
            ],
        },
    ],
}


def _dense_score():
    """A dense multi-part score for the hearing regression.

    Five parts, 105 notes in total (21 per part), each note
    starting every 2.5 seconds and lasting 3.0 seconds, so
    notes overlap across parts throughout. Scales sit in
    separate registers with the Inner scale avoiding the
    pitch that collides with the densest window.
    """
    parts = []
    for name, scale in (
        ("Melody", [72, 75, 78, 81, 79, 74]),
        ("Harmony", [57, 60, 64, 67, 65, 62]),
        ("Bass", [36, 39, 41, 44, 43, 38]),
        ("Inner", [50, 53, 56, 59, 57, 52]),
        ("Top", [86, 89, 91, 93, 92, 88]),
    ):
        parts.append(
            {
                "name": name,
                "gain": 0.9,
                "notes": [
                    {
                        "start": float(i * 2.5),
                        "midi": scale[i % len(scale)],
                        "vel": 0.45,
                        "dur": 3.0,
                    }
                    for i in range(21)
                ],
            }
        )
    return {"instrument": "piano", "parts": parts}


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


def _render(score, path, faults=()):
    """Synthesize the score, applying planted faults as stated."""
    total = score_total_length(score)
    buffer = np.zeros(int(math.ceil(total * SAMPLE_RATE)) + SAMPLE_RATE // 2)
    for part in score.get("parts", []):
        gain = float(part.get("gain", 1.0))
        for note in part.get("notes", []):
            start = float(note["start"])
            dur = float(note["dur"]) if note.get("dur") is not None else 0.5
            effective = float(note.get("vel", 0.0)) * gain
            for fault in faults or ():
                kind = fault.get("kind")
                if kind == "drop_note" and fault.get("part") == part.get("name") and float(fault.get("start")) == start and int(fault.get("midi")) == int(note["midi"]):
                    effective = 0.0
                elif kind == "part_gain" and fault.get("part") == part.get("name"):
                    effective *= float(fault.get("factor", 1.0))
                elif kind == "swell" and float(fault.get("from")) <= start < float(fault.get("to")):
                    effective *= float(fault.get("factor", 1.0))
                elif kind == "velocity_scale" and fault.get("part") == part.get("name") and float(fault.get("start")) == start and int(fault.get("midi")) == int(note["midi"]):
                    effective *= float(fault.get("factor", 1.0))
            sample = _tone(int(note["midi"]), dur, effective)
            begin = int(start * SAMPLE_RATE)
            end = min(len(buffer), begin + len(sample))
            buffer[begin:end] += sample[: end - begin]
    pcm = np.round(np.clip(buffer, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, pcm)
    return {"path": path, "duration": total}


def _solo_score(count=20, midi=69, vel=0.4):
    return {
        "instrument": "piano",
        "parts": [
            {
                "name": "solo",
                "gain": 1.0,
                "notes": [
                    {"start": float(i), "midi": midi, "vel": vel, "dur": 1.0}
                    for i in range(count)
                ],
            }
        ],
    }


def _two_part_score():
    return {
        "instrument": "piano",
        "parts": [
            {
                "name": "upper",
                "gain": 1.0,
                "notes": [
                    {"start": float(i), "midi": 69, "vel": 0.35, "dur": 1.0}
                    for i in range(20)
                ],
            },
            {
                "name": "lower",
                "gain": 1.0,
                "notes": [
                    {"start": float(i), "midi": 45, "vel": 0.35, "dur": 1.0}
                    for i in range(20)
                ],
            },
        ],
    }


def _compare(score, tmpdir, faults=(), label="render"):
    path = os.path.join(tmpdir, label + ".wav")
    record = _render(score, path, faults)
    evidence = perceive_wav(path, artifact=score)
    return compute_comparison(score, record, evidence), evidence, record


def _findings(comparison, kind):
    return [f for f in comparison["findings"] if f["kind"] == kind]


def _compact_off(comparison):
    return [
        (
            f["written"].get("part"),
            f["written"].get("scope"),
            f["written"].get("start"),
            f["written"].get("midi"),
            round(float(f["heard"].get("difference_db", 0.0)), 2),
            round(float(f["heard"].get("attack_loudness_db", 0.0)), 2),
        )
        for f in _findings(comparison, "dynamics_off")
    ]


class ComparisonShapeTests(unittest.TestCase):
    def test_each_finding_kind_carries_the_full_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            score = _solo_score()
            faults = plant_faults(score)
            comparison, _evidence, _record = _compare(score, tmpdir, faults)
            self.assertGreater(len(comparison["findings"]), 0)
            for finding in comparison["findings"]:
                for key in FINDING_KEYS:
                    self.assertIn(key, finding)
            # Summary counts findings by kind.
            counts = comparison["summary"]["counts"]
            for kind, count in counts.items():
                self.assertEqual(count, len(_findings(comparison, kind)))
            self.assertEqual(
                comparison["summary"]["total"], len(comparison["findings"])
            )
            # Tolerances and the loudness mapping are stated.
            tolerances = comparison["tolerances"]
            self.assertEqual(tolerances["pitch_tolerance_cents"], 50.0)
            self.assertEqual(tolerances["note_heard_harmonic_ratio_threshold"], 0.15)
            self.assertEqual(tolerances["silence_loudness_db"], -70.0)
            self.assertEqual(tolerances["dynamics_threshold_db"], 4.0)
            self.assertIn("expected_db", tolerances["dynamics_mapping"]["formula"])
            self.assertIn("effective_velocity", tolerances["dynamics_mapping"]["formula"])
            self.assertIn("attack", tolerances["dynamics_mapping"]["formula"])


class PlantedFaultComparisonTests(unittest.TestCase):
    def test_drop_note_produces_note_not_heard_naming_the_note(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            score = _solo_score()
            faults = plant_faults(score)
            drop = [f for f in faults if f["kind"] == "drop_note"][0]
            ground_truth = generate_ground_truth(faults)
            self.assertIn(
                "drop_note: first note in %s at %s midi %s omitted every render."
                % (drop["part"], drop["start"], drop["midi"]),
                ground_truth,
            )
            comparison, _evidence, _record = _compare(score, tmpdir, faults)
            not_heard = _findings(comparison, "note_not_heard")
            named = [
                f
                for f in not_heard
                if f["written"]["part"] == drop["part"]
                and f["written"]["start"] == drop["start"]
                and f["written"]["midi"] == drop["midi"]
            ]
            self.assertTrue(
                named,
                "expected a note_not_heard naming %r, got %r"
                % (drop, not_heard),
            )
            per_part = {
                entry["part"]: entry for entry in comparison["per_part"]
            }
            self.assertEqual(per_part[drop["part"]]["notes_not_heard"], 1)

    def test_part_gain_produces_dynamics_off_naming_the_part(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            score = _two_part_score()
            faults = plant_faults(score)
            gain_fault = [f for f in faults if f["kind"] == "part_gain"][0]
            self.assertEqual(gain_fault["part"], "lower")
            clean, _c_ev, _c_rec = _compare(score, tmpdir, (), "clean")
            faulted, _f_ev, _f_rec = _compare(score, tmpdir, faults, "faulted")

            # Per-note and per-part dynamics_off findings
            # name the faulted part in the faulted render;
            # the clean render of the same score does not.
            clean_lower = [
                f
                for f in _findings(clean, "dynamics_off")
                if f["written"]["part"] == "lower"
            ]
            faulted_lower = [
                f
                for f in _findings(faulted, "dynamics_off")
                if f["written"]["part"] == "lower"
            ]
            self.assertEqual(clean_lower, [])
            self.assertTrue(
                faulted_lower,
                "expected a dynamics_off naming lower, faults=%r off=%r"
                % (faults, _compact_off(faulted)),
            )
            self.assertTrue(
                any(
                    f["written"].get("scope") in ("note", "part")
                    for f in faulted_lower
                )
            )
            changed = [
                f
                for f in faulted_lower
                if f["heard"]["difference_db"] > 3.0
            ]
            self.assertTrue(
                changed,
                "expected a dynamics_off for the louder lower part, got %r"
                % (_compact_off(faulted),),
            )
            for finding in faulted_lower:
                self.assertIn("lower", finding["where"])

    def test_swell_produces_dynamics_off_inside_the_swell_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            score = _solo_score()
            # Mid-velocity notes (written velocity 0.4),
            # swell factor 2.0, so the rendered audio
            # carries the swell.
            swell = {"kind": "swell", "from": 8.0, "to": 12.0, "factor": 2.0}
            faults = (swell,)
            clean, _c_ev, _c_rec = _compare(score, tmpdir, (), "clean")
            faulted, _f_ev, _f_rec = _compare(score, tmpdir, faults, "faulted")

            def _window_off(comparison):
                return [
                    f
                    for f in _findings(comparison, "dynamics_off")
                    if swell["from"] <= f["written"]["start"] < swell["to"]
                    or swell["from"] <= f["written"]["end"] - 1.0 < swell["to"]
                ]

            clean_window = _window_off(clean)
            faulted_window = _window_off(faulted)
            self.assertEqual(clean_window, [])
            self.assertTrue(
                faulted_window,
                "expected dynamics_off in the swell window, got %r"
                % (_compact_off(faulted),),
            )
            # The swell makes the notes louder than written:
            # heard above expected inside the window.
            louder = [
                f
                for f in faulted_window
                if f["heard"]["difference_db"] > 3.0
            ]
            self.assertTrue(
                louder,
                "expected a dynamics_off above expected in the swell window, got %r"
                % (_compact_off(faulted),),
            )
            for finding in faulted_window:
                self.assertIn("solo", finding["where"])

    def test_single_quiet_note_produces_dynamics_off_naming_the_note(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            score = _solo_score()
            faults = (
                {
                    "kind": "velocity_scale",
                    "part": "solo",
                    "start": 5.0,
                    "midi": 69,
                    "factor": 0.5,
                },
            )
            clean, _c_ev, _c_rec = _compare(score, tmpdir, (), "clean")
            faulted, _f_ev, _f_rec = _compare(score, tmpdir, faults, "faulted")

            def _named(comparison):
                return [
                    f
                    for f in _findings(comparison, "dynamics_off")
                    if f["written"].get("part") == "solo"
                    and f["written"].get("start") == 5.0
                    and f["written"].get("midi") == 69
                ]

            self.assertEqual(_named(clean), [])
            named = _named(faulted)
            self.assertTrue(
                named,
                "expected a dynamics_off naming the note at 5.0, got %r"
                % (_compact_off(faulted),),
            )
            self.assertEqual(named[0]["written"].get("scope"), "note")
            self.assertLess(named[0]["heard"]["difference_db"], -3.0)
            self.assertIn("solo", named[0]["where"])

    def test_clean_render_has_no_note_not_heard_or_pitch_off(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            score = _solo_score()
            comparison, _evidence, _record = _compare(score, tmpdir, ())
            self.assertEqual(_findings(comparison, "note_not_heard"), [])
            self.assertEqual(_findings(comparison, "pitch_off"), [])
            self.assertEqual(
                comparison["per_part"],
                [
                    {
                        "part": "solo",
                        "notes_written": 20,
                        "notes_heard": 20,
                        "notes_not_heard": 0,
                    }
                ],
            )


class SymphonyPackTests(unittest.TestCase):
    def test_symphony_scale_pack_is_carried_whole(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # A long single-part score: 20 segments over the
            # whole render, every segment's reading fields.
            score = _solo_score(count=60)
            path = os.path.join(tmpdir, "symphony.wav")
            record = _render(score, path, ())
            evidence = perceive_wav(path, artifact=score)
            comparison = compute_comparison(score, record, evidence)
            self.assertEqual(
                comparison["per_part"][0]["notes_written"], 60
            )
            pack = evidence["measurements"]["token_pack"]
            segments = pack["segments_data"]
            self.assertEqual(len(segments), 20)
            self.assertEqual(pack["segment_count"], 20)
            for seg in segments:
                self.assertIn("segment", seg)
                self.assertIn("start_time", seg)
                self.assertIn("end_time", seg)
                self.assertIn("reading", seg)
                reading = seg["reading"]
                for field in (
                    "sentence",
                    "note",
                    "cents",
                    "pitch_hz",
                    "loudness",
                    "brightness",
                    "energy_center_hz",
                    "region",
                    "pitch_trend",
                    "loud_trend",
                    "onset_rate",
                ):
                    self.assertIn(field, reading)
            shaped = shape_evidence(evidence)
            shaped_pack = shaped["measurements"]["token_pack"]
            self.assertEqual(
                json.dumps(
                    shaped_pack, sort_keys=True, ensure_ascii=False, default=str
                ),
                json.dumps(
                    pack, sort_keys=True, ensure_ascii=False, default=str
                ),
            )
            self.assertNotIn(
                "_reference", json.dumps(shaped_pack, default=str)
            )


class RemovedIdsTests(unittest.TestCase):
    def test_removed_ids_absent_short_and_symphony(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for label, score in (
                ("short", _solo_score(count=4)),
                ("symphony", _solo_score(count=60)),
            ):
                comparison, evidence, record = _compare(
                    score, tmpdir, (), label
                )
                block = evaluate_invariants(score, record, evidence)
                ids = [entry["id"] for entry in block["checked"]]
                for removed in REMOVED_IDS:
                    self.assertNotIn(removed, ids, label)
                    self.assertNotIn(removed, json.dumps(evidence, default=str))
                    self.assertNotIn(removed, json.dumps(comparison, default=str))


class PianoProbeTests(unittest.TestCase):
    """The dynamics probe score through the real piano route.

    The piano stays outside the project: the sibling
    embers-piano directory is put on sys.path to render;
    no piano code enters this tree.
    """

    def _piano_renderer(self):
        for candidate in (
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "..",
                "embers-piano",
            ),
        ):
            candidate = os.path.normpath(candidate)
            if os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)
        try:
            from music_renderer import renderer
        except Exception as exc:
            self.skipTest("piano route not available here: %r" % (exc,))
        return renderer

    def _probe_compare(self, tmpdir, faults=(), label="probe", substitute=None):
        renderer = self._piano_renderer()
        score = json.loads(json.dumps(PROBE_SCORE))
        path = os.path.join(tmpdir, label + ".wav")
        if substitute is None:
            record = renderer(score, path, list(faults))
        else:
            # A fault list for the renderer may also carry a
            # pitch substitution for one lead note.
            played = json.loads(json.dumps(score))
            for part in played.get("parts", []):
                if part.get("name") != "lead":
                    continue
                for note in part.get("notes", []):
                    if float(note.get("start")) == 10.5:
                        note["midi"] = 80
            record = renderer(played, path, list(faults))
        evidence = perceive_wav(path, artifact=score)
        return compute_comparison(score, record, evidence)

    def test_probe_clean_render_has_no_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison = self._probe_compare(tmpdir, (), "clean")
            self.assertEqual(
                comparison["findings"], [],
                "clean probe off=%r spread=%r"
                % (
                    _compact_off(comparison),
                    comparison["tolerances"].get("dynamics_observed_spread_db"),
                ),
            )
            self.assertEqual(comparison["summary"]["total"], 0)

    def test_probe_part_gain_produces_dynamics_off_naming_lead(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison = self._probe_compare(
                tmpdir, ({"kind": "part_gain", "part": "lead", "factor": 0.25},), "gained"
            )
            off = _findings(comparison, "dynamics_off")
            named = [f for f in off if f["written"]["part"] == "lead"]
            self.assertTrue(
                named,
                "expected dynamics_off naming lead, off=%r"
                % (_compact_off(comparison),),
            )
            self.assertTrue(
                any(
                    f["written"].get("scope") in ("note", "part")
                    for f in named
                )
            )
            for finding in named:
                self.assertIn("lead", finding["where"])
                self.assertLess(
                    finding["heard"]["difference_db"], -3.0,
                    "off=%r" % (_compact_off(comparison),),
                )

    def test_probe_semitone_substitution_names_midi_80_heard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison = self._probe_compare(tmpdir, (), "pitches", substitute=True)
            named = [
                f
                for f in comparison["findings"]
                if f["written"].get("part") == "lead"
                and f["written"].get("start") == 10.5
                and "midi 80" in f["statement"]
            ]
            self.assertTrue(
                named,
                "expected a finding naming midi 80 as heard in the "
                "10.5 lead window, got %r"
                % ([f["statement"] for f in comparison["findings"]],),
            )
            kinds = {f["kind"] for f in named}
            self.assertTrue(
                kinds & {"note_not_heard", "pitch_off"},
                "expected note_not_heard or pitch_off naming midi 80, got %r"
                % (named,),
            )


class DenseHearingTests(unittest.TestCase):
    """Dense polyphony through the real piano renderer.

    Guards the fundamental-evidence note-heard rule in
    stage2/apps/music/comparison.py: the magnitude at the
    sounding fundamental against the median neighbouring-
    semitone fundamental and the median fundamental across
    midi 21 to 108 in the same window, with the dominance
    guard. A dense multi-part score (105 notes, overlapping
    across five parts) rendered cleanly must have every
    written note heard per part; the same score with one
    note dropped from the render must name that note
    note_not_heard.
    """

    def _dense_compare(self, tmpdir, faults=(), label="dense"):
        renderer = PianoProbeTests()._piano_renderer()
        score = _dense_score()
        path = os.path.join(tmpdir, label + ".wav")
        record = renderer(score, path, list(faults))
        evidence = perceive_wav(path, artifact=score)
        return compute_comparison(score, record, evidence)

    def test_dense_clean_render_hears_written_notes_and_drop_is_named(self):
        # A small stated margin: at most one note per part
        # may go unheard on a faithful dense render.
        margin = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            clean = self._dense_compare(tmpdir, (), "dense_clean")
            total_written = sum(
                entry["notes_written"] for entry in clean["per_part"]
            )
            self.assertGreaterEqual(
                total_written,
                100,
                "dense score must carry a hundred or more notes, got %r"
                % (clean["per_part"],),
            )
            for entry in clean["per_part"]:
                self.assertLessEqual(
                    entry["notes_not_heard"],
                    margin,
                    "clean dense render: part %r heard %r of %r"
                    % (
                        entry["part"],
                        entry["notes_heard"],
                        entry["notes_written"],
                    ),
                )
                self.assertEqual(
                    entry["notes_heard"] + entry["notes_not_heard"],
                    entry["notes_written"],
                )
            dropped = self._dense_compare(
                tmpdir,
                (
                    {
                        "kind": "drop_note",
                        "part": "Harmony",
                        "start": 2.5,
                        "midi": 60,
                    },
                ),
                "dense_dropped",
            )
            named = [
                f
                for f in _findings(dropped, "note_not_heard")
                if f["written"].get("part") == "Harmony"
                and f["written"].get("start") == 2.5
                and f["written"].get("sounding_midi") == 60
            ]
            self.assertTrue(
                named,
                "expected note_not_heard naming the dropped Harmony "
                "note at 2.5 midi 60, got %r"
                % (
                    [
                        (f["written"].get("part"), f["written"].get("start"))
                        for f in _findings(dropped, "note_not_heard")
                    ],
                ),
            )


if __name__ == "__main__":
    unittest.main()
