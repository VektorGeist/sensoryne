"""tests/test_validation_judgments.py - judgments are reported, never gates.

Every application accepts a readable work whose judgments are
unheld, and reports those judgments in its invariants.
"""

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.apps.music import app as music_app
from stage2.apps.music import invariants as music_invariants
from stage2.apps.poetry import app as poetry_app
from stage2.apps.story import app as story_app
from stage2.apps.drawing import app as drawing_app
from stage2.apps.code import app as code_app
from stage2.apps.animation import app as animation_app
from stage2.apps.sound_design import app as sound_app


def _check(block, invariant_id):
    for entry in block["checked"]:
        if entry["id"] == invariant_id:
            return entry
    raise AssertionError("missing invariant %r" % invariant_id)


class MusicJudgmentTests(unittest.TestCase):
    def test_unconventional_score_is_accepted_and_reported(self):
        artifact = {
            "instrument": "piano",
            "parts": [
                {"name": "solo", "gain": 1.0, "notes": [{"start": 0.0, "midi": 72, "vel": 0.9, "dur": 1.0}]},
                {"name": "extra", "gain": 1.0, "notes": [{"start": 30.0, "midi": 60, "vel": 0.4, "dur": 1.0}]},
            ],
        }
        # Accepted: no template judgment gates validation.
        self.assertEqual(music_app.validate(artifact), [])
        block = music_invariants.evaluate_invariants(artifact, {})
        ids = [entry["id"] for entry in block["checked"]]
        for removed in ("total_duration_measured", "effective_velocity_caps", "harmony_at_or_after_8"):
            self.assertNotIn(removed, ids)
        # Reported only through comparison-derived figures:
        # notes heard of written per part, and note counts.
        self.assertEqual(_check(block, "note_count_per_part")["value"], {"solo": 1, "extra": 1})
        heard = _check(block, "notes_heard_of_written")
        self.assertEqual(
            heard["value"],
            [
                {"part": "solo", "notes_written": 1, "notes_heard": 0},
                {"part": "extra", "notes_written": 1, "notes_heard": 0},
            ],
        )

    def test_revision_may_breach_a_judgment(self):
        artifact = {
            "instrument": "piano",
            "parts": [
                {"name": "solo", "gain": 1.0, "notes": [{"start": 0.0, "midi": 72, "vel": 0.4, "dur": 1.0}]},
            ],
        }
        revised = music_app.apply_revision(
            artifact,
            {"op": "add_note", "part": "solo", "start": 0.5, "midi": 74, "vel": 0.9, "dur": 1.0},
        )
        self.assertEqual(len(revised["parts"][0]["notes"]), 2)
        self.assertEqual(music_app.validate(revised), [])
        block = music_invariants.evaluate_invariants(revised, {})
        self.assertNotIn("effective_velocity_caps", [e["id"] for e in block["checked"]])
        self.assertEqual(_check(block, "note_count_per_part")["value"], {"solo": 2})

    def test_plant_faults_aim_at_submitted_work(self):
        artifact = {
            "instrument": "piano",
            "parts": [
                {"name": "upper", "gain": 1.0, "notes": [{"start": 0.0, "midi": 72, "vel": 0.4, "dur": 1.0}, {"start": 6.0, "midi": 74, "vel": 0.4, "dur": 1.0}]},
                {"name": "lower", "gain": 1.0, "notes": [{"start": 0.0, "midi": 36, "vel": 0.4, "dur": 1.0}, {"start": 7.0, "midi": 38, "vel": 0.4, "dur": 1.0}]},
            ],
        }
        faults = music_app.plant_faults(artifact)
        self.assertEqual([f["kind"] for f in faults], ["part_gain", "swell", "drop_note"])
        by_kind = {f["kind"]: f for f in faults}
        self.assertEqual(by_kind["part_gain"]["part"], "lower")
        self.assertEqual(by_kind["drop_note"]["part"], "upper")
        self.assertEqual(by_kind["drop_note"]["start"], 6.0)
        self.assertEqual(by_kind["drop_note"]["midi"], 74)
        total = 8.0
        self.assertGreaterEqual(by_kind["swell"]["from"], 0.0)
        self.assertLessEqual(by_kind["swell"]["to"], total)
        self.assertLess(by_kind["swell"]["from"], by_kind["swell"]["to"])
        text = music_app.generate_ground_truth(faults)
        self.assertEqual(len(text), 3)
        self.assertIn("lower", text[0])
        self.assertIn("upper", text[2])

    def test_revision_part_must_name_a_present_part(self):
        artifact = {
            "instrument": "piano",
            "parts": [
                {"name": "solo", "gain": 1.0, "notes": [{"start": 0.0, "midi": 72, "vel": 0.4, "dur": 1.0}]},
            ],
        }
        with self.assertRaises(ValueError):
            music_app.apply_revision(
                artifact,
                {"op": "set_part_gain", "part": "melody", "value": 0.5},
            )


class PoetryJudgmentTests(unittest.TestCase):
    def test_form_mismatch_is_accepted_and_reported(self):
        artifact = {
            "form": {"line_count": 4, "stanzas": 2},
            "stanzas": [["a cat"]],
        }
        self.assertEqual(poetry_app.validate(artifact), [])
        from stage2.apps.poetry import invariants as poetry_invariants
        block = poetry_invariants.evaluate_invariants(artifact, {})
        declared = _check(block, "declared_form_match")
        self.assertFalse(declared["held"])
        self.assertEqual(declared["value"]["stanza_count"], 1)
        self.assertEqual(declared["value"]["declared_stanzas"], 2)
        self.assertEqual(declared["value"]["line_count"], 1)
        self.assertEqual(declared["value"]["declared_line_count"], 4)

    def test_prompt_does_not_state_refusal(self):
        prompt = poetry_app.composition_prompt({"line_count": 4, "stanzas": 2})
        self.assertNotIn("refused", prompt.lower())
        self.assertNotIn("rejected", prompt.lower())


class StoryJudgmentTests(unittest.TestCase):
    def test_section_count_mismatch_is_accepted_and_reported(self):
        artifact = {
            "objective": {"section_count": 3},
            "sections": [{"title": "Only", "text": "A quiet room."}],
        }
        self.assertEqual(story_app.validate(artifact), [])
        from stage2.apps.story import invariants as story_invariants
        block = story_invariants.evaluate_invariants(artifact, {})
        declared = _check(block, "declared_section_count_match")
        self.assertFalse(declared["held"])
        self.assertEqual(declared["value"]["section_count"], 1)
        self.assertEqual(declared["value"]["declared_section_count"], 3)


class DrawingJudgmentTests(unittest.TestCase):
    def test_missing_viewbox_and_id_are_accepted_and_reported(self):
        artifact = {
            "svg": '<svg xmlns="http://www.w3.org/2000/svg"><rect id="a" x="0" y="0" width="20" height="20"/></svg>',
            "elements": [{"id": "a", "kind": "shape", "description": "first"}, {"id": "gone", "kind": "shape", "description": "missing"}],
        }
        self.assertEqual(drawing_app.validate(artifact), [])
        from stage2.apps.drawing import invariants as drawing_invariants
        block = drawing_invariants.evaluate_invariants(artifact, {})
        self.assertFalse(_check(block, "viewbox_present")["held"])
        declared = _check(block, "declared_element_ids_presence")
        self.assertFalse(declared["held"])
        self.assertEqual(declared["value"], {"a": True, "gone": False})


class CodeJudgmentTests(unittest.TestCase):
    def test_unstated_tests_and_purpose_are_accepted_and_reported(self):
        artifact = {"files": [{"name": "main.py", "text": "print('hi')\n"}]}
        self.assertEqual(code_app.validate(artifact), [])
        from stage2.apps.code import invariants as code_invariants
        block = code_invariants.evaluate_invariants(artifact, {})
        self.assertEqual(_check(block, "test_count_stated")["value"], 0)
        self.assertEqual(_check(block, "tests_with_expected_output")["value"], 0)
        self.assertFalse(_check(block, "purpose_stated")["held"])

    def test_prompt_removed_acceptance_sentence(self):
        prompt = code_app.composition_prompt("print hi")
        self.assertNotIn("at least one test must", prompt.lower())
        self.assertIn("expected", prompt.lower())


class AnimationJudgmentTests(unittest.TestCase):
    def test_outside_duration_and_unordered_are_accepted_and_reported(self):
        artifact = {
            "duration": 2.0,
            "frame_rate": 10,
            "canvas": {"width": 100, "height": 100},
            "elements": [
                {
                    "id": "ball",
                    "keyframes": [
                        {"time": 3.0, "x": 50, "y": 0, "width": 10, "height": 10, "opacity": 1.0, "easing": "linear"},
                        {"time": 0.0, "x": 0, "y": 0, "width": 10, "height": 10, "opacity": 1.0, "easing": "linear"},
                    ],
                }
            ],
        }
        self.assertEqual(animation_app.validate(artifact), [])
        from stage2.apps.animation import invariants as animation_invariants
        block = animation_invariants.evaluate_invariants(artifact, {})
        self.assertFalse(_check(block, "keyframes_inside_duration")["held"])
        self.assertFalse(_check(block, "keyframe_time_order")["held"])


class SoundDesignJudgmentTests(unittest.TestCase):
    def test_event_past_duration_is_accepted_and_reported(self):
        artifact = {
            "duration": 2.0,
            "events": [
                {"id": "hit", "kind": "percussive hit", "start": 1.5, "duration": 1.0, "gain": 0.8, "filter_cutoff": 180, "bandwidth": 0},
            ],
        }
        self.assertEqual(sound_app.validate(artifact), [])
        from stage2.apps.sound_design import invariants as sound_invariants
        block = sound_invariants.evaluate_invariants(
            artifact, {"delivered_events": copy.deepcopy(artifact["events"])},
        )
        self.assertFalse(_check(block, "events_within_duration")["held"])


if __name__ == "__main__":
    unittest.main()
