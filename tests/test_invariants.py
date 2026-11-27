"""tests/test_invariants.py - invariant block tests.

Exercises each application's declared invariants on a valid
artifact and on a faulted artifact.  Invariants are evaluated
from the artifact, render record, and evidence only.
"""

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.apps.music import invariants as music_invariants
from stage2.apps.poetry import invariants as poetry_invariants
from stage2.apps.story import invariants as story_invariants
from stage2.apps.drawing import invariants as drawing_invariants
from stage2.apps.code import invariants as code_invariants
from stage2.apps.typography import invariants as typography_invariants
from stage2.apps.animation import invariants as animation_invariants
from stage2.apps.sound_design import invariants as sound_design_invariants


def _check(block, invariant_id):
    for entry in block["checked"]:
        if entry["id"] == invariant_id:
            return entry
    raise AssertionError("missing invariant %r in %r" % (invariant_id, block))


def _assert_summary(testcase, block):
    summary = block["summary"]
    testcase.assertEqual(
        summary["total_checked"],
        summary["held"] + summary["broken"],
    )
    testcase.assertEqual(summary["total_checked"], len(block["checked"]))


def _music_score(velocity=0.4):
    return {
        "instrument": "piano",
        "parts": [
            {
                "name": "melody",
                "gain": 1.0,
                "notes": [
                    {"start": 0.0, "midi": 72, "vel": velocity, "dur": 0.9},
                    {"start": 16.0, "midi": 74, "vel": 0.4, "dur": 1.0},
                ],
            },
            {
                "name": "harmony",
                "gain": 1.0,
                "notes": [
                    {"start": 0.0, "midi": 60, "vel": 0.3, "dur": 0.9},
                    {"start": 8.0, "midi": 62, "vel": 0.3, "dur": 0.9},
                ],
            },
            {
                "name": "bass",
                "gain": 1.0,
                "notes": [
                    {"start": 0.0, "midi": 36, "vel": 0.3, "dur": 0.9},
                ],
            },
        ],
    }


REMOVED_MUSIC_IDS = [
    "total_duration_measured",
    "effective_velocity_caps",
    "harmony_at_or_after_8",
]

REMOVED_STORY_IDS = [
    "section_shares",
]


class MusicInvariantTests(unittest.TestCase):
    def test_valid_score_holds(self):
        block = music_invariants.evaluate_invariants(_music_score(), {})
        _assert_summary(self, block)
        # The removed template judgments appear in no music
        # evidence document.
        ids = [entry["id"] for entry in block["checked"]]
        for removed in REMOVED_MUSIC_IDS:
            self.assertNotIn(removed, ids)
        self.assertTrue(_check(block, "notes_in_instrument_range")["held"])
        counts = _check(block, "note_count_per_part")["value"]
        self.assertEqual(counts, {"melody": 2, "harmony": 2, "bass": 1})
        # With no comparison heard, notes heard of written is
        # reported per part against the notes written.
        heard = _check(block, "notes_heard_of_written")
        self.assertFalse(heard["held"])
        self.assertEqual(
            heard["value"],
            [
                {"part": "melody", "notes_written": 2, "notes_heard": 0},
                {"part": "harmony", "notes_written": 2, "notes_heard": 0},
                {"part": "bass", "notes_written": 1, "notes_heard": 0},
            ],
        )

    def test_faulted_score_breaks_caps_and_duration(self):
        # The template caps and duration judgments are gone:
        # a loud, short score is accepted and reported only
        # through the comparison-derived coverage.
        artifact = _music_score(velocity=0.9)
        artifact["parts"][0]["notes"] = artifact["parts"][0]["notes"][:1]
        artifact["parts"][1]["notes"] = artifact["parts"][1]["notes"][:1]
        block = music_invariants.evaluate_invariants(artifact, {})
        _assert_summary(self, block)
        ids = [entry["id"] for entry in block["checked"]]
        for removed in REMOVED_MUSIC_IDS:
            self.assertNotIn(removed, ids)
        self.assertTrue(_check(block, "notes_in_instrument_range")["held"])
        counts = _check(block, "note_count_per_part")["value"]
        self.assertEqual(counts, {"melody": 1, "harmony": 1, "bass": 1})

    def test_out_of_range_note_is_named(self):
        artifact = _music_score()
        artifact["parts"][0]["notes"][0]["midi"] = 120
        block = music_invariants.evaluate_invariants(artifact, {})
        entry = _check(block, "notes_in_instrument_range")
        self.assertFalse(entry["held"])
        self.assertEqual(entry["violations"][0]["midi"], 120)


class PoetryInvariantTests(unittest.TestCase):
    def test_valid_poem_holds(self):
        artifact = {
            "form": {"line_count": 2, "stanzas": 1, "syllables_per_line": 2},
            "stanzas": [["a cat", "the sun"]],
        }
        block = poetry_invariants.evaluate_invariants(artifact, {})
        _assert_summary(self, block)
        self.assertTrue(_check(block, "line_count")["held"])
        self.assertTrue(_check(block, "stanza_count")["held"])
        self.assertTrue(_check(block, "declared_form_match")["held"])
        self.assertTrue(_check(block, "syllable_estimates_match_target")["held"])

    def test_faulted_poem_breaks_counts_and_syllables(self):
        artifact = {
            "form": {"line_count": 2, "stanzas": 1, "syllables_per_line": 2},
            "stanzas": [["a cat", "elephant"], ["extra line"]],
        }
        block = poetry_invariants.evaluate_invariants(artifact, {})
        _assert_summary(self, block)
        self.assertFalse(_check(block, "line_count")["held"])
        self.assertFalse(_check(block, "stanza_count")["held"])
        declared = _check(block, "declared_form_match")
        self.assertFalse(declared["held"])
        self.assertEqual(declared["value"]["stanza_count"], 2)
        self.assertEqual(declared["value"]["declared_stanzas"], 1)
        self.assertEqual(declared["value"]["line_count"], 3)
        self.assertEqual(declared["value"]["declared_line_count"], 2)
        syllables = _check(block, "syllable_estimates_match_target")
        self.assertFalse(syllables["held"])
        self.assertEqual(syllables["violations"][0]["line"], 2)


class StoryInvariantTests(unittest.TestCase):
    def test_valid_story_holds(self):
        artifact = {
            "objective": {
                "section_count": 2,
                "movements": ["opening", "ending"],
                "promised_events": ["door", "key"],
            },
            "sections": [
                {"title": "Start", "text": "The door opened slowly."},
                {"title": "End", "text": "She turned the key."},
            ],
        }
        block = story_invariants.evaluate_invariants(artifact, {})
        _assert_summary(self, block)
        self.assertTrue(_check(block, "section_count")["held"])
        self.assertTrue(_check(block, "declared_section_count_match")["held"])
        # The section_shares tautology is removed as a check:
        # its honest content lives on as the section_share_off
        # comparison finding.
        ids = [entry["id"] for entry in block["checked"]]
        for removed in REMOVED_STORY_IDS:
            self.assertNotIn(removed, ids)
        self.assertTrue(_check(block, "promised_events_present")["held"])

    def test_faulted_story_breaks_count_and_events(self):
        artifact = {
            "objective": {
                "section_count": 2,
                "movements": ["opening", "ending"],
                "promised_events": ["door", "key"],
            },
            "sections": [
                {"title": "Only", "text": "A quiet room."},
            ],
        }
        block = story_invariants.evaluate_invariants(artifact, {})
        _assert_summary(self, block)
        self.assertFalse(_check(block, "section_count")["held"])
        declared = _check(block, "declared_section_count_match")
        self.assertFalse(declared["held"])
        self.assertEqual(declared["value"]["section_count"], 1)
        self.assertEqual(declared["value"]["declared_section_count"], 2)
        ids = [entry["id"] for entry in block["checked"]]
        for removed in REMOVED_STORY_IDS:
            self.assertNotIn(removed, ids)
        events = _check(block, "promised_events_present")
        self.assertFalse(events["held"])
        self.assertEqual(events["violations"], ["door", "key"])


class DrawingInvariantTests(unittest.TestCase):
    def _artifact(self, svg, elements, rules=None):
        artifact = {
            "svg": svg,
            "elements": elements,
        }
        if rules is not None:
            artifact["overlap_rules"] = rules
        return artifact

    def test_valid_drawing_holds(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<rect id="a" x="0" y="0" width="50" height="50"/>'
            '<rect id="b" x="25" y="25" width="50" height="50"/>'
            "</svg>"
        )
        artifact = self._artifact(
            svg,
            [
                {"id": "a", "kind": "shape", "description": "first"},
                {"id": "b", "kind": "shape", "description": "second"},
            ],
            [{"a": "a", "b": "b"}],
        )
        block = drawing_invariants.evaluate_invariants(artifact, {})
        _assert_summary(self, block)
        self.assertTrue(_check(block, "required_elements_present")["held"])
        viewbox = _check(block, "viewbox_present")
        self.assertTrue(viewbox["held"])
        self.assertEqual(viewbox["value"], "0 0 100 100")
        declared = _check(block, "declared_element_ids_presence")
        self.assertTrue(declared["held"])
        self.assertEqual(declared["value"], {"a": True, "b": True})
        self.assertTrue(_check(block, "declared_overlap_rules")["held"])

    def test_faulted_drawing_breaks_required_and_overlap(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<rect id="a" x="0" y="0" width="20" height="20"/>'
            '<rect id="b" x="60" y="60" width="20" height="20"/>'
            "</svg>"
        )
        artifact = self._artifact(
            svg,
            [
                {"id": "a", "kind": "shape", "description": "first"},
                {"id": "b", "kind": "shape", "description": "second"},
                {"id": "c", "kind": "text", "description": "missing"},
            ],
            [{"a": "a", "b": "b"}],
        )
        block = drawing_invariants.evaluate_invariants(artifact, {})
        _assert_summary(self, block)
        required = _check(block, "required_elements_present")
        self.assertFalse(required["held"])
        self.assertEqual(required["violations"], ["c"])
        declared = _check(block, "declared_element_ids_presence")
        self.assertFalse(declared["held"])
        self.assertEqual(declared["value"], {"a": True, "b": True, "c": False})
        self.assertFalse(_check(block, "declared_overlap_rules")["held"])

    def test_missing_viewbox_is_reported_not_rejected(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect id="a" x="0" y="0" width="20" height="20"/>'
            "</svg>"
        )
        artifact = self._artifact(
            svg,
            [{"id": "a", "kind": "shape", "description": "first"}],
        )
        block = drawing_invariants.evaluate_invariants(artifact, {})
        _assert_summary(self, block)
        viewbox = _check(block, "viewbox_present")
        self.assertFalse(viewbox["held"])
        self.assertIsNone(viewbox["value"])
        self.assertEqual(viewbox["violations"], [{"viewBox": None}])


class CodeInvariantTests(unittest.TestCase):
    def _artifact(self):
        return {
            "files": [{"name": "main.py", "text": "print('hi')\n"}],
            "purpose": "print hi",
            "tests": [{"name": "t", "input": "", "expected": "hi\n"}],
        }

    def test_valid_program_holds(self):
        artifact = self._artifact()
        block = code_invariants.evaluate_invariants(
            artifact, {"original_artifact": copy.deepcopy(artifact)}
        )
        _assert_summary(self, block)
        self.assertTrue(_check(block, "tests_still_defined")["held"])
        self.assertTrue(_check(block, "test_count_stated")["held"])
        self.assertEqual(_check(block, "test_count_stated")["value"], 1)
        self.assertTrue(_check(block, "tests_with_expected_output")["held"])
        self.assertEqual(_check(block, "tests_with_expected_output")["value"], 1)
        self.assertTrue(_check(block, "purpose_stated")["held"])
        self.assertTrue(_check(block, "purpose_stated")["value"])
        self.assertTrue(_check(block, "purpose_unchanged")["held"])
        self.assertTrue(_check(block, "entry_point_present")["held"])

    def test_faulted_program_breaks_tests_and_purpose(self):
        original = self._artifact()
        artifact = copy.deepcopy(original)
        artifact["purpose"] = "print bye"
        artifact["tests"] = [{"name": "t", "input": ""}]
        block = code_invariants.evaluate_invariants(
            artifact, {"original_artifact": original}
        )
        _assert_summary(self, block)
        self.assertFalse(_check(block, "tests_still_defined")["held"])
        measured = _check(block, "tests_with_expected_output")
        # The judgment "at least one stated test states an expected
        # output" does not hold when the only test states none.
        self.assertFalse(measured["held"])
        self.assertEqual(measured["value"], 0)
        purpose = _check(block, "purpose_unchanged")
        self.assertFalse(purpose["held"])
        self.assertEqual(purpose["violations"][0]["original"], "print hi")

    def test_unstated_tests_and_purpose_are_reported(self):
        artifact = {
            "files": [{"name": "main.py", "text": "print('hi')\n"}],
        }
        block = code_invariants.evaluate_invariants(artifact, {})
        _assert_summary(self, block)
        self.assertEqual(_check(block, "test_count_stated")["value"], 0)
        self.assertEqual(_check(block, "tests_with_expected_output")["value"], 0)
        stated = _check(block, "purpose_stated")
        self.assertFalse(stated["held"])
        self.assertIsNone(stated["value"])


class TypographyInvariantTests(unittest.TestCase):
    def _artifact(self, heading_size=32, x=50):
        return {
            "page": {"width": 200, "height": 200},
            "elements": [
                {"id": "title", "kind": "heading", "text": "Title", "font_size": heading_size, "x": x, "y": 10, "width": 100, "height": 40, "alignment": "left"},
                {"id": "body", "kind": "body", "text": "Body", "font_size": 16, "x": 50, "y": 60, "width": 100, "height": 40, "alignment": "left"},
                {"id": "cap", "kind": "caption", "text": "Cap", "font_size": 12, "x": 50, "y": 110, "width": 100, "height": 20, "alignment": "left"},
            ],
        }

    def test_valid_page_holds(self):
        block = typography_invariants.evaluate_invariants(self._artifact(), {})
        _assert_summary(self, block)
        self.assertTrue(_check(block, "elements_inside_page")["held"])
        self.assertTrue(_check(block, "heading_larger_than_body")["held"])
        self.assertTrue(_check(block, "body_larger_than_caption")["held"])
        self.assertTrue(_check(block, "shared_alignment_edge")["held"])

    def test_faulted_page_breaks_hierarchy_and_bounds(self):
        block = typography_invariants.evaluate_invariants(
            self._artifact(heading_size=10, x=150), {}
        )
        _assert_summary(self, block)
        self.assertFalse(_check(block, "heading_larger_than_body")["held"])
        self.assertFalse(_check(block, "elements_inside_page")["held"])


class AnimationInvariantTests(unittest.TestCase):
    def _artifact(self, easing="linear", time=2.0):
        return {
            "duration": 2.0,
            "frame_rate": 10,
            "canvas": {"width": 100, "height": 100},
            "loop": True,
            "elements": [
                {
                    "id": "ball",
                    "keyframes": [
                        {"time": 0.0, "x": 0, "y": 0, "width": 10, "height": 10, "opacity": 1.0, "easing": easing},
                        {"time": time, "x": 50, "y": 0, "width": 10, "height": 10, "opacity": 1.0, "easing": easing},
                    ],
                }
            ],
        }

    def test_valid_scene_holds(self):
        evidence = {"measurements": {"final_first_difference": 0.0}}
        block = animation_invariants.evaluate_invariants(
            self._artifact(), {"frame_count": 20}, evidence
        )
        _assert_summary(self, block)
        self.assertTrue(_check(block, "frame_count_matches")["held"])
        self.assertTrue(_check(block, "keyframes_inside_duration")["held"])
        self.assertTrue(_check(block, "keyframe_time_order")["held"])
        self.assertTrue(_check(block, "easings_named")["held"])
        self.assertTrue(_check(block, "loop_final_matches_first")["held"])

    def test_faulted_scene_breaks_easing_time_and_loop(self):
        evidence = {"measurements": {"final_first_difference": 1.5}}
        block = animation_invariants.evaluate_invariants(
            self._artifact(easing="bounce", time=3.0),
            {"frame_count": 19},
            evidence,
        )
        _assert_summary(self, block)
        self.assertFalse(_check(block, "frame_count_matches")["held"])
        self.assertFalse(_check(block, "keyframes_inside_duration")["held"])
        self.assertFalse(_check(block, "easings_named")["held"])
        self.assertFalse(_check(block, "loop_final_matches_first")["held"])

    def test_out_of_order_keyframes_are_reported(self):
        artifact = self._artifact()
        track = artifact["elements"][0]["keyframes"]
        artifact["elements"][0]["keyframes"] = [track[1], track[0]]
        block = animation_invariants.evaluate_invariants(artifact, {})
        _assert_summary(self, block)
        order = _check(block, "keyframe_time_order")
        self.assertFalse(order["held"])
        self.assertEqual(order["violations"][0]["times"], [2.0, 0.0])


class SoundDesignInvariantTests(unittest.TestCase):
    def _artifact(self):
        return {
            "duration": 5.0,
            "events": [
                {"id": "hit", "kind": "percussive hit", "start": 0.0, "duration": 0.5, "gain": 0.8},
                {"id": "wash", "kind": "filtered noise", "start": 1.0, "duration": 1.0, "gain": 0.4},
            ],
        }

    def _evidence(self, events):
        return {
            "measurements": {
                "events": events,
                "silence_share": 0.2,
            }
        }

    def test_valid_work_holds(self):
        artifact = self._artifact()
        evidence = self._evidence([
            {"id": "hit", "onset_time": 0.0, "realized_duration": 0.5},
            {"id": "wash", "onset_time": 1.0, "realized_duration": 1.0},
        ])
        block = sound_design_invariants.evaluate_invariants(
            artifact, {"delivered_events": copy.deepcopy(artifact["events"])}, evidence
        )
        _assert_summary(self, block)
        self.assertTrue(_check(block, "delivered_event_count_matches")["held"])
        self.assertTrue(_check(block, "event_ids_present")["held"])
        self.assertTrue(_check(block, "events_within_duration")["held"])
        self.assertTrue(_check(block, "realized_onsets_present")["held"])
        self.assertTrue(_check(block, "realized_durations_positive")["held"])
        self.assertTrue(_check(block, "silence_share_bounded")["held"])

    def test_faulted_work_breaks_delivery_and_onsets(self):
        artifact = self._artifact()
        evidence = self._evidence([
            {"id": "hit", "onset_time": 2.0, "realized_duration": 0.0},
        ])
        block = sound_design_invariants.evaluate_invariants(
            artifact,
            {"delivered_events": [copy.deepcopy(artifact["events"][0])]},
            evidence,
        )
        _assert_summary(self, block)
        self.assertFalse(_check(block, "delivered_event_count_matches")["held"])
        self.assertFalse(_check(block, "event_ids_present")["held"])
        self.assertFalse(_check(block, "realized_onsets_present")["held"])
        self.assertFalse(_check(block, "realized_durations_positive")["held"])


if __name__ == "__main__":
    unittest.main()
