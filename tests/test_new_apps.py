"""tests/test_new_apps.py - new application tests.

For typography, animation, and sound design: validation, a
rendered and perceived clean artifact, each planted fault
detected in measurements or invariants, and one addressed
revision applied. No model calls are made.

The animation instrument samples the last rendered frame at
or before a stated time. Frame times span 0 to the duration
inclusive, so no frame sits at exactly 1.0 seconds and the
clean sampled position at 1.0 is the realized state of the
frame at 18/19 seconds. The animation drop_keyframe fault
removes the final keyframe, so the faulted track ends at the
interior keyframe and the sampled position at the end differs
from the written final position. The animation time-shift
fault moves an interior keyframe, so it is detected at an
interior sampled time, not at the start. The typography
margin fault places an element in the page margin band inside
the page bounds, so it is detected in the margin measurements
and the alignment consistency reading.
"""

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.apps.typography import invariants as typography_invariants
from stage2.apps.typography.app import (
    apply_revision as typography_revision,
    plant_faults as typography_faults,
    perceive as typography_perceive,
    structure_record as typography_structure,
    validate as typography_validate,
)
from stage2.apps.animation import invariants as animation_invariants
from stage2.apps.animation.app import (
    apply_revision as animation_revision,
    plant_faults as animation_faults,
    perceive as animation_perceive,
    structure_record as animation_structure,
    validate as animation_validate,
)
from stage2.apps.sound_design import invariants as sound_invariants
from stage2.apps.sound_design.app import (
    apply_revision as sound_revision,
    plant_faults as sound_faults,
    perceive as sound_perceive,
    structure_record as sound_structure,
    validate as sound_validate,
)
from typography_renderer import renderer as typography_renderer
from animation_renderer import renderer as animation_renderer
from sound_design_renderer import renderer as sound_renderer


def _check(block, invariant_id):
    for entry in block["checked"]:
        if entry["id"] == invariant_id:
            return entry
    raise AssertionError("missing invariant %r" % invariant_id)


def _point_at(element, time):
    for point in element["sampled_positions"]:
        if point["time"] == time:
            return point
    raise AssertionError("no sampled position at time %r" % (time,))


class TypographyTests(unittest.TestCase):
    def _artifact(self):
        return {
            "page": {"width": 400, "height": 500, "background": "#ffffff"},
            "elements": [
                {"id": "title", "kind": "heading", "text": "Title", "font_size": 32, "x": 40, "y": 30, "width": 300, "height": 50, "alignment": "left"},
                {"id": "body", "kind": "body", "text": "Body text.", "font_size": 16, "x": 40, "y": 100, "width": 300, "height": 80, "alignment": "left"},
                {"id": "cap", "kind": "caption", "text": "Caption.", "font_size": 12, "x": 40, "y": 200, "width": 300, "height": 20, "alignment": "left"},
            ],
        }

    def test_clean_render_and_perceive(self):
        artifact = self._artifact()
        self.assertEqual(typography_validate(artifact), [])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "page.png")
            record = typography_renderer(artifact, out, [])
            self.assertTrue(os.path.exists(out))
            evidence = typography_perceive(record)
            self.assertIsNone(evidence["interpretation"])
            measurements = evidence["measurements"]
            self.assertEqual(measurements["width"], 400)
            self.assertEqual(measurements["height"], 500)
            self.assertEqual(len(measurements["elements"]), 3)
            self.assertEqual(
                measurements["alignment_consistency"]["shared_left_edge_count"],
                3,
            )
            hierarchy = measurements["type_hierarchy"]
            self.assertEqual(hierarchy["heading_median"], 32.0)
            self.assertEqual(hierarchy["body_median"], 16.0)
            self.assertEqual(hierarchy["caption_median"], 12.0)
            self.assertGreaterEqual(measurements["whitespace_share"], 0.0)
            self.assertLessEqual(measurements["whitespace_share"], 1.0)
            block = typography_invariants.evaluate_invariants(
                artifact, record, evidence
            )
            self.assertTrue(_check(block, "elements_inside_page")["held"])
            self.assertTrue(_check(block, "heading_larger_than_body")["held"])

    def test_each_planted_fault_is_detected(self):
        artifact = self._artifact()
        faults = typography_faults(artifact)
        self.assertEqual(
            [fault["kind"] for fault in faults],
            ["shrink_heading", "shift_element", "move_into_margin"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            clean_path = os.path.join(tmpdir, "clean.png")
            clean_record = typography_renderer(artifact, clean_path, [])
            clean = typography_perceive(clean_record)
            faulted_path = os.path.join(tmpdir, "faulted.png")
            faulted_record = typography_renderer(artifact, faulted_path, faults)
            self.assertEqual(faulted_record["faults_applied"], faults)

            clean_sizes = {
                entry["id"]: entry["font_size"]
                for entry in clean["measurements"]["elements"]
            }
            faulted = typography_perceive(faulted_record)
            faulted_sizes = {
                entry["id"]: entry["font_size"]
                for entry in faulted["measurements"]["elements"]
            }
            self.assertLess(faulted_sizes["title"], clean_sizes["title"])
            self.assertEqual(faulted_sizes["title"], 16.0)

            shifted = faulted["measurements"]["elements"][0]
            self.assertEqual(shifted["realized_box"]["x"], 70.0)

            margins = faulted["measurements"]["elements"][1]["margins"]
            self.assertEqual(margins["left"], 2.0)
            self.assertEqual(margins["top"], 2.0)

            # The margin fault stays inside the page bounds, so
            # the inside-page invariant holds; the fault is
            # measured in the margins and in the alignment
            # consistency, which reads realized left edges.
            block = typography_invariants.evaluate_invariants(
                artifact, faulted_record, faulted
            )
            self.assertTrue(_check(block, "elements_inside_page")["held"])
            self.assertFalse(_check(block, "shared_alignment_edge")["held"])
            edges = faulted["measurements"]["alignment_consistency"]["left_edges"]
            self.assertEqual(edges["2.0"], 1)

    def test_one_addressed_revision_applied(self):
        artifact = self._artifact()
        revised = typography_revision(
            artifact,
            {"op": "set_font_size", "id": "title", "font_size": 48},
        )
        self.assertEqual(typography_validate(revised), [])
        title = [
            entry
            for entry in revised["elements"]
            if entry["id"] == "title"
        ][0]
        self.assertEqual(title["font_size"], 48.0)
        self.assertEqual(artifact["elements"][0]["font_size"], 32)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "revised.png")
            record = typography_renderer(revised, out, [])
            evidence = typography_perceive(record)
            sizes = {
                entry["id"]: entry["font_size"]
                for entry in evidence["measurements"]["elements"]
            }
            self.assertEqual(sizes["title"], 48.0)
            record_structure = typography_structure(revised)
            self.assertEqual(
                record_structure["type_hierarchy"]["heading_median"], 48.0
            )


class AnimationTests(unittest.TestCase):
    def _artifact(self):
        return {
            "duration": 2.0,
            "frame_rate": 10,
            "canvas": {"width": 120, "height": 80},
            "loop": False,
            "elements": [
                {
                    "id": "ball",
                    "keyframes": [
                        {"time": 0.0, "x": 5, "y": 30, "width": 20, "height": 20, "opacity": 1.0, "easing": "linear"},
                        {"time": 1.0, "x": 50, "y": 30, "width": 20, "height": 20, "opacity": 1.0, "easing": "linear"},
                        {"time": 2.0, "x": 95, "y": 30, "width": 20, "height": 20, "opacity": 1.0, "easing": "linear"},
                    ],
                },
                {
                    "id": "box",
                    "keyframes": [
                        {"time": 0.0, "x": 95, "y": 5, "width": 15, "height": 15, "opacity": 1.0, "easing": "ease-in"},
                        {"time": 2.0, "x": 5, "y": 5, "width": 15, "height": 15, "opacity": 1.0, "easing": "ease-in"},
                    ],
                },
            ],
        }

    def test_clean_render_and_perceive(self):
        artifact = self._artifact()
        self.assertEqual(animation_validate(artifact), [])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "scene.png")
            record = animation_renderer(artifact, out, [])
            self.assertEqual(record["frame_count"], 20)
            for path in record["frame_paths"]:
                self.assertTrue(os.path.exists(path))
            evidence = animation_perceive(record)
            self.assertIsNone(evidence["interpretation"])
            measurements = evidence["measurements"]
            self.assertEqual(measurements["frame_count"], 20)
            self.assertEqual(measurements["duration"], 2.0)
            self.assertEqual(measurements["frame_rate"], 10.0)
            self.assertGreater(measurements["frame_difference_total"], 0.0)
            self.assertEqual(len(measurements["elements"]), 2)
            ball = measurements["elements"][0]
            times = [point["time"] for point in ball["sampled_positions"]]
            self.assertEqual(times, [0.0, 0.5, 1.0, 1.5, 2.0])
            self.assertEqual(ball["sampled_positions"][0]["x"], 5.0)
            self.assertEqual(ball["sampled_positions"][-1]["x"], 95.0)
            block = animation_invariants.evaluate_invariants(
                artifact, record, evidence
            )
            self.assertTrue(_check(block, "frame_count_matches")["held"])
            self.assertTrue(_check(block, "keyframes_inside_duration")["held"])
            self.assertTrue(_check(block, "keyframe_time_order")["held"])
            self.assertTrue(_check(block, "easings_named")["held"])

    def test_out_of_duration_and_unordered_keyframes_are_accepted_and_reported(self):
        artifact = self._artifact()
        artifact["elements"][0]["keyframes"][1]["time"] = 3.0
        self.assertEqual(animation_validate(artifact), [])
        block = animation_invariants.evaluate_invariants(artifact, {})
        self.assertFalse(_check(block, "keyframes_inside_duration")["held"])
        self.assertFalse(_check(block, "keyframe_time_order")["held"])

    def test_each_planted_fault_is_detected(self):
        artifact = self._artifact()
        faults = animation_faults(artifact)
        self.assertEqual(
            [fault["kind"] for fault in faults],
            ["drop_keyframe", "shift_keyframe_time", "swap_easings"],
        )
        shift = [f for f in faults if f["kind"] == "shift_keyframe_time"][0]
        # The shift fault moves an interior keyframe, so the
        # first keyframe stays fixed.
        self.assertEqual(shift["time"], 1.0)
        self.assertEqual(shift["new_time"], 1.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            clean_out = os.path.join(tmpdir, "clean.png")
            clean_record = animation_renderer(artifact, clean_out, [])
            clean = animation_perceive(clean_record)
            faulted_out = os.path.join(tmpdir, "faulted.png")
            faulted_record = animation_renderer(artifact, faulted_out, faults)
            self.assertEqual(faulted_record["faults_applied"], faults)
            faulted = animation_perceive(faulted_record)

            clean_ball = clean["measurements"]["elements"][0]
            faulted_ball = faulted["measurements"]["elements"][0]

            # The start position is unchanged: neither fault
            # moves the first keyframe.
            self.assertEqual(
                _point_at(faulted_ball, 0.0)["x"],
                _point_at(clean_ball, 0.0)["x"],
            )
            self.assertEqual(_point_at(clean_ball, 0.0)["x"], 5.0)
            self.assertEqual(_point_at(faulted_ball, 0.0)["x"], 5.0)

            # Sampling reads the last rendered frame at or
            # before the stated time. Frames span 0 to 2.0
            # inclusive in steps of 2/19, so no frame sits at
            # exactly 1.0 and the clean reading at 1.0 is the
            # realized state at 18/19 seconds on the written
            # track, not the keyframe value at 1.0.
            self.assertAlmostEqual(
                _point_at(clean_ball, 1.0)["x"],
                47.63157894736842,
                places=9,
            )

            # The dropped final keyframe changes the realized
            # track: the faulted track ends at the interior
            # keyframe, so the end position differs from the
            # written final position.
            self.assertEqual(_point_at(clean_ball, 2.0)["x"], 95.0)
            self.assertNotEqual(
                _point_at(faulted_ball, 2.0)["x"],
                _point_at(clean_ball, 2.0)["x"],
            )
            self.assertEqual(_point_at(faulted_ball, 2.0)["x"], 50.0)

            # The shifted interior keyframe changes the
            # realized position sampled at its written time.
            self.assertNotEqual(
                _point_at(faulted_ball, 1.0)["x"],
                _point_at(clean_ball, 1.0)["x"],
            )
            self.assertNotEqual(
                _point_at(faulted_ball, 0.5)["x"],
                _point_at(clean_ball, 0.5)["x"],
            )

            # Easings are exchanged between the two elements.
            clean_easings = animation_structure(artifact)["elements"][0]["easings"]
            faulted_easings = animation_structure(
                {
                    "duration": artifact["duration"],
                    "frame_rate": artifact["frame_rate"],
                    "canvas": artifact["canvas"],
                    "elements": faulted_record["delivered_elements"],
                }
            )["elements"][0]["easings"]
            self.assertNotEqual(faulted_easings, clean_easings)

    def test_one_addressed_revision_applied(self):
        artifact = self._artifact()
        revised = animation_revision(
            artifact,
            {"op": "set_easing", "id": "ball", "time": 0.0, "easing": "ease-out"},
        )
        self.assertEqual(animation_validate(revised), [])
        ball = revised["elements"][0]
        self.assertEqual(ball["keyframes"][0]["easing"], "ease-out")
        self.assertEqual(artifact["elements"][0]["keyframes"][0]["easing"], "linear")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "revised.png")
            record = animation_renderer(revised, out, [])
            self.assertEqual(record["frame_count"], 20)
            evidence = animation_perceive(record)
            self.assertEqual(evidence["measurements"]["frame_count"], 20)


class SoundDesignTests(unittest.TestCase):
    def _artifact(self):
        return {
            "duration": 5.0,
            "events": [
                {"id": "hit", "kind": "percussive hit", "start": 0.0, "duration": 0.5, "gain": 0.8, "filter_cutoff": 180, "bandwidth": 0},
                {"id": "wash", "kind": "filtered noise", "start": 1.0, "duration": 2.0, "gain": 0.4, "filter_cutoff": 1200, "bandwidth": 800},
                {"id": "gap", "kind": "silence", "start": 3.5, "duration": 1.0, "gain": 0.0, "filter_cutoff": 440, "bandwidth": 0},
            ],
        }

    def test_clean_render_and_perceive(self):
        artifact = self._artifact()
        self.assertEqual(sound_validate(artifact), [])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "work.wav")
            record = sound_renderer(artifact, out, [])
            self.assertTrue(os.path.exists(out))
            self.assertEqual(record["event_count"], 3)
            evidence = sound_perceive(record)
            self.assertIsNone(evidence["interpretation"])
            measurements = evidence["measurements"]
            self.assertEqual(measurements["sample_rate"], 44100)
            self.assertEqual(measurements["event_count"], 3)
            self.assertEqual(len(measurements["events"]), 3)
            self.assertGreater(len(measurements["loudness_contour"]), 0)
            self.assertGreaterEqual(measurements["silence_share"], 0.0)
            self.assertLessEqual(measurements["silence_share"], 1.0)
            hit = measurements["events"][0]
            self.assertIsNotNone(hit["onset_time"])
            self.assertAlmostEqual(hit["onset_time"], 0.0, delta=0.1)
            self.assertGreater(hit["realized_duration"], 0.0)
            self.assertGreaterEqual(hit["spectral_centroid_hz"], 0.0)
            self.assertGreaterEqual(hit["spectral_bandwidth_hz"], 0.0)
            block = sound_invariants.evaluate_invariants(
                artifact, record, evidence
            )
            self.assertTrue(_check(block, "delivered_event_count_matches")["held"])
            self.assertTrue(_check(block, "events_within_duration")["held"])
            self.assertTrue(_check(block, "realized_onsets_present")["held"])

    def test_event_ending_after_duration_is_accepted_and_reported(self):
        artifact = self._artifact()
        artifact["events"][1]["start"] = 4.5
        artifact["events"][1]["duration"] = 1.0
        self.assertEqual(sound_validate(artifact), [])
        block = sound_invariants.evaluate_invariants(
            artifact,
            {"delivered_events": copy.deepcopy(artifact["events"])},
        )
        within = _check(block, "events_within_duration")
        self.assertFalse(within["held"])


    def test_each_planted_fault_is_detected(self):
        artifact = self._artifact()
        faults = sound_faults(artifact)
        self.assertEqual(
            [fault["kind"] for fault in faults],
            ["drop_event", "shift_event_start", "raise_event_gain"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            clean_out = os.path.join(tmpdir, "clean.wav")
            clean_record = sound_renderer(artifact, clean_out, [])
            clean = sound_perceive(clean_record)
            faulted_out = os.path.join(tmpdir, "faulted.wav")
            faulted_record = sound_renderer(artifact, faulted_out, faults)
            self.assertEqual(faulted_record["faults_applied"], faults)
            faulted = sound_perceive(faulted_record)

            self.assertEqual(faulted["measurements"]["event_count"], 2)
            faulted_ids = [
                event["id"] for event in faulted["measurements"]["events"]
            ]
            self.assertNotIn("hit", faulted_ids)

            clean_by_id = {
                event["id"]: event
                for event in clean["measurements"]["events"]
            }
            faulted_by_id = {
                event["id"]: event
                for event in faulted["measurements"]["events"]
            }
            self.assertAlmostEqual(
                faulted_by_id["wash"]["onset_time"], 1.25, delta=0.1
            )
            self.assertGreater(
                faulted_by_id["wash"]["peak_amplitude"],
                clean_by_id["wash"]["peak_amplitude"],
            )

            block = sound_invariants.evaluate_invariants(
                artifact, faulted_record, faulted
            )
            self.assertFalse(
                _check(block, "delivered_event_count_matches")["held"]
            )
            self.assertFalse(_check(block, "event_ids_present")["held"])

    def test_one_addressed_revision_applied(self):
        artifact = self._artifact()
        revised = sound_revision(
            artifact,
            {"op": "set_gain", "id": "wash", "gain": 0.9},
        )
        self.assertEqual(sound_validate(revised), [])
        wash = [
            event for event in revised["events"] if event["id"] == "wash"
        ][0]
        self.assertEqual(wash["gain"], 0.9)
        self.assertEqual(artifact["events"][1]["gain"], 0.4)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "revised.wav")
            record = sound_renderer(revised, out, [])
            evidence = sound_perceive(record)
            self.assertEqual(evidence["measurements"]["event_count"], 3)
            by_id = {
                event["id"]: event
                for event in evidence["measurements"]["events"]
            }
            self.assertGreater(
                by_id["wash"]["peak_amplitude"],
                0.4,
            )


if __name__ == "__main__":
    unittest.main()
