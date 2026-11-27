"""tests/test_animation_comparison.py - comparison judgment tests
(animation).

A delivery short by several frames produces frame_count_off
with both counts; a missing keyframed element produces
keyframe_not_realized naming it; and a clean delivery produces
zero findings, with coverage counting the written keyframes
realized and the delivered frames matching the written count.

Delivery is through the animation renderer's own fault
application, and the delivered frames it renders are perceived
through the application's own frame analyzer.  No model calls
are made.
"""

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.apps.animation.app import perceive
from stage2.apps.animation.comparison import compute_comparison
from stage2.apps.animation.invariants import objective_coverage
from animation_renderer import renderer


def _animation():
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


def _compare(artifact, tmpdir, delivered_artifact=None, faults=(), label="render"):
    source = copy.deepcopy(delivered_artifact) if delivered_artifact is not None else artifact
    path = os.path.join(tmpdir, label + ".png")
    record = renderer(source, path, list(faults))
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


class AnimationComparisonTests(unittest.TestCase):
    def test_delivery_short_by_several_frames_produces_frame_count_off_with_both_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _animation()
            delivered = copy.deepcopy(artifact)
            delivered["duration"] = 1.7
            comparison, record, evidence = _compare(
                artifact,
                tmpdir,
                delivered_artifact=delivered,
                label="short",
            )
            matched = _findings(comparison, "frame_count_off")
            self.assertTrue(
                matched,
                "expected a frame_count_off, got %r"
                % (comparison["findings"],),
            )
            finding = matched[0]
            self.assertEqual(finding["written"].get("frame_count"), 20)
            self.assertEqual(finding["written"].get("written_frame_count"), 20)
            self.assertEqual(finding["delivered"].get("frame_count"), 17)
            self.assertEqual(finding["delivered"].get("delivered_frame_count"), 17)
            self.assertIn("20", finding["statement"])
            self.assertIn("17", finding["statement"])
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertIn("frame_count_off", comparison["summary"]["counts"])
            self.assertEqual(comparison["frame_count_written"], 20)
            self.assertEqual(comparison["frame_count_delivered"], 17)
            coverage = _coverage_with_comparison(
                artifact, record, evidence, comparison
            )
            self.assertIn(
                {
                    "item": "delivered frames matching the written count, 17 of 20",
                    "covered": False,
                },
                coverage,
            )

    def test_missing_keyframed_element_produces_keyframe_not_realized_naming_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _animation()
            delivered = copy.deepcopy(artifact)
            delivered["elements"] = [
                element
                for element in delivered["elements"]
                if element["id"] != "box"
            ]
            comparison, record, evidence = _compare(
                artifact,
                tmpdir,
                delivered_artifact=delivered,
                label="missing-element",
            )
            matched = _findings(comparison, "keyframe_not_realized")
            named = [
                finding
                for finding in matched
                if finding["written"].get("id") == "box"
                or finding["written"].get("element") == "box"
                or "'box'" in finding["statement"]
            ]
            self.assertTrue(
                named,
                "expected a keyframe_not_realized naming box, got %r"
                % (comparison["findings"],),
            )
            finding = named[0]
            self.assertIn("box", finding["statement"])
            self.assertEqual(finding["written"].get("id"), "box")
            self.assertAlmostEqual(
                finding["written"].get("written_time"), 0.0, places=6
            )
            self.assertIn("0.0", finding["statement"])
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertIn(
                "keyframe_not_realized",
                comparison["summary"]["counts"],
            )
            self.assertEqual(
                comparison["tolerances"]["keyframe_time_tolerance_seconds"], 0.1
            )
            coverage = _coverage_with_comparison(
                artifact, record, evidence, comparison
            )
            self.assertIn(
                {
                    "item": "written keyframes realized, 3 of 5",
                    "covered": False,
                },
                coverage,
            )

    def test_clean_delivery_produces_zero_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _animation()
            comparison, record, evidence = _compare(artifact, tmpdir)
            self.assertEqual(comparison["findings"], [])
            self.assertEqual(comparison["summary"]["total"], 0)
            self.assertEqual(comparison["summary"]["counts"], {})
            self.assertIn("No findings", comparison["summary"]["line"])
            self.assertEqual(comparison["keyframes_written"], 5)
            self.assertEqual(comparison["keyframes_realized"], 5)
            coverage = _coverage_with_comparison(
                artifact, record, evidence, comparison
            )
            self.assertEqual(
                coverage,
                [
                    {
                        "item": "written keyframes realized, 5 of 5",
                        "covered": True,
                    },
                    {
                        "item": "delivered frames matching the written count, 20 of 20",
                        "covered": True,
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()
