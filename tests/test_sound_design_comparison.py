"""tests/test_sound_design_comparison.py - comparison judgment tests
(sound_design).

A delivered render missing one written event produces
event_not_realized naming it; an event rendered half a second
late produces onset_displaced with both times and the
displacement; an extra rendered sound produces
onset_not_written naming its realized onset; and a clean
delivery produces zero findings, with coverage counting the
events realized at their written onsets.

No model calls are made.
"""

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.apps.sound_design.app import perceive
from stage2.apps.sound_design.comparison import compute_comparison
from stage2.apps.sound_design.invariants import objective_coverage
from sound_design_renderer import renderer


def _work():
    return {
        "duration": 6.0,
        "events": [
            {
                "id": "hit",
                "kind": "percussive hit",
                "start": 0.0,
                "duration": 0.5,
                "gain": 0.8,
                "filter_cutoff": 180,
                "bandwidth": 0,
            },
            {
                "id": "wash",
                "kind": "filtered noise",
                "start": 1.0,
                "duration": 2.0,
                "gain": 0.4,
                "filter_cutoff": 1200,
                "bandwidth": 800,
            },
            {
                "id": "tail",
                "kind": "noise burst",
                "start": 4.0,
                "duration": 0.5,
                "gain": 0.5,
                "filter_cutoff": 4000,
                "bandwidth": 1000,
            },
        ],
    }


def _compare(artifact, tmpdir, delivered_artifact=None, faults=(), label="render"):
    source = copy.deepcopy(delivered_artifact) if delivered_artifact is not None else artifact
    path = os.path.join(tmpdir, label + ".wav")
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


class SoundDesignComparisonTests(unittest.TestCase):
    def test_missing_written_event_produces_event_not_realized_naming_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _work()
            comparison, _record, _evidence = _compare(
                artifact,
                tmpdir,
                faults=({"kind": "drop_event", "id": "wash"},),
                label="missing-event",
            )
            matched = _findings(comparison, "event_not_realized")
            named = [
                finding
                for finding in matched
                if finding["written"].get("id") == "wash"
                or finding["written"].get("event") == "wash"
                or "'wash'" in finding["statement"]
            ]
            self.assertTrue(
                named,
                "expected an event_not_realized naming wash, got %r"
                % (comparison["findings"],),
            )
            finding = named[0]
            self.assertIn("wash", finding["statement"])
            self.assertEqual(finding["written"].get("id"), "wash")
            self.assertAlmostEqual(
                finding["written"].get("written_onset"), 1.0, places=6
            )
            self.assertIn("1.0", finding["statement"])
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertIn(
                "event_not_realized",
                comparison["summary"]["counts"],
            )
            self.assertEqual(
                comparison["tolerances"]["onset_tolerance_seconds"], 0.1
            )

    def test_event_half_a_second_late_produces_onset_displaced_with_both_times(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _work()
            comparison, _record, _evidence = _compare(
                artifact,
                tmpdir,
                faults=(
                    {
                        "kind": "shift_event_start",
                        "id": "wash",
                        "from": 1.0,
                        "new_start": 1.5,
                    },
                ),
                label="displaced",
            )
            matched = _findings(comparison, "onset_displaced")
            named = [
                finding
                for finding in matched
                if finding["written"].get("id") == "wash"
                or finding["written"].get("event") == "wash"
            ]
            self.assertTrue(
                named,
                "expected an onset_displaced naming wash, got %r"
                % (comparison["findings"],),
            )
            finding = named[0]
            self.assertAlmostEqual(
                finding["written"].get("written_onset"), 1.0, places=6
            )
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
                displacement = float(realized) - float(
                    finding["written"].get("written_onset")
                )
            self.assertAlmostEqual(displacement, 0.5, delta=0.1)
            self.assertIn("wash", finding["statement"])
            self.assertIn(str(finding["written"].get("written_onset")), finding["statement"])
            self.assertIn(str(realized), finding["statement"])
            self.assertEqual(_findings(comparison, "event_not_realized"), [])

    def test_extra_rendered_sound_produces_onset_not_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _work()
            delivered = copy.deepcopy(artifact)
            delivered["events"].append(
                {
                    "id": "extra",
                    "kind": "noise burst",
                    "start": 5.0,
                    "duration": 0.5,
                    "gain": 0.5,
                    "filter_cutoff": 4000,
                    "bandwidth": 1000,
                }
            )
            comparison, _record, _evidence = _compare(
                artifact,
                tmpdir,
                delivered_artifact=delivered,
                label="extra-sound",
            )
            matched = _findings(comparison, "onset_not_written")
            self.assertTrue(
                matched,
                "expected an onset_not_written, got %r"
                % (comparison["findings"],),
            )
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
            self.assertEqual(
                comparison["events_realized_at_written_onsets"], 3
            )
            coverage = _coverage_with_comparison(
                artifact, record, evidence, comparison
            )
            self.assertEqual(
                coverage,
                [
                    {
                        "item": "events realized at their written onsets, 3 of 3",
                        "covered": True,
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()
