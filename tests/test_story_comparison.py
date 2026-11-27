"""tests/test_story_comparison.py - comparison judgment tests
(story).

A delivered story missing one promised event produces
event_not_delivered naming that event; a story missing a whole
section produces section_missing; a section delivered at half
its written share produces section_share_off with both shares;
and a clean delivery produces zero findings.

No model calls are made.
"""

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.apps.story.app import perceive
from stage2.apps.story.comparison import compute_comparison
from stage2.apps.story.invariants import objective_coverage
from story_renderer import renderer


def _story():
    return {
        "objective": {
            "section_count": 2,
            "movements": ["opening", "ending"],
            "promised_events": ["a door opens", "a key turns", "rain falls"],
        },
        "sections": [
            {
                "title": "Opening",
                "text": "A door opens in the old house and Mara steps inside today.",
            },
            {
                "title": "Closing",
                "text": "A key turns in the lock as rain falls on the roof all night long today.",
            },
        ],
    }


def _compare(artifact, tmpdir, delivered_artifact=None, faults=(), label="render"):
    source = copy.deepcopy(delivered_artifact) if delivered_artifact is not None else artifact
    path = os.path.join(tmpdir, label + ".txt")
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


class StoryComparisonTests(unittest.TestCase):
    def test_missing_promised_event_produces_event_not_delivered_naming_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _story()
            comparison, _record, _evidence = _compare(artifact, tmpdir, faults=())
            self.assertEqual(comparison["findings"], [])
            delivered = copy.deepcopy(artifact)
            delivered["sections"][1]["text"] = (
                "She pauses before the old gate and listens to the night outside."
            )
            comparison, _record, _evidence = _compare(
                artifact, tmpdir, delivered_artifact=delivered, label="missing-event"
            )
            matched = _findings(comparison, "event_not_delivered")
            named = [
                finding
                for finding in matched
                if finding["written"].get("event") == "a key turns"
                or finding["written"].get("promised_event") == "a key turns"
                or "a key turns" in finding["statement"]
            ]
            self.assertTrue(
                named,
                "expected an event_not_delivered naming a key turns, got %r"
                % (comparison["findings"],),
            )
            finding = named[0]
            self.assertIn("a key turns", finding["statement"])
            self.assertEqual(finding["written"].get("event"), "a key turns")
            self.assertEqual(finding["written"].get("section"), 2)
            self.assertIn("Closing", finding["statement"])
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertEqual(
                comparison["summary"]["counts"].get("event_not_delivered"), 2
            )
            coverage = _coverage_with_comparison(
                artifact, _record, _evidence, comparison
            )
            self.assertIn(
                {
                    "item": "promised events delivered, 1 of 3",
                    "covered": False,
                },
                coverage,
            )
            self.assertIn(
                {
                    "item": "sections delivered in written order, 2 of 2",
                    "covered": True,
                },
                coverage,
            )

    def test_missing_section_produces_section_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _story()
            comparison, _record, _evidence = _compare(
                artifact, tmpdir, faults=({"kind": "drop_section", "section": 1},),
                label="missing-section",
            )
            matched = _findings(comparison, "section_missing")
            named = [
                finding
                for finding in matched
                if (finding["written"].get("section") == 1 and "Opening" in finding["statement"])
                or "Opening" in finding["where"]
            ]
            self.assertTrue(
                named,
                "expected a section_missing naming Opening, got %r"
                % (comparison["findings"],),
            )
            finding = named[0]
            self.assertEqual(finding["written"].get("title"), "Opening")
            self.assertEqual(finding["written"].get("section"), 1)
            self.assertIn("Opening", finding["statement"])
            self.assertIn("section 1", finding["statement"].lower())

    def test_section_at_half_written_share_produces_section_share_off(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _story()
            written_words = len(artifact["sections"][0]["text"].split())
            self.assertEqual(written_words, 12)
            delivered = copy.deepcopy(artifact)
            new_first = "A door opens in the house."
            first_words = len(new_first.split())
            self.assertEqual(first_words * 2, written_words)
            delivered["sections"][0]["text"] = new_first
            removed = written_words - first_words
            filler = " ".join(["clearly"] * removed)
            delivered["sections"][1]["text"] += " " + filler
            self.assertEqual(
                sum(len(s["text"].split()) for s in delivered["sections"]),
                sum(len(s["text"].split()) for s in artifact["sections"]),
            )
            comparison, _record, _evidence = _compare(
                artifact, tmpdir, delivered_artifact=delivered, label="share"
            )
            matched = _findings(comparison, "section_share_off")
            named = [
                finding
                for finding in matched
                if finding["written"].get("title") == "Opening"
                or "Opening" in finding["where"]
            ]
            self.assertTrue(
                named,
                "expected a section_share_off for Opening with "
                "both shares, got %r" % (comparison["findings"],),
            )
            finding = named[0]
            written_share = finding["written"].get("written_share")
            if written_share is None:
                written_share = finding["written"].get("share_of_whole")
            delivered_share = finding["delivered"].get("delivered_share")
            if delivered_share is None:
                delivered_share = finding["delivered"].get("share_of_whole")
            written_total = sum(
                len(s["text"].split()) for s in artifact["sections"]
            )
            delivered_total = sum(
                len(s["text"].split()) for s in delivered["sections"]
            )
            expected_written = float(written_words) / written_total
            expected_delivered = float(first_words) / delivered_total
            self.assertAlmostEqual(written_share, expected_written, places=6)
            self.assertAlmostEqual(delivered_share, expected_delivered, places=6)
            self.assertAlmostEqual(delivered_share * 2.0, written_share, places=6)
            self.assertIn(str(delivered_share), finding["statement"])
            self.assertIn(str(written_share), finding["statement"])
            self.assertEqual(
                comparison["tolerances"]["section_share_tolerance"], 0.05
            )

    def test_clean_delivery_produces_zero_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _story()
            comparison, record, evidence = _compare(artifact, tmpdir, faults=())
            self.assertEqual(comparison["findings"], [])
            self.assertEqual(comparison["summary"]["total"], 0)
            self.assertEqual(comparison["summary"]["counts"], {})
            self.assertIn("No findings", comparison["summary"]["line"])
            coverage = _coverage_with_comparison(
                artifact, record, evidence, comparison
            )
            self.assertEqual(
                coverage,
                [
                    {
                        "item": "promised events delivered, 3 of 3",
                        "covered": True,
                    },
                    {
                        "item": "sections delivered in written order, 2 of 2",
                        "covered": True,
                    },
                ],
            )

    def test_swapped_sections_produce_section_out_of_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _story()
            comparison, _record, _evidence = _compare(
                artifact,
                tmpdir,
                faults=(
                    {"kind": "swap_sections", "first": 1, "second": 2},
                ),
                label="swapped",
            )
            matched = _findings(comparison, "section_out_of_order")
            self.assertTrue(
                matched,
                "expected a section_out_of_order naming Opening "
                "and Closing, got %r" % (comparison["findings"],),
            )
            finding = matched[0]
            self.assertIn("Opening", finding["statement"])
            self.assertIn("Closing", finding["statement"])
            self.assertIn("written section", finding["statement"])


if __name__ == "__main__":
    unittest.main()
