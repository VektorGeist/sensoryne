"""tests/test_typography_comparison.py - comparison judgment
tests (typography).

A delivery missing one declared block produces block_missing
naming it; a block delivered displaced produces
box_displaced with both boxes; a block whose text overflows
produces text_overflow; and a clean delivery produces zero
findings, with coverage counting the declared blocks
realized at their declared placements.

Delivery is through the typography renderer's own fault
application and the delivered elements it reports, perceived
through the application's own raster analyzer.  No model
calls are made.
"""

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.apps.typography.app import perceive
from stage2.apps.typography.comparison import compute_comparison
from stage2.apps.typography.invariants import objective_coverage
from typography_renderer import renderer


def _page():
    return {
        "page": {"width": 400, "height": 500, "background": "#ffffff"},
        "elements": [
            {
                "id": "title",
                "kind": "heading",
                "text": "Title",
                "font_size": 32,
                "x": 40,
                "y": 30,
                "width": 300,
                "height": 50,
                "alignment": "left",
            },
            {
                "id": "body",
                "kind": "body",
                "text": "Body text.",
                "font_size": 16,
                "x": 40,
                "y": 100,
                "width": 300,
                "height": 80,
                "alignment": "left",
            },
            {
                "id": "cap",
                "kind": "caption",
                "text": "Caption.",
                "font_size": 12,
                "x": 40,
                "y": 200,
                "width": 300,
                "height": 20,
                "alignment": "left",
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


class TypographyComparisonTests(unittest.TestCase):
    def test_missing_declared_block_produces_block_missing_naming_it(self):
        """A delivery missing one declared block produces
        block_missing naming it by opening words and declared
        placement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _page()
            delivered = copy.deepcopy(artifact)
            delivered["elements"] = [
                element
                for element in delivered["elements"]
                if element.get("id") != "cap"
            ]
            comparison, _record, _evidence = _compare(
                artifact,
                tmpdir,
                delivered_artifact=delivered,
                label="missing-block",
            )
            matched = _findings(comparison, "block_missing")
            named = [
                finding
                for finding in matched
                if finding["written"].get("id") == "cap"
                or "cap" in finding["statement"]
                or "Caption" in finding["statement"]
            ]
            self.assertTrue(
                named,
                "expected a block_missing naming cap, got %r"
                % (comparison["findings"],),
            )
            finding = named[0]
            self.assertIn("cap", finding["statement"])
            self.assertIn("Caption", finding["statement"])
            self.assertIn("x=40", finding["statement"])
            self.assertIn("y=200", finding["statement"])
            self.assertEqual(finding["written"].get("id"), "cap")
            self.assertEqual(finding["written"].get("opening_words"), "Caption.")
            self.assertEqual(
                finding["written"].get("declared_box"),
                {"x": 40.0, "y": 200.0, "width": 300.0, "height": 20.0},
            )
            self.assertEqual(finding["written"].get("box"), {"x": 40.0, "y": 200.0, "width": 300.0, "height": 20.0})
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertIn("block_missing", comparison["summary"]["counts"])
            self.assertEqual(comparison["summary"]["counts"]["block_missing"], 1)

    def test_displaced_block_produces_box_displaced_with_both_boxes(self):
        """A block delivered displaced produces box_displaced
        with both the declared and the realized boxes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _page()
            comparison, _record, _evidence = _compare(
                artifact,
                tmpdir,
                faults=({"kind": "shift_element", "id": "body", "dx": 30, "dy": 20},),
                label="displaced-block",
            )
            matched = _findings(comparison, "box_displaced")
            named = [
                finding
                for finding in matched
                if finding["written"].get("id") == "body"
                or "body" in finding["statement"]
            ]
            self.assertTrue(
                named,
                "expected a box_displaced naming body, got %r"
                % (comparison["findings"],),
            )
            finding = named[0]
            self.assertIn("body", finding["statement"])
            self.assertIn("x=40", finding["statement"])
            self.assertIn("x=70", finding["statement"])
            self.assertIn("y=100", finding["statement"])
            self.assertIn("y=120", finding["statement"])
            self.assertEqual(finding["written"].get("id"), "body")
            self.assertEqual(
                finding["written"].get("declared_box"),
                {"x": 40.0, "y": 100.0, "width": 300.0, "height": 80.0},
            )
            self.assertEqual(finding["delivered"].get("realized_box", {}).get("x"), 70.0)
            self.assertEqual(finding["delivered"].get("realized_box", {}).get("y"), 120.0)
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertIn("box_displaced", comparison["summary"]["counts"])
            self.assertEqual(comparison["tolerances"]["box_tolerance"], 0.5)

    def test_overflowing_block_produces_text_overflow(self):
        """A block whose text overflows produces text_overflow
        with the measured extent and the declared box."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _page()
            delivered = copy.deepcopy(artifact)
            for element in delivered["elements"]:
                if element.get("id") == "body":
                    element["text"] = "Body text is far too long for its declared box." * 3
            comparison, _record, _evidence = _compare(
                artifact,
                tmpdir,
                delivered_artifact=delivered,
                label="overflow-block",
            )
            matched = _findings(comparison, "text_overflow")
            named = [
                finding
                for finding in matched
                if finding["written"].get("id") == "body"
                or "body" in finding["statement"]
            ]
            self.assertTrue(
                named,
                "expected a text_overflow naming body, got %r"
                % (comparison["findings"],),
            )
            finding = named[0]
            self.assertIn("body", finding["statement"])
            self.assertEqual(finding["written"].get("id"), "body")
            self.assertEqual(
                finding["written"].get("declared_box"),
                {"x": 40.0, "y": 100.0, "width": 300.0, "height": 80.0},
            )
            measured = finding["delivered"].get("measured_extent") or finding["delivered"].get("realized_box")
            self.assertIsNotNone(measured)
            self.assertGreater(measured.get("width", 0.0), 300.0)
            self.assertIn("overflows", finding["statement"])
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertIn("text_overflow", comparison["summary"]["counts"])

    def test_clean_delivery_produces_zero_findings(self):
        """A clean delivery produces zero findings and no
        findings line, with coverage counting the declared
        blocks realized at their declared placements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _page()
            comparison, record, evidence = _compare(artifact, tmpdir)
            self.assertEqual(comparison["findings"], [])
            self.assertEqual(comparison["summary"]["total"], 0)
            self.assertEqual(comparison["summary"]["counts"], {})
            self.assertIn("No findings", comparison["summary"]["line"])
            self.assertEqual(comparison["blocks_declared"], 3)
            self.assertEqual(comparison["blocks_realized_at_declared_placements"], 3)
            coverage = _coverage_with_comparison(
                artifact, record, evidence, comparison
            )
            self.assertEqual(
                coverage,
                [
                    {
                        "item": "declared blocks realized at their declared placements, 3 of 3",
                        "covered": True,
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()
