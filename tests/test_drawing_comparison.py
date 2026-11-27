"""tests/test_drawing_comparison.py - comparison judgment tests
(drawing).

A delivery missing one declared shape produces shape_missing
naming it; a shape delivered displaced produces geometry_off
with both positions; an extra delivered shape produces
shape_not_declared; and a clean delivery produces zero
findings, with coverage counting the declared shapes realized
as declared.

Delivery is through the drawing renderer's own fault
application, and the delivered SVG it reports is perceived
through the application's own raster analyzer where a
renderer is available.  Where no SVG renderer is available
in the environment, the instrument reading is taken from the
delivered SVG itself, as the other comparisons fall back to
the delivered document.  No model calls are made.
"""

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.apps.drawing.comparison import compute_comparison
from stage2.apps.drawing.invariants import objective_coverage
from drawing_renderer import renderer


def _drawing():
    return {
        "svg": (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
            '<circle id="sun" cx="60" cy="60" r="30" fill="#ffcc00"/>'
            '<rect id="house" x="100" y="100" width="60" height="40" fill="#336699"/>'
            "</svg>"
        ),
        "elements": [
            {"id": "sun", "kind": "shape", "description": "the sun"},
            {"id": "house", "kind": "shape", "description": "the house"},
        ],
    }


def _render_record(artifact, tmpdir, faults=(), label="render"):
    path = os.path.join(tmpdir, label + ".png")
    try:
        record = renderer(artifact, path, list(faults))
        evidence = None
        if os.path.isfile(path):
            from stage2.apps.drawing.app import perceive

            evidence = perceive(record)
        _attach_delivered_svg(record, artifact, faults)
        return record, evidence
    except RuntimeError:
        # No SVG renderer is available in the environment.
        # Fall back to the delivered SVG, applied through the
        # renderer's own fault definitions and carried on the
        # record as the renderers carry their delivered
        # structures.
        record = {
            "path": path,
            "kind": "image",
            "faults_applied": copy.deepcopy(list(faults or [])),
        }
        _attach_delivered_svg(record, artifact, faults)
        return record, None


def _attach_delivered_svg(record, artifact, faults):
    # The delivered SVG is the fault-applied written SVG, as
    # the renderer reports it; the fallback applies the same
    # fault definitions to the same written SVG.
    if isinstance(record, dict) and record.get("delivered_svg"):
        return
    import xml.etree.ElementTree as ET

    root = ET.fromstring(artifact.get("svg", ""))
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_element":
            parent = None
            element = None
            for candidate in root.iter():
                if candidate.get("id") == fault.get("id"):
                    element = candidate
                    break
            if element is not None:
                for candidate in root.iter():
                    for child in list(candidate):
                        if child is element:
                            parent = candidate
                            break
            if parent is not None:
                parent.remove(element)
        elif kind == "shift_element":
            element = None
            for candidate in root.iter():
                if candidate.get("id") == fault.get("id"):
                    element = candidate
                    break
            if element is not None:
                def _number(value, default=0.0):
                    try:
                        return float(value)
                    except Exception:
                        return default

                dx = _number(fault.get("dx"))
                dy = _number(fault.get("dy"))
                for key in ("x", "x1", "x2", "cx"):
                    if element.get(key) is not None:
                        element.set(key, str(_number(element.get(key)) + dx))
                for key in ("y", "y1", "y2", "cy"):
                    if element.get(key) is not None:
                        element.set(key, str(_number(element.get(key)) + dy))
        elif kind == "recolor":
            element = None
            for candidate in root.iter():
                if candidate.get("id") == fault.get("id"):
                    element = candidate
                    break
            if element is not None:
                element.set("fill", fault.get("color"))
    record["delivered_svg"] = ET.tostring(root, encoding="unicode")


def _compare(artifact, tmpdir, delivered_artifact=None, faults=(), label="render"):
    source = copy.deepcopy(delivered_artifact) if delivered_artifact is not None else artifact
    record, evidence = _render_record(source, tmpdir, faults, label)
    return compute_comparison(artifact, record, evidence), record, evidence


def _findings(comparison, kind):
    return [finding for finding in comparison["findings"] if finding["kind"] == kind]


def _coverage_with_comparison(artifact, record, evidence, comparison):
    evidence_with_comparison = copy.deepcopy(evidence) if isinstance(evidence, dict) else {}
    record_with_comparison = copy.deepcopy(record) if isinstance(record, dict) else {}
    evidence_with_comparison["comparison"] = comparison
    record_with_comparison["comparison"] = comparison
    return objective_coverage(artifact, record_with_comparison, evidence_with_comparison)


class DrawingComparisonTests(unittest.TestCase):
    def test_missing_declared_shape_produces_shape_missing_naming_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _drawing()
            comparison, _record, _evidence = _compare(
                artifact,
                tmpdir,
                faults=({"kind": "drop_element", "id": "house"},),
                label="missing-shape",
            )
            matched = _findings(comparison, "shape_missing")
            named = [
                finding
                for finding in matched
                if finding["written"].get("id") == "house"
                or "'house'" in finding["statement"]
                or "house" in finding["statement"]
            ]
            self.assertTrue(
                named,
                "expected a shape_missing naming house, got %r"
                % (comparison["findings"],),
            )
            finding = named[0]
            self.assertIn("house", finding["statement"])
            self.assertIn("rect", finding["statement"])
            self.assertIn("x=100.0", finding["statement"])
            self.assertEqual(finding["written"].get("id"), "house")
            self.assertEqual(finding["written"].get("kind"), "rect")
            self.assertEqual(
                finding["written"].get("position"), {"x": 100.0, "y": 100.0}
            )
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertIn("shape_missing", comparison["summary"]["counts"])
            self.assertEqual(
                comparison["summary"]["counts"]["shape_missing"], 1
            )
            coverage = _coverage_with_comparison(
                artifact, _record, _evidence, comparison
            )
            self.assertIn(
                {
                    "item": "declared shapes realized as declared, 1 of 2",
                    "covered": False,
                },
                coverage,
            )

    def test_displaced_shape_produces_geometry_off_with_both_positions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _drawing()
            comparison, _record, _evidence = _compare(
                artifact,
                tmpdir,
                faults=(
                    {"kind": "shift_element", "id": "house", "dx": 20, "dy": 10},
                ),
                label="displaced-shape",
            )
            matched = _findings(comparison, "geometry_off")
            named = [
                finding
                for finding in matched
                if finding["written"].get("id") == "house"
                or "'house'" in finding["statement"]
            ]
            self.assertTrue(
                named,
                "expected a geometry_off naming house, got %r"
                % (comparison["findings"],),
            )
            finding = named[0]
            self.assertIn("house", finding["statement"])
            self.assertIn("x=100.0", finding["statement"])
            self.assertIn("x=120.0", finding["statement"])
            self.assertIn("y=100.0", finding["statement"])
            self.assertIn("y=110.0", finding["statement"])
            self.assertEqual(finding["written"].get("id"), "house")
            self.assertEqual(
                finding["written"].get("position"), {"x": 100.0, "y": 100.0}
            )
            self.assertEqual(
                finding["delivered"].get("position"), {"x": 120.0, "y": 110.0}
            )
            self.assertEqual(
                finding["written"].get("declared_position"), {"x": 100.0, "y": 100.0}
            )
            self.assertEqual(
                finding["delivered"].get("declared_position"),
                {"x": 120.0, "y": 110.0},
            )
            self.assertEqual(
                comparison["tolerances"]["geometry_tolerance"], 0.5
            )
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertIn("geometry_off", comparison["summary"]["counts"])

    def test_extra_delivered_shape_produces_shape_not_declared(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _drawing()
            delivered = copy.deepcopy(artifact)
            delivered["svg"] = delivered["svg"].replace(
                "</svg>",
                '<circle id="moon" cx="150" cy="40" r="15" fill="#dddddd"/></svg>',
            )
            comparison, _record, _evidence = _compare(
                artifact,
                tmpdir,
                delivered_artifact=delivered,
                label="extra-shape",
            )
            matched = _findings(comparison, "shape_not_declared")
            self.assertTrue(
                matched,
                "expected a shape_not_declared, got %r"
                % (comparison["findings"],),
            )
            finding = matched[0]
            self.assertIn("moon", finding["statement"])
            self.assertIn("circle", finding["statement"])
            self.assertIn("cx=150.0", finding["statement"])
            self.assertEqual(finding["delivered"].get("id"), "moon")
            self.assertEqual(finding["delivered"].get("kind"), "circle")
            self.assertEqual(
                finding["delivered"].get("position"), {"cx": 150.0, "cy": 40.0}
            )
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertIn("shape_not_declared", comparison["summary"]["counts"])
            self.assertEqual(_findings(comparison, "shape_missing"), [])

    def test_clean_delivery_produces_zero_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _drawing()
            comparison, record, evidence = _compare(artifact, tmpdir)
            self.assertEqual(comparison["findings"], [])
            self.assertEqual(comparison["summary"]["total"], 0)
            self.assertEqual(comparison["summary"]["counts"], {})
            self.assertIn("No findings", comparison["summary"]["line"])
            self.assertEqual(comparison["shapes_declared"], 2)
            self.assertEqual(comparison["shapes_realized_as_declared"], 2)
            coverage = _coverage_with_comparison(
                artifact, record, evidence, comparison
            )
            self.assertEqual(
                coverage,
                [
                    {
                        "item": "declared shapes realized as declared, 2 of 2",
                        "covered": True,
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()
