"""tests/test_shaping_comparison.py - comparison reaches the model whole.

Regression test for the defect verified from session
music-perceptual-5: the stored comparison carried 9 dynamics_off
findings, but the shaped copy handed to the model replaced the
whole comparison object by a reference ("18026 chars; full data
in session record"), so the findings were never visible to the
model and the loop closed on an acceptance the evidence did
not support.

The comparison is protected in the same sense as the invariants:
in any shaped copy its findings list is present in full, every
finding carrying its kind, where, written, delivered/heard and
statement fields, with its per_part figures, summary and
tolerances present. It is never reference-replaced whole and
no finding is dropped or replaced. Only a bulky detail payload
strictly inside a finding may be trimmed under budget pressure.
"""

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.shape import EVIDENCE_BUDGET, _size, shape_document, shape_evidence


def _large_comparison_evidence(num_findings=180):
    findings = []
    for i in range(num_findings):
        findings.append(
            {
                "kind": "dynamics_off" if i % 2 == 0 else "check_failed",
                "where": "part Melody start %s note A4 midi 69" % float(i),
                "written": {
                    "part": "Melody",
                    "start": float(i),
                    "midi": 69,
                    "expected_output": "expected output %d" % i,
                },
                "delivered": {
                    "part": "Melody",
                    "start": float(i),
                    "midi": 69,
                    "actual_output": "actual output %d" % i,
                    "exit_status": 1,
                },
                "heard": {
                    "difference_db": -5.0,
                    "attack_loudness_db": -20.0,
                    "median_adjusted_level_db": -10.0,
                },
                "statement": (
                    "Finding %d: part Melody note A4 midi 69 at %s seconds "
                    "heard attack loudness differs from the written work "
                    "by more than the stated tolerance for this check."
                    % (i, float(i))
                ),
            }
        )
    return {
        "summary": "Heard a tonal sound: pitched at A4, 440.0 Hz.",
        "measurements": {
            "value": 3,
            "pitch_hz": 440.0,
            "waveform": list(range(20000)),
        },
        "invariants": {
            "checked": [
                {"id": "notes_heard_of_written", "description": "Notes heard against written.", "held": False, "value": [{"part": "Melody", "notes_written": 76, "notes_heard": 76}]},
                {"id": "notes_in_instrument_range", "description": "Every written note in range.", "held": True, "value": [{"part": "Melody", "out_of_range": 0}]},
            ],
            "summary": {"total_checked": 2, "held": 1, "broken": 1},
        },
        "reliability": {"instrument": "Sensoryne hearing", "limits": ["Comparison pitch tolerance is 50 cents."]},
        "objective_coverage": [{"item": "part Melody: 76 of 76 notes heard", "covered": True}],
        "comparison": {
            "findings": findings,
            "summary": {"total": len(findings), "counts": {"dynamics_off": 90, "check_failed": 90}, "line": "%d finding(s): check_failed=90, dynamics_off=90." % len(findings)},
            "per_part": [{"part": "Melody", "notes_written": 76, "notes_heard": 76, "notes_not_heard": 0}],
            "tolerances": {"pitch_tolerance_cents": 50.0, "dynamics_threshold_db": 4.0},
        },
        "delta": {"from_version": "v1", "to_version": "v2", "changed_measurements": []},
    }


class ComparisonProtectionTests(unittest.TestCase):
    def test_comparison_reaches_model_whole_when_over_budget(self):
        evidence = _large_comparison_evidence()
        original = copy.deepcopy(evidence)
        # The comparison is large enough to be a replacement
        # candidate under the old shaping, and the whole
        # document is over budget, so budget enforcement runs.
        self.assertGreater(_size(evidence), EVIDENCE_BUDGET)
        self.assertGreater(_size(evidence["comparison"]), 10000)

        shaped = shape_evidence(evidence)

        # Shaping never mutates its input.
        self.assertEqual(evidence, original)

        comparison = shaped["comparison"]
        self.assertNotIn("_reference", comparison)
        self.assertEqual(len(comparison["findings"]), len(original["comparison"]["findings"]))
        for shaped_finding, original_finding in zip(comparison["findings"], original["comparison"]["findings"]):
            self.assertNotIn("_reference", shaped_finding)
            for field in ("kind", "where", "written", "delivered", "heard", "statement"):
                self.assertIn(field, shaped_finding)
                self.assertEqual(shaped_finding[field], original_finding[field])
        self.assertEqual(comparison["per_part"], original["comparison"]["per_part"])
        self.assertEqual(comparison["summary"], original["comparison"]["summary"])
        self.assertEqual(comparison["tolerances"], original["comparison"]["tolerances"])

    def test_bulky_detail_trim_keeps_named_fields_and_document_shape(self):
        evidence = _large_comparison_evidence()
        # Bulky per-note detail carried inside one finding: only
        # this may be reference-replaced under budget pressure.
        evidence["comparison"]["findings"][0]["written"]["notes"] = [
            {"start": float(i), "midi": 60 + (i % 12), "detail": ("per-note measurement detail %d " % i) * 20}
            for i in range(200)
        ]
        original = copy.deepcopy(evidence)
        self.assertGreater(_size(evidence), EVIDENCE_BUDGET)

        shaped_doc = shape_document({"evidence": copy.deepcopy(evidence)})
        comparison = shaped_doc["evidence"]["comparison"]

        self.assertNotIn("_reference", comparison)
        self.assertEqual(len(comparison["findings"]), len(original["comparison"]["findings"]))
        for shaped_finding, original_finding in zip(comparison["findings"], original["comparison"]["findings"]):
            self.assertNotIn("_reference", shaped_finding)
            for field in ("kind", "where", "written", "delivered", "heard", "statement"):
                self.assertIn(field, shaped_finding)
            self.assertIsInstance(shaped_finding["written"], dict)
            self.assertIsInstance(shaped_finding["delivered"], dict)
            self.assertEqual(shaped_finding["kind"], original_finding["kind"])
            self.assertEqual(shaped_finding["where"], original_finding["where"])
            self.assertEqual(shaped_finding["statement"], original_finding["statement"])
            self.assertEqual(shaped_finding["delivered"], original_finding["delivered"])
            self.assertEqual(shaped_finding["heard"], original_finding["heard"])
        self.assertEqual(comparison["per_part"], original["comparison"]["per_part"])
        self.assertEqual(comparison["summary"], original["comparison"]["summary"])
        self.assertEqual(comparison["tolerances"], original["comparison"]["tolerances"])


if __name__ == "__main__":
    unittest.main()
