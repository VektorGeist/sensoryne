"""tests/test_poetry_comparison.py - comparison judgment tests
(poetry).

Each planted delivery fault, applied through the
application's own fault definitions and rendered through the
actual poetry renderer, produces the named comparison
finding naming the ground truth's stanza and line; a clean
delivery produces zero findings; a poem delivered complete
but with one line's text altered produces line_text_differs
naming both texts; and a stated syllables-per-line form
produces a form_off naming the syllable estimate.

No model calls are made.
"""

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.apps.poetry.app import (
    generate_ground_truth,
    perceive,
    plant_faults,
)
from stage2.apps.poetry.comparison import compute_comparison
from stage2.apps.poetry.instrument import syllable_estimate
from stage2.apps.poetry.invariants import objective_coverage
from poetry_renderer import renderer


def _poem():
    return {
        "stanzas": [
            ["The brook descends the mossy hill", "A thrush repeats the hidden spring"],
            ["The moon steadies the meadow gate", "The lantern keeps the valley road"],
        ]
    }


def _locate(artifact, line_number):
    """Return (stanza, line_in_stanza, global_line) for a numbered line."""
    remaining = line_number
    for stanza_index, stanza in enumerate(artifact["stanzas"], start=1):
        if 1 <= remaining <= len(stanza):
            return stanza_index, remaining, line_number
        remaining -= len(stanza)
    raise AssertionError("line %r is out of range" % (line_number,))


def _compare(artifact, tmpdir, faults=(), label="render"):
    path = os.path.join(tmpdir, label + ".txt")
    record = renderer(artifact, path, list(faults))
    evidence = perceive(record)
    return compute_comparison(artifact, record, evidence), record, evidence


def _findings(comparison, kind):
    return [finding for finding in comparison["findings"] if finding["kind"] == kind]


class PlantedFaultPoetryComparisonTests(unittest.TestCase):
    def test_each_planted_fault_produces_its_finding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _poem()
            faults = plant_faults(artifact)
            self.assertEqual(
                [fault["kind"] for fault in faults],
                ["drop_line", "duplicate_line", "swap_lines"],
            )
            ground_truth = generate_ground_truth(faults)
            fault_by_kind = {fault["kind"]: fault for fault in faults}
            cases = (
                ("drop_line", "line_missing"),
                ("duplicate_line", "line_duplicated"),
                ("swap_lines", "lines_swapped"),
            )
            for fault_kind, finding_kind in cases:
                single_fault = [fault_by_kind[fault_kind]]
                label = fault_kind.replace("_", "-")
                comparison, _record, _evidence = _compare(
                    artifact, tmpdir, single_fault, label=label
                )
                matched = _findings(comparison, finding_kind)
                self.assertTrue(
                    matched,
                    "expected a %s finding for %s, got %r"
                    % (finding_kind, ground_truth, comparison["findings"]),
                )
                if fault_kind == "drop_line":
                    line_number = fault_by_kind[fault_kind].get("line")
                    stanza, line_in_stanza, global_line = _locate(
                        artifact, line_number
                    )
                    written = artifact["stanzas"][stanza - 1][
                        line_in_stanza - 1
                    ]
                    named = [
                        finding
                        for finding in matched
                        if finding["written"]["stanza"] == stanza
                        and finding["written"]["line_in_stanza"]
                        == line_in_stanza
                        and finding["written"]["line"] == global_line
                        and finding["written"]["global_line"] == global_line
                    ]
                    self.assertTrue(
                        named,
                        "expected a line_missing naming stanza %d "
                        "line %d (global %d), got %r"
                        % (stanza, line_in_stanza, global_line, matched),
                    )
                    self.assertIn(
                        " ".join(written.split()[:4]),
                        named[0]["statement"],
                    )
                    self.assertEqual(named[0]["written"]["text"], written)
                elif fault_kind == "duplicate_line":
                    line_number = fault_by_kind[fault_kind].get("line")
                    stanza, line_in_stanza, global_line = _locate(
                        artifact, line_number
                    )
                    written = artifact["stanzas"][stanza - 1][
                        line_in_stanza - 1
                    ]
                    named = [
                        finding
                        for finding in matched
                        if finding["written"]["stanza"] == stanza
                        and finding["written"]["line_in_stanza"]
                        == line_in_stanza
                        and finding["written"]["line"] == global_line
                        and finding["written"]["global_line"] == global_line
                    ]
                    self.assertTrue(
                        named,
                        "expected a line_duplicated naming stanza %d "
                        "line %d (global %d), got %r"
                        % (stanza, line_in_stanza, global_line, matched),
                    )
                    self.assertEqual(named[0]["delivered"]["count"], 2)
                    self.assertEqual(named[0]["written"]["text"], written)
                    self.assertIn("appears 2 times", named[0]["statement"])
                else:
                    first = fault_by_kind["swap_lines"].get("first")
                    second = fault_by_kind["swap_lines"].get("second")
                    first_stanza, first_in_stanza, first_global = _locate(
                        artifact, first
                    )
                    second_stanza, second_in_stanza, second_global = _locate(
                        artifact, second
                    )
                    first_written = artifact["stanzas"][first_stanza - 1][
                        first_in_stanza - 1
                    ]
                    second_written = artifact["stanzas"][second_stanza - 1][
                        second_in_stanza - 1
                    ]
                    named = []
                    for finding in matched:
                        written = finding["written"]
                        if (
                            written.get("first_stanza") == first_stanza
                            and written.get("first_line_in_stanza")
                            == first_in_stanza
                            and written.get("first_global_line")
                            == first_global
                            and written.get("second_stanza") == second_stanza
                            and written.get("second_line_in_stanza")
                            == second_in_stanza
                            and written.get("second_global_line")
                            == second_global
                            and written.get("first_text") == first_written
                            and written.get("second_text") == second_written
                        ):
                            named.append(finding)
                    self.assertTrue(
                        named,
                        "expected a lines_swapped naming stanza %d "
                        "line %d and stanza %d line %d, got %r"
                        % (
                            first_stanza,
                            first_in_stanza,
                            second_stanza,
                            second_in_stanza,
                            matched,
                        ),
                    )
                    statement = named[0]["statement"]
                    self.assertIn(first_written, statement)
                    self.assertIn(second_written, statement)

    def test_clean_delivery_produces_zero_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _poem()
            comparison, record, evidence = _compare(artifact, tmpdir, ())
            self.assertEqual(comparison["findings"], [])
            self.assertEqual(comparison["summary"]["total"], 0)
            self.assertEqual(comparison["summary"]["counts"], {})
            self.assertIn("No findings", comparison["summary"]["line"])
            evidence_with_comparison = copy.deepcopy(evidence)
            evidence_with_comparison["comparison"] = comparison
            coverage = objective_coverage(
                artifact, record, evidence_with_comparison
            )
            self.assertEqual(
                coverage,
                [
                    {
                        "item": "lines delivered as written, 4 of 4",
                        "covered": True,
                    },
                    {
                        "item": "stanzas complete, 2 of 2",
                        "covered": True,
                    },
                ],
            )

    def test_altered_line_produces_line_text_differs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _poem()
            original_line = artifact["stanzas"][1][0]
            altered_line = "The harbor lights the silver bay"
            rendered_artifact = copy.deepcopy(artifact)
            rendered_artifact["stanzas"][1][0] = altered_line

            # The written poem is artifact; the delivered text
            # is rendered from a copy that differs in one line,
            # so the delivery is complete in count but one
            # line's text is altered.
            path = os.path.join(tmpdir, "altered.txt")
            record = renderer(rendered_artifact, path, [])
            evidence = perceive(record)
            comparison = compute_comparison(artifact, record, evidence)

            differs = _findings(comparison, "line_text_differs")
            named = [
                finding
                for finding in differs
                if finding["written"]["text"] == original_line
                and finding["delivered"]["text"] == altered_line
                and finding["written"]["stanza"] == 2
                and finding["written"]["line_in_stanza"] == 1
            ]
            self.assertTrue(
                named,
                "expected a line_text_differs naming both texts, "
                "got %r" % (differs,),
            )
            statement = named[0]["statement"]
            self.assertIn(original_line, statement)
            self.assertIn(altered_line, statement)
            self.assertEqual(named[0]["written"]["text"], original_line)
            self.assertEqual(named[0]["delivered"]["text"], altered_line)

            evidence_with_comparison = copy.deepcopy(evidence)
            evidence_with_comparison["comparison"] = comparison
            coverage = objective_coverage(
                artifact, record, evidence_with_comparison
            )
            self.assertEqual(
                coverage,
                [
                    {
                        "item": "lines delivered as written, 3 of 4",
                        "covered": False,
                    },
                    {
                        "item": "stanzas complete, 1 of 2",
                        "covered": False,
                    },
                ],
            )

    def test_form_off_names_syllable_estimate_against_the_forms_number(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = {
                "stanzas": [
                    # A short written line meeting the stated
                    # form's syllables per line, and a longer
                    # written line whose estimate departs.
                    ["The brook runs down", "A thrush repeats the hidden spring by the water"],
                ]
            }
            stated = 4
            artifact["form"] = {
                "line_count": 2,
                "stanzas": 1,
                "syllables_per_line": stated,
            }
            estimates = [
                sum(syllable_estimate(word) for word in line.split())
                for stanza in artifact["stanzas"]
                for line in stanza
            ]
            self.assertEqual(
                estimates[0],
                stated,
                "the first line must meet the stated form",
            )
            self.assertNotEqual(
                estimates[1],
                stated,
                "a written line must depart from the stated form",
            )
            comparison, _record, _evidence = _compare(
                artifact, tmpdir, (), label="form"
            )
            off = _findings(comparison, "form_off")
            syllable_off = [
                finding
                for finding in off
                if finding["where"] not in ("line count", "stanza count")
                and "syllable estimate" in finding["statement"]
                and finding["written"]["syllables_per_line"] == stated
                and finding["delivered"]["syllable_estimate"] == estimates[1]
            ]
            self.assertTrue(
                syllable_off,
                "expected a form_off naming a syllable estimate "
                "against the form's number, got %r" % (off,),
            )
            for finding in syllable_off:
                self.assertIn("an estimate", finding["statement"])
                self.assertEqual(
                    finding["delivered"]["syllable_estimate"], estimates[1]
                )
                self.assertEqual(
                    finding["written"]["syllables_per_line"], stated
                )


if __name__ == "__main__":
    unittest.main()
