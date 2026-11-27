"""tests/test_code_comparison.py - comparison judgment tests (code).

A delivered program that fails one stated check produces
check_failed naming the check with its expected and actual
outcomes; a program that raises inside a check produces
check_errored; and a clean program produces zero findings,
with coverage counting the checks passed of the checks the
objective states.

Execution is through the application's own instrument and
renderer, as in the other comparison suites.  No model calls
are made.
"""

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.apps.code.app import perceive
from stage2.apps.code.comparison import compute_comparison
from stage2.apps.code.invariants import objective_coverage
from code_renderer import renderer


def _program(text, tests):
    return {
        "files": [{"name": "main.py", "text": text}],
        "purpose": "echo the fed input to standard output",
        "tests": tests,
    }


def _clean_program():
    return _program(
        "import sys\nprint(sys.stdin.read().strip())\n",
        [
            {"name": "echo_hello", "input": "hello", "expected": "hello\n"},
            {"name": "echo_world", "input": "world", "expected": "world\n"},
        ],
    )


def _compare(artifact, tmpdir, delivered_artifact=None, faults=(), label="render"):
    source = copy.deepcopy(delivered_artifact) if delivered_artifact is not None else artifact
    path = os.path.join(tmpdir, label + ".py")
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


class CodeComparisonTests(unittest.TestCase):
    def test_failing_check_produces_check_failed_naming_check_with_expected_and_actual(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _clean_program()
            delivered = copy.deepcopy(artifact)
            delivered["files"][0]["text"] = (
                "import sys\n"
                "data = sys.stdin.read().strip()\n"
                "print(data.upper() if data == 'world' else data)\n"
            )
            comparison, _record, _evidence = _compare(
                artifact, tmpdir, delivered_artifact=delivered
            )
            matched = _findings(comparison, "check_failed")
            named = [
                finding
                for finding in matched
                if finding["written"].get("name") == "echo_world"
                or finding["written"].get("check") == "echo_world"
                or "echo_world" in finding["statement"]
                or "echo_world" in finding["where"]
            ]
            self.assertTrue(
                named,
                "expected a check_failed naming echo_world, got %r"
                % (comparison["findings"],),
            )
            finding = named[0]
            self.assertIn("echo_world", finding["statement"])
            self.assertEqual(finding["written"].get("expected"), "world\n")
            self.assertEqual(finding["written"].get("name"), "echo_world")
            self.assertEqual(finding["delivered"].get("stdout"), "WORLD\n")
            self.assertIn("world", finding["statement"])
            self.assertIn("WORLD", finding["statement"])
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertIn("check_failed", comparison["summary"]["counts"])
            self.assertEqual(comparison["summary"]["counts"]["check_failed"], 1)
            self.assertEqual(comparison["checks_written"], 2)
            self.assertEqual(comparison["checks_passed"], 1)

    def test_raising_program_produces_check_errored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _clean_program()
            delivered = copy.deepcopy(artifact)
            delivered["files"][0]["text"] = (
                "import sys\n"
                "raise ValueError('boom')\n"
            )
            comparison, _record, _evidence = _compare(
                artifact, tmpdir, delivered_artifact=delivered
            )
            matched = _findings(comparison, "check_errored")
            named = [
                finding
                for finding in matched
                if finding["written"].get("name") == "echo_hello"
                or finding["written"].get("check") == "echo_hello"
                or "echo_hello" in finding["statement"]
            ]
            self.assertTrue(
                named,
                "expected a check_errored naming echo_hello, got %r"
                % (comparison["findings"],),
            )
            finding = named[0]
            self.assertIn("echo_hello", finding["statement"])
            self.assertIn("ValueError", finding["statement"])
            self.assertIn("boom", finding["statement"])
            self.assertIn("ValueError", finding["delivered"].get("stderr", ""))
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)
            self.assertIn("check_errored", comparison["summary"]["counts"])
            self.assertEqual(comparison["checks_passed"], 0)

    def test_clean_program_produces_zero_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _clean_program()
            comparison, record, evidence = _compare(artifact, tmpdir)
            self.assertEqual(comparison["findings"], [])
            self.assertEqual(comparison["summary"]["total"], 0)
            self.assertEqual(comparison["summary"]["counts"], {})
            self.assertIn("No findings", comparison["summary"]["line"])
            self.assertEqual(comparison["checks_written"], 2)
            self.assertEqual(comparison["checks_passed"], 2)
            coverage = _coverage_with_comparison(artifact, record, evidence, comparison)
            self.assertEqual(
                coverage,
                [
                    {
                        "item": "checks passed, 2 of 2",
                        "covered": True,
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()
