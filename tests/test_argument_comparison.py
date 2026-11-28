"""tests/test_argument_comparison.py - comparison judgment tests
(argument).

A clean argument returns no findings and its fact claims read
supported or unverified as cited; a dropped premise is caught as
premise_dropped and claim_missing; a swapped date in a fact claim
is caught as claim_altered and its source check reads contradicted
or not_present; a misquote is caught as source_misquoted; a claim
with its support removed is caught as claim_unsupported; a circular
conclusion is caught as conclusion_restates_premise; an uncited
fact claim is marked unverified and produces no falsehood finding;
a revision can repair an unsupported claim by setting its
premises, or a source-uncovered fact claim by setting its key
phrase, and a revision naming an unknown premise or source id is
rejected leaving the argument unchanged.

No model calls are made.
"""

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.apps.argument.app import apply_revision, perceive
from stage2.apps.argument.comparison import compute_comparison
from stage2.apps.argument.invariants import objective_coverage
from argument_renderer import renderer


def _argument():
    return {
        "objective": {"claim_count": 4, "kinds": ["fact", "logic", "value"]},
        "title": "The engine and the law",
        "claims": [
            {
                "id": "c1",
                "kind": "fact",
                "text": "The engine was commissioned on 2020-03-15.",
                "key_phrase": "engine was commissioned",
                "premises": [],
                "sources": ["s1"],
                "quote": "commissioned on 2020-03-15",
            },
            {
                "id": "c2",
                "kind": "logic",
                "text": "A commission is the act of putting a work into service.",
                "key_phrase": "act of putting a work into service",
                "premises": ["c1"],
                "sources": [],
            },
            {
                "id": "c3",
                "kind": "value",
                "text": "The commission should be recorded in the public register.",
                "key_phrase": "recorded in the public register",
                "premises": ["c1", "c2"],
                "sources": [],
            },
            {
                "id": "c4",
                "kind": "logic",
                "text": "The register therefore holds the true commission date.",
                "key_phrase": "register holds the true commission date",
                "premises": ["c1", "c2", "c3"],
                "sources": [],
            },
        ],
        "definitions": [
            {"term": "commission", "definition": "the act of putting a work into service"},
        ],
        "sources": [
            {"id": "s1", "label": "Register entry", "passage": "The engine was commissioned on 2020-03-15, the act of putting a work into service."},
        ],
        "conclusion": "c4",
        "text": (
            "The engine was commissioned on 2020-03-15. "
            "A commission is the act of putting a work into service. "
            "The commission should be recorded in the public register. "
            "The register holds the true commission date."
        ),
    }


def _compare(artifact, tmpdir, faults=(), label="render"):
    path = os.path.join(tmpdir, label + ".txt")
    record = renderer(artifact, path, list(faults))
    record["claims"] = copy.deepcopy(artifact["claims"])
    record["sources"] = copy.deepcopy(artifact["sources"])
    record["definitions"] = copy.deepcopy(artifact["definitions"])
    evidence = perceive(record)
    return compute_comparison(artifact, record, evidence), record, evidence


def _findings(comparison, kind):
    return [finding for finding in comparison["findings"] if finding["kind"] == kind]


def _checks_by_id(comparison):
    return {check["claim_id"]: check for check in comparison["source_checks"]}


def _coverage_with_comparison(artifact, record, evidence, comparison):
    evidence_with_comparison = copy.deepcopy(evidence)
    record_with_comparison = copy.deepcopy(record)
    evidence_with_comparison["comparison"] = comparison
    record_with_comparison["comparison"] = comparison
    return objective_coverage(artifact, record_with_comparison, evidence_with_comparison)


class ArgumentComparisonTests(unittest.TestCase):
    def test_clean_argument_returns_no_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _argument()
            comparison, record, evidence = _compare(artifact, tmpdir, faults=())
            self.assertEqual(comparison["findings"], [])
            self.assertEqual(comparison["summary"]["total"], 0)
            self.assertIn("No findings", comparison["summary"]["line"])
            checks = _checks_by_id(comparison)
            self.assertEqual(checks["c1"]["status"], "supported")
            coverage = _coverage_with_comparison(artifact, record, evidence, comparison)
            self.assertIn(
                {"item": "claims delivered, 4 of 4", "covered": True},
                coverage,
            )
            self.assertIn(
                {"item": "fact claims checked, 1 of 1", "covered": True},
                coverage,
            )
            self.assertIn(
                {"item": "conclusion delivered, yes", "covered": True},
                coverage,
            )

    def test_dropped_premise_is_caught_as_premise_dropped_and_claim_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _argument()
            comparison, _record, _evidence = _compare(
                artifact, tmpdir, faults=({"kind": "drop_claim", "claim": "c1"},),
                label="dropped-premise",
            )
            missing = _findings(comparison, "claim_missing")
            named = [
                finding
                for finding in missing
                if finding["written"].get("claim_id") == "c1"
                or "engine was commissioned" in finding["statement"]
            ]
            self.assertTrue(
                named,
                "expected a claim_missing naming c1, got %r" % (comparison["findings"],),
            )
            dropped = _findings(comparison, "premise_dropped")
            named_dropped = [
                finding
                for finding in dropped
                if finding["written"].get("premise_claim_id") == "c1"
            ]
            self.assertTrue(
                named_dropped,
                "expected a premise_dropped naming c1, got %r" % (comparison["findings"],),
            )
            finding = named_dropped[0]
            self.assertIn("c1", finding["statement"])
            self.assertEqual(finding["written"].get("premise_claim_id"), "c1")
            self.assertIn("commissioned on 2020-03-15", finding["written"].get("premise_text", ""))
            for key in ("kind", "where", "written", "delivered", "statement"):
                self.assertIn(key, finding)

    def test_swapped_date_is_caught_as_claim_altered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _argument()
            comparison, _record, _evidence = _compare(
                artifact, tmpdir, faults=({"kind": "swap_date", "claim": "c1"},),
                label="swapped-date",
            )
            matched = _findings(comparison, "claim_altered")
            named = [
                finding
                for finding in matched
                if finding["written"].get("claim_id") == "c1"
            ]
            self.assertTrue(
                named,
                "expected a claim_altered naming c1, got %r" % (comparison["findings"],),
            )
            finding = named[0]
            self.assertIn("2020-03-15", finding["written"].get("text", ""))
            delivered = finding["delivered"].get("delivered_statement", "")
            self.assertNotIn("2020-03-15", delivered)
            checks = _checks_by_id(comparison)
            self.assertIn(checks["c1"]["status"], ("contradicted", "not_present"))

    def test_misquote_is_caught_as_source_misquoted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _argument()
            artifact["claims"][0]["quote"] = "commissioned on 2020-03-16"
            comparison, _record, _evidence = _compare(artifact, tmpdir, label="misquote")
            matched = _findings(comparison, "source_misquoted")
            named = [
                finding
                for finding in matched
                if finding["written"].get("claim_id") == "c1"
                or "2020-03-16" in finding["statement"]
            ]
            self.assertTrue(
                named,
                "expected a source_misquoted naming the quote, got %r" % (comparison["findings"],),
            )
            finding = named[0]
            self.assertEqual(finding["written"].get("quote"), "commissioned on 2020-03-16")
            self.assertIn("2020-03-15", finding["delivered"].get("passage", ""))

    def test_unsupported_claim_is_caught_as_claim_unsupported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _argument()
            artifact["claims"][1]["premises"] = []
            comparison, _record, _evidence = _compare(artifact, tmpdir, label="unsupported")
            matched = _findings(comparison, "claim_unsupported")
            named = [
                finding
                for finding in matched
                if finding["written"].get("claim_id") == "c2"
            ]
            self.assertTrue(
                named,
                "expected a claim_unsupported naming c2, got %r" % (comparison["findings"],),
            )
            finding = named[0]
            self.assertIn("nothing under it", finding["statement"])

    def test_circular_conclusion_is_caught_as_conclusion_restates_premise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _argument()
            artifact["claims"][3]["text"] = artifact["claims"][0]["text"]
            comparison, _record, _evidence = _compare(artifact, tmpdir, label="circular")
            matched = _findings(comparison, "conclusion_restates_premise")
            self.assertTrue(
                matched,
                "expected a conclusion_restates_premise, got %r" % (comparison["findings"],),
            )
            finding = matched[0]
            self.assertIn("c4", finding["statement"])
            self.assertIn("0.9", finding["statement"])

    def test_uncited_fact_claim_is_unverified_and_no_falsehood_finding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _argument()
            artifact["claims"][0]["sources"] = []
            comparison, _record, _evidence = _compare(artifact, tmpdir, label="uncited")
            checks = _checks_by_id(comparison)
            self.assertEqual(checks["c1"]["status"], "unverified")
            falsehood = [
                finding
                for finding in comparison["findings"]
                if finding["kind"] in ("source_contradicts_claim", "source_does_not_cover_claim")
            ]
            self.assertEqual(falsehood, [])

    def test_revision_setting_premises_clears_claim_unsupported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _argument()
            artifact["claims"][1]["premises"] = []
            artifact["claims"][1]["sources"] = []
            comparison, _record, _evidence = _compare(artifact, tmpdir, label="unsupported-before")
            before = [
                finding
                for finding in _findings(comparison, "claim_unsupported")
                if finding["written"].get("claim_id") == "c2"
            ]
            self.assertTrue(
                before,
                "expected a claim_unsupported naming c2, got %r" % (comparison["findings"],),
            )
            revised = apply_revision(
                artifact,
                {
                    "op": "replace_claim",
                    "id": "c2",
                    "text": "A commission is the act of putting a work into service.",
                    "premises": ["c1"],
                },
            )
            self.assertNotEqual(revised, artifact)
            self.assertEqual(revised["claims"][1]["premises"], ["c1"])
            comparison_after, _record, _evidence = _compare(revised, tmpdir, label="unsupported-after")
            after = [
                finding
                for finding in _findings(comparison_after, "claim_unsupported")
                if finding["written"].get("claim_id") == "c2"
            ]
            self.assertEqual(
                after,
                [],
                "expected no claim_unsupported for c2 after revision, got %r" % (comparison_after["findings"],),
            )

    def test_revision_setting_key_phrase_clears_source_does_not_cover_claim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = _argument()
            # A key phrase that is present in the delivered text (the
            # register sentence) but absent from c1's cited passage
            # s1, with no figure shared between the delivered
            # statement and the passage, draws the finding.  That
            # phrase is the one comparison.py flags: key phrase
            # absent from the passage and no figure shared.
            artifact["claims"][0]["key_phrase"] = "recorded in the public register"
            comparison, _record, _evidence = _compare(artifact, tmpdir, label="uncovered-before")
            before = [
                finding
                for finding in _findings(comparison, "source_does_not_cover_claim")
                if finding["written"].get("claim_id") == "c1"
            ]
            self.assertTrue(
                before,
                "expected a source_does_not_cover_claim naming c1, got %r" % (comparison["findings"],),
            )
            self.assertIn("s1", before[0]["statement"])
            self.assertIn("recorded in the public register", before[0]["written"].get("key_phrase", ""))
            revised = apply_revision(
                artifact,
                {
                    "op": "change_claim",
                    "id": "c1",
                    "text": "The engine was commissioned on 2020-03-15.",
                    "key_phrase": "engine was commissioned",
                },
            )
            self.assertEqual(revised["claims"][0]["key_phrase"], "engine was commissioned")
            self.assertEqual(revised["claims"][0]["sources"], ["s1"])
            comparison_after, _record, _evidence = _compare(revised, tmpdir, label="uncovered-after")
            after = [
                finding
                for finding in _findings(comparison_after, "source_does_not_cover_claim")
                if finding["written"].get("claim_id") == "c1"
            ]
            self.assertEqual(
                after,
                [],
                "expected no source_does_not_cover_claim for c1 after revision, got %r" % (comparison_after["findings"],),
            )

    def test_revision_naming_unknown_premise_or_source_is_rejected(self):
        artifact = _argument()
        before = copy.deepcopy(artifact)
        for revision in (
            {
                "op": "replace_claim",
                "id": "c2",
                "text": "A commission is the act of putting a work into service.",
                "premises": ["no-such-claim"],
            },
            {
                "op": "change_claim",
                "id": "c1",
                "text": "The engine was commissioned on 2020-03-15.",
                "sources": ["no-such-source"],
            },
        ):
            with self.assertRaises(ValueError, msg=str(revision)):
                apply_revision(artifact, revision)
            self.assertEqual(artifact, before)


if __name__ == "__main__":
    unittest.main()
