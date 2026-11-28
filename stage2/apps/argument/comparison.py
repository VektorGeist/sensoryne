"""stage2/apps/argument/comparison.py - Sensoryne's judgment of the
delivered text against the claim map.

The comparison works from the text analyzer's own analysis of the
delivered text (the evidence measurements for the delivered
claims, figures, definitions, and quotations, or the delivered
text itself), set against the written argument as the artifact
and its structure record state it: the artifact's claims in
order, their kind, premises, and cited sources, the definitions,
and the conclusion aimed at.  It never works from delivery-fault
definitions: a finding names what the delivered text shows
against what was written.  For fact claims it checks the claim
against the source the argument itself cites: supported,
contradicted, or not present, with the source's passage quoted
in the evidence.  A claim with no cited source is marked
unverified, never false.  No web search, no outside finder,
nothing reaches outside the artifact and its cited passages.

Each finding is an object with kind, where, written, delivered,
and statement.  The kinds are claim_missing, claim_altered,
claim_unsupported, premise_dropped, term_defined_otherwise,
conclusion_restates_premise, source_contradicts_claim,
source_does_not_cover_claim, and source_misquoted.
"""

import copy
import os
import re

from .app import structure_record
from .instrument import analyze_text, extract_figures

__all__ = [
    "CIRCULARITY_THRESHOLD",
    "FINDING_KINDS",
    "compute_comparison",
    "build_comparison",
]

CIRCULARITY_THRESHOLD = 0.9

FINDING_KINDS = (
    "claim_missing",
    "claim_altered",
    "claim_unsupported",
    "premise_dropped",
    "term_defined_otherwise",
    "conclusion_restates_premise",
    "source_contradicts_claim",
    "source_does_not_cover_claim",
    "source_misquoted",
)


def _finding(kind, where, written, delivered, statement):
    return {
        "kind": kind,
        "where": where,
        "written": copy.deepcopy(written),
        "delivered": copy.deepcopy(delivered),
        "statement": statement,
    }


def _summary(findings):
    counts = {}
    for finding in findings:
        counts[finding["kind"]] = counts.get(finding["kind"], 0) + 1
    total = len(findings)
    if total == 0:
        line = (
            "No findings: the delivered text matches the claim map "
            "within the stated tolerances."
        )
    else:
        line = "%d finding(s): %s." % (
            total,
            ", ".join(
                "%s=%d" % (kind, counts[kind]) for kind in sorted(counts)
            ),
        )
    return {"counts": counts, "total": total, "line": line}


def _words_set(text):
    return set(re.findall(r"[a-z0-9']+", str(text or "").lower()))


def _normalize(text):
    return " ".join(re.findall(r"[a-z0-9']+", str(text or "").lower()))


def _overlap(a, b):
    set_a, set_b = _words_set(a), _words_set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / float(len(set_a | set_b))


def _claims_of(artifact):
    if isinstance(artifact, dict) and isinstance(artifact.get("claims"), list):
        return [c for c in artifact["claims"] if isinstance(c, dict)]
    return []


def _sources_by_id(artifact):
    by_id = {}
    if isinstance(artifact, dict) and isinstance(artifact.get("sources"), list):
        for source in artifact["sources"]:
            if isinstance(source, dict) and source.get("id") is not None:
                by_id[source.get("id")] = source
    return by_id


def _measurements_of(source):
    for candidate in (source,):
        if isinstance(candidate, dict) and isinstance(candidate.get("measurements"), dict):
            return candidate["measurements"]
        if isinstance(candidate, dict) and isinstance(candidate.get("full"), dict):
            full = candidate["full"]
            if isinstance(full.get("measurements"), dict):
                return full["measurements"]
    return None


def _delivered_text_of(evidence, render_record):
    for source in (evidence, render_record):
        if not isinstance(source, dict):
            continue
        if isinstance(source.get("delivered_text"), str):
            return source["delivered_text"]
        full = source.get("full")
        if isinstance(full, dict) and isinstance(full.get("delivered_text"), str):
            return full["delivered_text"]
    return None


def _delivered_claims(render_record, evidence, artifact):
    """Return (measurements_per_claim, delivered_text, measurements)."""
    if evidence is None and isinstance(render_record, dict):
        if "measurements" in render_record or "delivered_text" in render_record:
            evidence = render_record
            render_record = None

    delivered_text = _delivered_text_of(evidence, render_record)
    measurements = None
    for source in (evidence, render_record):
        if isinstance(source, dict):
            meas = _measurements_of(source)
            if meas is not None:
                measurements = meas
                break

    claims = _claims_of(artifact)
    sources = []
    definitions = []
    if isinstance(artifact, dict):
        if isinstance(artifact.get("sources"), list):
            sources = artifact["sources"]
        if isinstance(artifact.get("definitions"), list):
            definitions = artifact["definitions"]

    if isinstance(measurements, dict) and isinstance(measurements.get("claims"), list) and measurements["claims"]:
        return measurements["claims"], delivered_text, measurements

    if delivered_text is None:
        path = None
        if isinstance(render_record, dict):
            candidate = render_record.get("path")
            if isinstance(candidate, str) and os.path.isfile(candidate):
                path = candidate
        if path is not None:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    delivered_text = handle.read()
            except Exception:
                delivered_text = None

    if delivered_text is not None:
        analysis = analyze_text(delivered_text, claims, sources, definitions)
        return analysis["measurements"]["claims"], delivered_text, analysis["measurements"]
    return [], None, measurements


def compute_comparison(artifact, render_record=None, evidence=None):
    """Compare the delivered text against the claim map."""
    if evidence is None and isinstance(render_record, dict):
        if "measurements" in render_record or "delivered_text" in render_record:
            evidence = render_record
            render_record = None

    claims = _claims_of(artifact)
    by_id = {claim.get("id"): claim for claim in claims}
    sources_by_id = _sources_by_id(artifact)
    conclusion_id = artifact.get("conclusion") if isinstance(artifact, dict) else None

    delivered_claims, _delivered_text, measurements = _delivered_claims(
        render_record, evidence, artifact
    )
    delivered_by_id = {}
    for entry in delivered_claims:
        if isinstance(entry, dict):
            delivered_by_id[entry.get("claim_id")] = entry

    findings = []

    # claim_missing: a claim whose key phrase does not appear in
    # the delivered text.
    for claim in claims:
        entry = delivered_by_id.get(claim.get("id"))
        present = bool(entry.get("present")) if isinstance(entry, dict) else False
        if present:
            continue
        where = "claim %s (%s)" % (claim.get("id"), claim.get("kind"))
        findings.append(_finding(
            "claim_missing",
            where,
            {
                "claim_id": claim.get("id"),
                "kind": claim.get("kind"),
                "text": claim.get("text"),
                "key_phrase": claim.get("key_phrase"),
            },
            {
                "claim_id": claim.get("id"),
                "kind": claim.get("kind"),
                "key_phrase": claim.get("key_phrase"),
                "present": False,
                "delivered_statement": None,
            },
            "Claim %s (%s) text %r, key phrase %r, does not appear in the delivered text." % (
                claim.get("id"), claim.get("kind"), claim.get("text"), claim.get("key_phrase")
            ),
        ))

    # claim_altered: the delivered statement of a fact claim
    # carries a figure that differs from the figure in the
    # claim's written text; both quoted.
    for claim in claims:
        if claim.get("kind") != "fact":
            continue
        entry = delivered_by_id.get(claim.get("id"))
        if not isinstance(entry, dict) or not entry.get("present"):
            continue
        delivered_statement = entry.get("delivered_statement")
        claim_figures = list(entry.get("claim_figures") or [])
        delivered_figures = list(entry.get("delivered_figures") or [])
        differing = [fig for fig in delivered_figures if fig not in claim_figures]
        if not differing:
            continue
        where = "claim %s (%s)" % (claim.get("id"), claim.get("kind"))
        findings.append(_finding(
            "claim_altered",
            where,
            {
                "claim_id": claim.get("id"),
                "kind": claim.get("kind"),
                "text": claim.get("text"),
                "figures": claim_figures,
            },
            {
                "claim_id": claim.get("id"),
                "kind": claim.get("kind"),
                "delivered_statement": delivered_statement,
                "figures": delivered_figures,
                "differing_figures": differing,
            },
            "Claim %s's delivered statement carries figure(s) %s differing from the figure(s) %s in the claim's written text: written %r, delivered %r." % (
                claim.get("id"),
                ", ".join(repr(f) for f in differing),
                ", ".join(repr(f) for f in claim_figures),
                claim.get("text"),
                delivered_statement,
            ),
        ))

    # claim_unsupported: a claim with an empty premises list and
    # an empty sources list; nothing stands under it.
    for claim in claims:
        if claim.get("premises") or claim.get("sources"):
            continue
        where = "claim %s (%s)" % (claim.get("id"), claim.get("kind"))
        findings.append(_finding(
            "claim_unsupported",
            where,
            {
                "claim_id": claim.get("id"),
                "kind": claim.get("kind"),
                "text": claim.get("text"),
                "premises": [],
                "sources": [],
            },
            {
                "claim_id": claim.get("id"),
                "kind": claim.get("kind"),
                "premises": [],
                "sources": [],
            },
            "Claim %s (%s) text %r asserts with nothing under it: empty premises and empty sources." % (
                claim.get("id"), claim.get("kind"), claim.get("text")
            ),
        ))

    # premise_dropped: a claim whose stated premise claim is
    # missing from the delivered text (its key phrase absent).
    for claim in claims:
        for premise_id in claim.get("premises") or []:
            premise = by_id.get(premise_id)
            if premise is None:
                continue
            entry = delivered_by_id.get(premise_id)
            present = bool(entry.get("present")) if isinstance(entry, dict) else False
            if present:
                continue
            where = "claim %s (%s)" % (claim.get("id"), claim.get("kind"))
            findings.append(_finding(
                "premise_dropped",
                where,
                {
                    "claim_id": claim.get("id"),
                    "kind": claim.get("kind"),
                    "text": claim.get("text"),
                    "premise_claim_id": premise_id,
                    "premise_text": premise.get("text"),
                    "premise_key_phrase": premise.get("key_phrase"),
                },
                {
                    "claim_id": claim.get("id"),
                    "kind": claim.get("kind"),
                    "premise_claim_id": premise_id,
                    "premise_present": False,
                    "premise_key_phrase": premise.get("key_phrase"),
                },
                "Claim %s's stated premise %s (text %r, key phrase %r) is missing from the delivered text." % (
                    claim.get("id"), premise_id, premise.get("text"), premise.get("key_phrase")
                ),
            ))

    # term_defined_otherwise: the delivered text states a
    # definition of a defined term that differs from the claim
    # map's definition; both definitions quoted.
    definitions = artifact.get("definitions") if isinstance(artifact, dict) else None
    if isinstance(definitions, list):
        checks = []
        if isinstance(measurements, dict) and isinstance(measurements.get("definition_checks"), list):
            checks = measurements["definition_checks"]
        for definition in definitions:
            if not isinstance(definition, dict):
                continue
            term = definition.get("term") or ""
            written = definition.get("definition") or ""
            delivered = None
            for check in checks:
                if isinstance(check, dict) and str(check.get("term", "")).strip().lower() == str(term).strip().lower():
                    delivered = check.get("delivered_definition")
                    break
            if not isinstance(delivered, str) or not delivered.strip():
                continue
            if _normalize(delivered) == _normalize(written):
                continue
            where = "term %r" % (term,)
            findings.append(_finding(
                "term_defined_otherwise",
                where,
                {"term": term, "definition": written},
                {"term": term, "definition": delivered, "delivered_definition": delivered},
                "Term %r is defined in the claim map as %r but the delivered text states %r." % (
                    term, written, delivered
                ),
            ))

    # conclusion_restates_premise: the conclusion claim's text
    # restates one of its own premises (normalized text equality
    # or word-set overlap at a stated threshold of 0.9, the
    # threshold named in the reliability block); the argument
    # argues in a circle.
    conclusion = by_id.get(conclusion_id)
    if isinstance(conclusion, dict):
        for premise_id in conclusion.get("premises") or []:
            premise = by_id.get(premise_id)
            if not isinstance(premise, dict):
                continue
            same = _normalize(conclusion.get("text", "")) == _normalize(premise.get("text", ""))
            overlap = _overlap(conclusion.get("text", ""), premise.get("text", ""))
            if same or overlap >= CIRCULARITY_THRESHOLD:
                where = "claim %s (%s)" % (conclusion.get("id"), conclusion.get("kind"))
                findings.append(_finding(
                    "conclusion_restates_premise",
                    where,
                    {
                        "claim_id": conclusion.get("id"),
                        "kind": conclusion.get("kind"),
                        "text": conclusion.get("text"),
                        "premise_claim_id": premise_id,
                        "premise_text": premise.get("text"),
                    },
                    {
                        "claim_id": conclusion.get("id"),
                        "kind": conclusion.get("kind"),
                        "text": conclusion.get("text"),
                        "premise_claim_id": premise_id,
                        "premise_text": premise.get("text"),
                        "overlap": overlap,
                        "circularity_threshold": CIRCULARITY_THRESHOLD,
                    },
                    "Conclusion %s restates its own premise %s: text %r restates premise text %r (word-set overlap %s at the %s circularity threshold)." % (
                        conclusion.get("id"), premise_id, conclusion.get("text"), premise.get("text"), overlap, CIRCULARITY_THRESHOLD
                    ),
                ))

    # Source checks: per fact claim, the delivered statement
    # against each source the claim itself cites. A claim with
    # no cited source is unverified, never false, and never a
    # finding of falsehood.
    source_checks = []
    for claim in claims:
        if claim.get("kind") != "fact":
            continue
        cited_ids = list(claim.get("sources") or [])
        if not cited_ids:
            source_checks.append({
                "claim_id": claim.get("id"),
                "status": "unverified",
                "sources": [],
                "passages": [],
            })
            continue
        entry = delivered_by_id.get(claim.get("id"))
        if isinstance(entry, dict) and entry.get("present") and entry.get("delivered_statement"):
            delivered_figures = list(entry.get("delivered_figures") or [])
        else:
            delivered_figures = extract_figures(claim.get("text", ""))
        passages = []
        for source_id in cited_ids:
            source = sources_by_id.get(source_id)
            passage = source.get("passage") if isinstance(source, dict) else ""
            if not isinstance(passage, str):
                passage = ""
            passages.append({
                "source_id": source_id,
                "label": source.get("label") if isinstance(source, dict) else None,
                "passage": passage,
                "passage_figures": extract_figures(passage),
                "contains_key_phrase": bool(claim.get("key_phrase")) and str(claim.get("key_phrase")).lower() in passage.lower(),
            })

        union_figures = set()
        for cited in passages:
            union_figures.update(cited["passage_figures"])

        contradicted_passages = []
        uncovered_passages = []
        for cited in passages:
            passage_figures = list(cited["passage_figures"])
            differs = [fig for fig in delivered_figures if fig not in passage_figures]
            differs_in_passage = [fig for fig in passage_figures if fig not in delivered_figures]
            if differs and differs_in_passage:
                contradicted_passages.append(cited)
                continue
            shares = bool(set(delivered_figures) & set(passage_figures))
            if not cited["contains_key_phrase"] and not shares:
                uncovered_passages.append(cited)

        if contradicted_passages:
            status = "contradicted"
        elif (
            any(fig not in union_figures for fig in delivered_figures)
            or (not any(cited["contains_key_phrase"] for cited in passages)
                and not (set(delivered_figures) & union_figures))
        ):
            status = "not_present"
        else:
            status = "supported"

        source_checks.append({
            "claim_id": claim.get("id"),
            "status": status,
            "sources": cited_ids,
            "passages": [
                {
                    "source_id": cited["source_id"],
                    "label": cited["label"],
                    "passage": cited["passage"],
                    "passage_figures": cited["passage_figures"],
                }
                for cited in passages
            ],
        })

        # source_contradicts_claim: a cited source's passage
        # states a different figure for the same measured thing
        # the claim states; the passage quoted.
        for cited in contradicted_passages:
            passage_figures = list(cited["passage_figures"])
            differs = [fig for fig in delivered_figures if fig not in passage_figures]
            differs_in_passage = [fig for fig in passage_figures if fig not in delivered_figures]
            where = "claim %s (%s)" % (claim.get("id"), claim.get("kind"))
            findings.append(_finding(
                "source_contradicts_claim",
                where,
                {
                    "claim_id": claim.get("id"),
                    "kind": claim.get("kind"),
                    "text": claim.get("text"),
                    "figures": extract_figures(claim.get("text", "")),
                    "delivered_figures": delivered_figures,
                    "source_id": cited["source_id"],
                    "source_label": cited["label"],
                },
                {
                    "claim_id": claim.get("id"),
                    "kind": claim.get("kind"),
                    "delivered_statement": entry.get("delivered_statement") if isinstance(entry, dict) else None,
                    "passage": cited["passage"],
                    "passage_figures": passage_figures,
                    "differing_figures": differs,
                    "differing_passage_figures": differs_in_passage,
                },
                "Cited source %s (%s) passage %r states figure(s) %s differing from the figure(s) %s the claim %s states: written %r, delivered %r." % (
                    cited["source_id"], cited["label"], cited["passage"],
                    ", ".join(repr(f) for f in differs_in_passage),
                    ", ".join(repr(f) for f in delivered_figures),
                    claim.get("id"), claim.get("text"),
                    entry.get("delivered_statement") if isinstance(entry, dict) else None,
                ),
            ))

        # source_does_not_cover_claim: a cited source's passage
        # carries nothing that speaks to the claim (the claim's
        # key phrase absent from the passage and no figure
        # shared); the passage quoted.
        for cited in uncovered_passages:
            where = "claim %s (%s)" % (claim.get("id"), claim.get("kind"))
            findings.append(_finding(
                "source_does_not_cover_claim",
                where,
                {
                    "claim_id": claim.get("id"),
                    "kind": claim.get("kind"),
                    "text": claim.get("text"),
                    "key_phrase": claim.get("key_phrase"),
                    "source_id": cited["source_id"],
                    "source_label": cited["label"],
                },
                {
                    "claim_id": claim.get("id"),
                    "kind": claim.get("kind"),
                    "passage": cited["passage"],
                    "passage_figures": cited["passage_figures"],
                },
                "Cited source %s (%s) passage %r carries nothing that speaks to claim %s (text %r, key phrase %r): the key phrase is absent from the passage and no figure is shared." % (
                    cited["source_id"], cited["label"], cited["passage"],
                    claim.get("id"), claim.get("text"), claim.get("key_phrase"),
                ),
            ))

        # source_misquoted: a claim's quote is not a verbatim
        # substring of its cited source's passage; both quoted.
        quote = claim.get("quote")
        if isinstance(quote, str) and quote:
            for source_id in cited_ids:
                source = sources_by_id.get(source_id)
                passage = source.get("passage") if isinstance(source, dict) else ""
                if not isinstance(passage, str):
                    passage = ""
                if quote in passage:
                    continue
                where = "claim %s (%s)" % (claim.get("id"), claim.get("kind"))
                findings.append(_finding(
                    "source_misquoted",
                    where,
                    {
                        "claim_id": claim.get("id"),
                        "kind": claim.get("kind"),
                        "text": claim.get("text"),
                        "quote": quote,
                        "source_id": source_id,
                        "source_label": source.get("label") if isinstance(source, dict) else None,
                    },
                    {
                        "claim_id": claim.get("id"),
                        "kind": claim.get("kind"),
                        "quote": quote,
                        "passage": passage,
                        "source_id": source_id,
                    },
                    "Claim %s's quote %r is not a verbatim substring of cited source %s's passage %r." % (
                        claim.get("id"), quote, source_id, passage
                    ),
                ))
                break

    claims_delivered = sum(
        1 for claim in claims
        if isinstance(delivered_by_id.get(claim.get("id")), dict) and delivered_by_id[claim.get("id")].get("present")
    )
    fact_claims = [claim for claim in claims if claim.get("kind") == "fact"]
    fact_checked = sum(1 for check in source_checks if check.get("status") != "unverified")
    conclusion_delivered = False
    if conclusion_id is not None:
        entry = delivered_by_id.get(conclusion_id)
        conclusion_delivered = bool(isinstance(entry, dict) and entry.get("present"))

    return {
        "findings": findings,
        "summary": _summary(findings),
        "claims_written": len(claims),
        "claims_delivered": claims_delivered,
        "claims_total": len(claims),
        "fact_claims_written": len(fact_claims),
        "fact_claims_checked": fact_checked,
        "fact_claims_total": len(fact_claims),
        "conclusion_delivered": conclusion_delivered,
        "conclusion_id": conclusion_id,
        "source_checks": source_checks,
        "tolerances": {
            "circularity_threshold": CIRCULARITY_THRESHOLD,
            "conclusion_restatement_threshold": CIRCULARITY_THRESHOLD,
        },
    }


build_comparison = compute_comparison
