"""stage2/apps/argument/app.py - the argument application.

Sensoryne receives the delivered text produced on the composing
side.  Inside Sensoryne this application keeps the artifact
format and validation, revisions, the text analyzer adapter,
the input summary and structure record, delivery-fault
definitions and ground-truth text, and the prompts.  Delivery
faults are definitions recorded here and applied by the
composing side; the delivered text Sensoryne receives already
carries them.
"""

import copy
import json
import re

from ...decision import DECISION_SHAPE_TEXT
from .instrument import perceive_text

NAME = "argument"

REVISION_SHAPE_TEXT = """Revisions are operations applied to the argument, in order. Claim ids are strings. References are checked, text must be non-empty, and the resulting argument is re-validated before it becomes the new version. A rejected revision leaves the current argument unchanged.
- {"op": "replace_claim", "id": "c1", "text": "...", "premises": ["c0"], "sources": ["s1"], "key_phrase": "..."}: replace the claim's text; "premises", "sources", and "key_phrase" are optional, and each replaces the claim's field when present.
- {"op": "change_claim", "id": "c1", "text": "...", "premises": ["c0"], "sources": ["s1"], "key_phrase": "..."}: replace the claim's text; "premises", "sources", and "key_phrase" are optional, and each replaces the claim's field when present.
- {"op": "insert_claim", "after": "c1", "id": "c2", "kind": "fact", "text": "...", "key_phrase": "..."}: insert a new claim after the given claim (null inserts at the beginning).
- {"op": "remove_claim", "id": "c2"}: remove the claim."""

_CLAIM_KINDS = ("fact", "logic", "value")


def validate(artifact):
    """Structure only: an object with a claims list, at least
    one claim, each claim holding an id, a valid kind, a text
    and a key_phrase; premise and source references resolve;
    the conclusion names an existing claim; text non-empty.
    Judgments are reported, never enforced; validate() gains no
    content rules."""
    errors = []
    if not isinstance(artifact, dict) or "claims" not in artifact:
        return ["argument must be an object with a 'claims' list"]
    claims = artifact.get("claims")
    if not isinstance(claims, list) or not claims:
        return ["argument must contain at least one claim"]
    seen = set()
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            errors.append("claim %d must be an object" % index)
            continue
        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not claim_id.strip():
            errors.append("claim %d has an empty id" % index)
        elif claim_id in seen:
            errors.append("claim id %r is duplicated" % (claim_id,))
        else:
            seen.add(claim_id)
        if claim.get("kind") not in _CLAIM_KINDS:
            errors.append("claim %s has an invalid kind" % (claim_id,))
        if not isinstance(claim.get("text"), str) or not claim["text"].strip():
            errors.append("claim %s has an empty text" % (claim_id,))
        if not isinstance(claim.get("key_phrase"), str) or not claim["key_phrase"].strip():
            errors.append("claim %s has an empty key_phrase" % (claim_id,))
    sources = artifact.get("sources") or []
    if not isinstance(sources, list):
        return errors + ["argument's 'sources' must be a list"]
    source_ids = set()
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            errors.append("source %d must be an object" % index)
            continue
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id.strip():
            errors.append("source %d has an empty id" % index)
        elif source_id in source_ids:
            errors.append("source id %r is duplicated" % (source_id,))
        else:
            source_ids.add(source_id)
        if not isinstance(source.get("label"), str) or not source["label"].strip():
            errors.append("source %s has an empty label" % (source_id,))
        if not isinstance(source.get("passage"), str) or not source["passage"].strip():
            errors.append("source %s has an empty passage" % (source_id,))
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = claim.get("id")
        for premise in claim.get("premises") or []:
            if premise not in seen:
                errors.append("claim %s cites unknown premise %r" % (claim_id, premise))
        for source_id in claim.get("sources") or []:
            if source_id not in source_ids:
                errors.append("claim %s cites unknown source %r" % (claim_id, source_id))
        quote = claim.get("quote")
        if quote is not None and (not isinstance(quote, str) or not quote.strip()):
            errors.append("claim %s has an empty quote" % (claim_id,))
    definitions = artifact.get("definitions") or []
    if not isinstance(definitions, list):
        errors.append("argument's 'definitions' must be a list")
    else:
        for index, definition in enumerate(definitions, start=1):
            if not isinstance(definition, dict):
                errors.append("definition %d must be an object" % index)
                continue
            if not isinstance(definition.get("term"), str) or not definition["term"].strip():
                errors.append("definition %d has an empty term" % index)
            if not isinstance(definition.get("definition"), str) or not definition["definition"].strip():
                errors.append("definition %d has an empty definition" % index)
    conclusion = artifact.get("conclusion")
    if not isinstance(conclusion, str) or conclusion not in seen:
        errors.append("conclusion must name an existing claim")
    if not isinstance(artifact.get("text"), str) or not artifact["text"].strip():
        errors.append("argument has an empty text")
    if not isinstance(artifact.get("title"), str) or not artifact["title"].strip():
        errors.append("argument has an empty title")
    return errors


def _objective_text(objective):
    if isinstance(objective, dict):
        return json.dumps(objective, indent=2)
    return str(objective)


def composition_prompt(objective):
    return """Write an argument meeting the objective's intended shape. Answer with the artifact JSON only, in this format and nothing else:
{"title": "...", "claims": [{"id": "c1", "kind": "fact", "text": "...", "key_phrase": "...", "premises": [], "sources": []}], "definitions": [{"term": "...", "definition": "..."}], "sources": [{"id": "s1", "label": "...", "passage": "..."}], "conclusion": "c2", "text": "..."}

The objective states the intended shape as data: the claim count, the kinds (fact, logic, value), and the conclusion's claim id. Every claim's key phrase must appear verbatim in the delivered text, and every source passage is quoted from the argument's own cited source. A claim with no cited source is marked unverified, never false.

Objective:
%s

Answer with one argument JSON object in the format {"title": "...", "claims": [...], "definitions": [...], "sources": [...], "conclusion": "...", "text": "..."} and nothing else.""" % _objective_text(objective)


def composition_retry_prompt(objective, errors, rejected):
    return """Your previous argument did not pass validation. Please provide a corrected argument.

Objective:
%s

Failed checks:
%s

Rejected argument:
%s

Answer with one corrected argument JSON object in the format {"title": "...", "claims": [...], "definitions": [...], "sources": [...], "conclusion": "...", "text": "..."} and nothing else.""" % (
        _objective_text(objective),
        "\n".join("- " + str(e) for e in errors),
        json.dumps(rejected, indent=2),
    )


def control_prompt(objective, artifact):
    return """You are evaluating an argument you composed. You cannot perceive the delivered text in this condition; judge from the argument alone.

Objective:
%s

Argument:
%s

State any faults you find in the argument itself, order revisions if needed, and judge whether the work meets the objective.

%s

%s""" % (
        _objective_text(objective),
        json.dumps(artifact, indent=2),
        DECISION_SHAPE_TEXT,
        REVISION_SHAPE_TEXT,
    )


def perceptual_prompt(objective, evidence, input_summary, structure_record):
    return """You are evaluating an argument you composed, given evidence from an independent text analyzer about the delivered text, the input summary, and the structure record of what you wrote.

Objective:
%s

INPUT (what was composed): the argument exactly as written, below.

Input summary:
%s

Structure record (the claim map: one row per claim with its kind, its premises, its cited sources, and the conclusion marked; the definitions and sources as written; facts of shape and position only):
%s

OUTPUT (what was delivered): the delivered text, with the analyzer's full evidence about it below. The analyzer measures and reports; it does not judge. Judge the delivered argument against what was written and the objective. No web search, no outside finder: nothing reaches outside the artifact and its cited passages.

Evidence:
%s

The evidence's comparison section is Sensoryne's judgment of the delivered text against the claim map and the objective, and the decision must state how it answers that judgment.

State any faults you find, order revisions if needed, and judge whether the delivered work meets the objective.

%s

%s""" % (
        _objective_text(objective),
        json.dumps(input_summary, indent=2),
        json.dumps(structure_record, indent=2),
        json.dumps(evidence, indent=2),
        DECISION_SHAPE_TEXT,
        REVISION_SHAPE_TEXT,
    )


def input_summary(artifact):
    """The argument exactly as written."""
    return copy.deepcopy(artifact)


def structure_record(artifact):
    claims = artifact.get("claims", []) if isinstance(artifact, dict) else []
    sources = artifact.get("sources", []) if isinstance(artifact, dict) else []
    definitions = artifact.get("definitions", []) if isinstance(artifact, dict) else []
    entries = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        entries.append({
            "id": claim.get("id"),
            "kind": claim.get("kind"),
            "text": claim.get("text"),
            "key_phrase": claim.get("key_phrase"),
            "premises": list(claim.get("premises") or []),
            "premise_count": len(claim.get("premises") or []),
            "sources": list(claim.get("sources") or []),
            "source_count": len(claim.get("sources") or []),
            "is_conclusion": claim.get("id") == artifact.get("conclusion") if isinstance(artifact, dict) else False,
        })
    return {
        "title": artifact.get("title") if isinstance(artifact, dict) else None,
        "claim_count": len(entries),
        "claims": entries,
        "conclusion": artifact.get("conclusion") if isinstance(artifact, dict) else None,
        "definitions": copy.deepcopy(definitions),
        "sources": copy.deepcopy(sources),
    }


def generate_ground_truth(faults):
    """One ground-truth string per fault, from the definitions."""
    lines = []
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_claim":
            lines.append(
                "drop_claim: claim %s's key phrase passage is omitted from the delivered "
                "text, so a written claim does not appear." % fault.get("claim")
            )
        elif kind == "swap_date":
            lines.append(
                "swap_date: a date in the delivered statement of claim %s is changed "
                "in the delivered text." % fault.get("claim")
            )
        elif kind == "drop_passage":
            lines.append(
                "drop_passage: a stretch of %s characters is removed from the delivered "
                "text." % fault.get("chars")
            )
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return lines


def plant_faults(artifact):
    """Return the fault set for the given work, deterministically,
    using only defined kinds and only elements that exist."""
    faults = []
    claims = artifact.get("claims", []) if isinstance(artifact, dict) else []
    if claims:
        faults.append({"kind": "drop_claim", "claim": claims[0].get("id")})
    for claim in claims:
        if isinstance(claim, dict) and claim.get("kind") == "fact":
            faults.append({"kind": "swap_date", "claim": claim.get("id")})
            break
    faults.append({"kind": "drop_passage", "chars": 20})
    return faults


def perceive(render_record):
    path = render_record["path"]
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    claims = []
    sources = []
    definitions = []
    artifact = render_record.get("_artifact")
    if isinstance(artifact, dict):
        if isinstance(artifact.get("claims"), list):
            claims = list(artifact.get("claims") or [])
        if isinstance(artifact.get("sources"), list):
            sources = list(artifact.get("sources") or [])
        if isinstance(artifact.get("definitions"), list):
            definitions = list(artifact.get("definitions") or [])
    if not claims and isinstance(render_record.get("claims"), list):
        claims = list(render_record.get("claims") or [])
    if not sources and isinstance(render_record.get("sources"), list):
        sources = list(render_record.get("sources") or [])
    if not definitions and isinstance(render_record.get("definitions"), list):
        definitions = list(render_record.get("definitions") or [])
    if not claims:
        record = render_record.get("structure_record")
        if isinstance(record, dict):
            claims = list(record.get("claims") or [])
            sources = list(record.get("sources") or [])
            definitions = list(record.get("definitions") or [])
    return perceive_text(text, claims, sources, definitions)


def apply_revision(artifact, revision):
    """Return a revised deep copy; input never modified."""
    if not isinstance(revision, dict) or "op" not in revision:
        raise ValueError("revision must be an object with an 'op'")
    op = revision["op"]
    revised = copy.deepcopy(artifact)
    claims = revised.get("claims")
    if not isinstance(claims, list):
        raise ValueError("argument has no claims")

    def _index(claim_id):
        for index, claim in enumerate(claims):
            if isinstance(claim, dict) and claim.get("id") == claim_id:
                return index
        return None

    def _require_id():
        value = revision.get("id")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("'id' must be a non-empty string")
        return value

    def _require_text():
        value = revision.get("text")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("'text' must be a non-empty string")
        return value

    if op in ("replace_claim", "change_claim"):
        claim_id = _require_id()
        index = _index(claim_id)
        if index is None:
            raise ValueError("claim %r does not exist" % (claim_id,))
        claims[index]["text"] = _require_text()
        if "premises" in revision:
            value = revision.get("premises")
            if value is None:
                value = []
            if not isinstance(value, list):
                raise ValueError("'premises' must be a list")
            claims[index]["premises"] = list(value)
        if "sources" in revision:
            value = revision.get("sources")
            if value is None:
                value = []
            if not isinstance(value, list):
                raise ValueError("'sources' must be a list")
            claims[index]["sources"] = list(value)
        if "key_phrase" in revision:
            key_phrase = revision.get("key_phrase")
            if not isinstance(key_phrase, str) or not key_phrase.strip():
                raise ValueError("'key_phrase' must be a non-empty string")
            claims[index]["key_phrase"] = key_phrase
    elif op == "insert_claim":
        new_id = _require_id()
        if _index(new_id) is not None:
            raise ValueError("claim id %r is duplicated" % (new_id,))
        after = revision.get("after")
        if after is None:
            position = 0
        else:
            if not isinstance(after, str) or not after.strip():
                raise ValueError("'after' must be a claim id or null")
            anchor = _index(after)
            if anchor is None:
                raise ValueError("claim %r does not exist" % (after,))
            position = anchor + 1
        kind = revision.get("kind")
        if kind not in _CLAIM_KINDS:
            raise ValueError("'kind' must be one of fact, logic, value")
        key_phrase = revision.get("key_phrase")
        if not isinstance(key_phrase, str) or not key_phrase.strip():
            raise ValueError("'key_phrase' must be a non-empty string")
        claims.insert(position, {
            "id": new_id,
            "kind": kind,
            "text": _require_text(),
            "key_phrase": key_phrase,
            "premises": list(revision.get("premises") or []),
            "sources": list(revision.get("sources") or []),
        })
    elif op == "remove_claim":
        claim_id = _require_id()
        index = _index(claim_id)
        if index is None:
            raise ValueError("claim %r does not exist" % (claim_id,))
        del claims[index]
    else:
        raise ValueError("unknown revision op: %r" % (op,))

    errors = validate(revised)
    if errors:
        raise ValueError(
            "resulting argument failed validation: %s" % "; ".join(errors)
        )
    return revised


def judge_brief():
    return """Judge this argument on whether its premises carry its conclusion, whether its terms are defined once and used honestly, whether its fact claims rest on the sources they cite as the record shows, and whether anything is asserted with nothing under it. Ground every claim in facts in the pack: the written work, the comparison findings, the objective coverage, and the maker's decisions. Where the record carries no fact, say the record does not show it; never invent one. Judge the work itself, not delivery alone. Record the verdict with bottom_line, judgment, strengths, and weaknesses."""


def apply_revisions(artifact, revisions):
    current = copy.deepcopy(artifact)
    for revision in revisions or []:
        current = apply_revision(current, revision)
    return current
