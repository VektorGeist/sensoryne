"""stage2/apps/argument/instrument.py - text analyzer over delivered text.

Reports measurements only, with its methods stated.  It never
calls anything wrong, broken, or good.
"""

import re

METHOD = (
    "delivered text checked for each claim's key phrase as a "
    "case-insensitive substring; figures (numbers, dates, "
    "percentages) extracted from claim texts and delivered "
    "statements; quotations checked as verbatim substrings of "
    "cited passages"
)


def extract_figures(text):
    """All figures (dates, percentages, numbers) in a string, in order."""
    text = str(text or "")
    figures = []
    spans = []

    def _taken(start, end):
        return any(start < e and end > s for s, e in spans)

    def _add(pattern):
        for match in re.finditer(pattern, text):
            if not _taken(match.start(), match.end()):
                figures.append(match.group(0))
                spans.append(match.span())

    _add(r"\b\d{4}-\d{2}-\d{2}\b")
    _add(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
    _add(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b")
    _add(r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b")
    _add(r"\b\d+(?:\.\d+)?\s*%")
    _add(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
    return figures


def _words(text):
    return re.findall(r"\S+", text)


def _sentences(text):
    parts = [p.strip() for p in re.split(r"[.!?]+\s+", str(text).strip()) if p.strip()]
    return parts


def _sentence_containing(text, needle):
    lowered = str(needle).lower()
    for sentence in _sentences(text):
        if lowered in sentence.lower():
            return sentence
    return None


def _definition_statements(text):
    found = []
    text = str(text or "")
    for match in re.finditer(r"([A-Za-z][A-Za-z \-]{1,40}?)\s+(?:is|means|refers to)\s+([^.!?\n]+)", text):
        term = match.group(1).strip()
        definition = match.group(2).strip().rstrip(",;")
        if term and definition:
            found.append({"term": term, "definition": definition})
    return found


def analyze_text(text, claims=None, sources=None, definitions=None):
    """Measure the delivered text's claim presence and figures.

    Claims carry id, kind, text, key_phrase, premises, sources,
    and optional quote. Sources carry id, label, passage.
    Definitions carry term, definition.
    """
    claims = list(claims or [])
    sources = list(sources or [])
    definitions = list(definitions or [])
    text = str(text or "")
    lowered = text.lower()
    source_by_id = {}
    for source in sources:
        if isinstance(source, dict) and source.get("id") is not None:
            source_by_id[source.get("id")] = source

    per_claim = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        key_phrase = claim.get("key_phrase") or ""
        present = bool(key_phrase) and str(key_phrase).lower() in lowered
        delivered_statement = _sentence_containing(text, key_phrase) if present else None
        claim_figures = extract_figures(claim.get("text", ""))
        delivered_figures = extract_figures(delivered_statement) if delivered_statement else []
        cited = []
        for source_id in claim.get("sources") or []:
            source = source_by_id.get(source_id)
            if source is None:
                continue
            passage = source.get("passage") or ""
            passage_figures = extract_figures(passage)
            quote = claim.get("quote")
            cited.append({
                "source_id": source_id,
                "label": source.get("label"),
                "passage": passage,
                "passage_figures": passage_figures,
                "contains_key_phrase": bool(key_phrase) and str(key_phrase).lower() in passage.lower(),
                "shares_figure": bool(set(delivered_figures) & set(passage_figures)),
                "quote_verbatim": (quote in passage) if isinstance(quote, str) and quote else None,
            })
        per_claim.append({
            "claim_id": claim.get("id"),
            "kind": claim.get("kind"),
            "key_phrase": key_phrase,
            "present": present,
            "delivered_statement": delivered_statement,
            "claim_figures": claim_figures,
            "delivered_figures": delivered_figures,
            "cited_sources": cited,
        })

    claims_present = [entry["claim_id"] for entry in per_claim if entry["present"]]
    claims_missing = [entry["claim_id"] for entry in per_claim if not entry["present"]]

    delivered_definitions = _definition_statements(text)
    definition_checks = []
    for definition in definitions:
        if not isinstance(definition, dict):
            continue
        term = definition.get("term") or ""
        written = definition.get("definition") or ""
        stated = None
        for candidate in delivered_definitions:
            if candidate["term"].strip().lower() == str(term).strip().lower():
                stated = candidate["definition"]
                break
        definition_checks.append({
            "term": term,
            "written_definition": written,
            "delivered_definition": stated,
            "stated": stated is not None,
        })

    quoted_spans = re.findall(r'"([^"]*)"', text)
    quoted_spans += re.findall(r"'([^']*)'", text)

    total_words = len(_words(text))
    sentence_count = len(_sentences(text))
    if per_claim:
        summary = (
            "%d claims delivered of %d written. "
            "Claims present: %s. Claims missing: %s. "
            "%d definition statement(s) found in the delivered text. "
            "%d quoted span(s) found."
            % (
                len(claims_present),
                len(per_claim),
                ", ".join(str(c) for c in claims_present) if claims_present else "none",
                ", ".join(str(c) for c in claims_missing) if claims_missing else "none",
                len(delivered_definitions),
                len(quoted_spans),
            )
        )
    else:
        summary = "0 claims delivered of 0 written."
    measurements = {
        "instrument": "argument analyzer",
        "method": METHOD,
        "claim_count": len(per_claim),
        "claims_present": claims_present,
        "claims_missing": claims_missing,
        "claims": per_claim,
        "definition_statements": delivered_definitions,
        "definition_checks": definition_checks,
        "quoted_spans": quoted_spans,
        "total_word_count": total_words,
        "sentence_count": sentence_count,
    }
    full = {
        "instrument": "argument analyzer",
        "method": METHOD,
        "delivered_text": text,
        "measurements": measurements,
        "interpretation": None,
    }
    return {
        "summary": summary,
        "measurements": measurements,
        "full": full,
        "delivered_text": text,
        "instrument": "argument analyzer",
        "interpretation": None,
    }


def perceive_text(text, claims=None, sources=None, definitions=None):
    return analyze_text(text, claims, sources, definitions)
