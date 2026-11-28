"""argument_renderer.py - composing-side argument renderer.

renderer(artifact, out_path, faults): writes the delivered text
to out_path with delivery faults applied to the delivered copy
only; the written argument is unchanged.  The render record
carries the claim map (claims, definitions, sources, and the
conclusion) alongside its other keys, so the application's
perceive can check each claim's key phrase and figures against
the delivered text; the story application's renderer does the
same with its structure record.
"""

import copy
import os
import re


def _claim_index(claims, claim_id):
    for index, claim in enumerate(claims):
        if isinstance(claim, dict) and claim.get("id") == claim_id:
            return index
    return None


def _swap_first_date(text, start):
    """Change the first date in text[start:] by one year."""
    for pattern in (
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    ):
        for match in re.finditer(pattern, text[start:]):
            old = match.group(0)
            digits = re.findall(r"\d+", old)
            if not digits:
                continue
            year_index = int(old.rfind(digits[-1]))
            year = int(digits[-1])
            new_year = year + 1 if year < 9999 else year - 1
            replacement = old[:year_index] + str(new_year) + old[year_index + len(digits[-1]):]
            return text[: start + match.start()] + replacement + text[start + match.end():]
    return text


def _apply_faults(text, claims, faults):
    text = str(text or "")
    claims = copy.deepcopy(list(claims or []))
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_claim":
            index = _claim_index(claims, fault.get("claim"))
            if index is not None:
                key_phrase = claims[index].get("key_phrase") or ""
                if key_phrase and key_phrase in text:
                    text = text.replace(key_phrase, "", 1)
        elif kind == "swap_date":
            index = _claim_index(claims, fault.get("claim"))
            if index is not None:
                key_phrase = claims[index].get("key_phrase") or ""
                start = text.find(key_phrase) if key_phrase else -1
                if start >= 0:
                    text = _swap_first_date(text, start)
        elif kind == "drop_passage":
            chars = fault.get("chars")
            if isinstance(chars, int) and chars > 0 and len(text) > chars:
                middle = len(text) // 2
                text = text[:middle] + text[middle + chars:]
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return text


def _objective_data(artifact):
    objective = artifact.get("objective") if isinstance(artifact, dict) else None
    claim_count = None
    kinds = []
    conclusion = None
    if isinstance(objective, dict):
        claim_count = objective.get("claim_count")
        kinds = list(objective.get("kinds") or [])
    if isinstance(artifact, dict):
        conclusion = artifact.get("conclusion")
    return claim_count, kinds, conclusion


def renderer(artifact, out_path, faults):
    # The claim map rides on the record through the artifact's
    # own lists, so perceive can check key phrases and figures
    # against the delivered text; the structure record carries
    # the map as the app's structure_record states it.
    claims = artifact.get("claims") if isinstance(artifact, dict) else []
    sources = artifact.get("sources") if isinstance(artifact, dict) else []
    definitions = artifact.get("definitions") if isinstance(artifact, dict) else []
    text = _apply_faults(
        artifact.get("text", "") if isinstance(artifact, dict) else "",
        claims if isinstance(claims, list) else [],
        faults,
    )
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    words = sum(len(str(claim.get("text", "")).split()) for claim in claims if isinstance(claim, dict)) if isinstance(claims, list) else 0
    claim_count, kinds, conclusion = _objective_data(artifact)
    claim_map = {
        "claims": copy.deepcopy(claims),
        "definitions": copy.deepcopy(definitions),
        "sources": copy.deepcopy(sources),
        "conclusion": conclusion,
    }
    try:
        from stage2.apps.argument.app import structure_record
        structure = structure_record(artifact)
    except Exception:
        structure = None
    return {
        "path": str(out_path),
        "kind": "text",
        "claims": len(claims) if isinstance(claims, list) else 0,
        "words": words,
        "claim_map": copy.deepcopy(claim_map),
        "structure_record": structure,
        "claims_list": copy.deepcopy(claims),
        "definitions": copy.deepcopy(definitions),
        "sources": copy.deepcopy(sources),
        "conclusion": conclusion,
        "claim_count": claim_count,
        "kinds": kinds,
        "faults_applied": copy.deepcopy(list(faults or [])),
    }
