"""stage2/apps/story/app.py - the story application.

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

NAME = "story"

REVISION_SHAPE_TEXT = """Revisions are operations applied to the story, in order. Section and sentence numbers are 1-based. References are checked, text must be non-empty, and the resulting story is re-validated before it becomes the new version. A rejected revision leaves the current story unchanged.
- {"op": "replace_section", "section": 2, "text": "..."}: replace the section's text.
- {"op": "insert_section", "after": 2, "title": "...", "text": "..."}: insert a new section after the given section (0 inserts at the beginning).
- {"op": "remove_section", "section": 2}: remove the section.
- {"op": "change_sentence", "section": 2, "sentence": 1, "text": "..."}: replace the sentence at the 1-based sentence position in the section."""


def validate(artifact):
    """Structure only: section count matches the objective when
    stated, every section non-empty."""
    errors = []
    if not isinstance(artifact, dict) or "sections" not in artifact:
        return ["story must be an object with a 'sections' list"]
    sections = artifact.get("sections")
    if not isinstance(sections, list) or not sections:
        return ["story must contain at least one section"]
    for index, section in enumerate(sections, start=1):
        if not isinstance(section, dict):
            errors.append("section %d must be an object" % index)
            continue
        if not isinstance(section.get("title"), str) or not section["title"].strip():
            errors.append("section %d has an empty title" % index)
        if not isinstance(section.get("text"), str) or not section["text"].strip():
            errors.append("section %d is empty" % index)
    objective = artifact.get("objective")
    if isinstance(objective, dict) and "section_count" in objective:
        if objective["section_count"] != len(sections):
            errors.append(
                "section count %d does not match the objective's %s"
                % (len(sections), objective["section_count"])
            )
    return errors


def _objective_text(objective):
    if isinstance(objective, dict):
        return json.dumps(objective, indent=2)
    return str(objective)


def composition_prompt(objective):
    return """Write a story meeting the objective's intended shape. Answer with the artifact JSON only, in this format and nothing else:
{"sections": [{"title": "...", "text": "..."}, {"title": "...", "text": "..."}]}

The objective states the intended shape as data: the section count, the movements (opening, rise, climax, fall, ending) assigned to section positions, and the promised events as short strings. Every promised event must appear in the delivered text.

Objective:
%s

Answer with one story JSON object in the format {"sections": [{"title": "...", "text": "..."}]} and nothing else.""" % _objective_text(objective)


def composition_retry_prompt(objective, errors, rejected):
    return """Your previous story did not pass validation. Please provide a corrected story.

Objective:
%s

Failed checks:
%s

Rejected story:
%s

Answer with one corrected story JSON object in the format {"sections": [{"title": "...", "text": "..."}]} and nothing else.""" % (
        _objective_text(objective),
        "\n".join("- " + str(e) for e in errors),
        json.dumps(rejected, indent=2),
    )


def control_prompt(objective, artifact):
    return """You are evaluating a story you composed. You cannot perceive the delivered text in this condition; judge from the story alone.

Objective:
%s

Story:
%s

State any faults you find in the story itself, order revisions if needed, and judge whether the work meets the objective.

%s

%s""" % (
        _objective_text(objective),
        json.dumps(artifact, indent=2),
        DECISION_SHAPE_TEXT,
        REVISION_SHAPE_TEXT,
    )


def perceptual_prompt(objective, evidence, input_summary, structure_record):
    return """You are evaluating a story you composed, given evidence from an independent text analyzer about the delivered text, the input summary, and the structure record of what you wrote.

Objective:
%s

INPUT (what was composed): the story exactly as written, below.

Input summary:
%s

Structure record (sections in order with word counts, each section's share of the whole, the position of the longest section, the movement assigned to each position, and the promised events; facts of shape and position only):
%s

OUTPUT (what was delivered): the delivered text, with the analyzer's full evidence about it below. The analyzer measures and reports; it does not judge. Judge the delivered story against what was written and the objective.

Evidence:
%s

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
    """The story exactly as written."""
    return copy.deepcopy(artifact)


def _word_count(text):
    return len(re.findall(r"\S+", text or ""))


def structure_record(artifact):
    sections = artifact.get("sections", []) if isinstance(artifact, dict) else []
    counts = [_word_count(s.get("text", "")) for s in sections if isinstance(s, dict)]
    total = sum(counts)
    entries = []
    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        share = (counts[index] / total) if total else 0.0
        entries.append(
            {
                "section": index + 1,
                "title": section.get("title"),
                "word_count": counts[index],
                "share_of_whole": share,
            }
        )
    longest = None
    if counts:
        longest = counts.index(max(counts)) + 1
    objective = artifact.get("objective") if isinstance(artifact, dict) else None
    movements = []
    promised_events = []
    if isinstance(objective, dict):
        movements = list(objective.get("movements") or [])
        promised_events = list(objective.get("promised_events") or [])
    elif isinstance(artifact, dict) and isinstance(artifact.get("movements"), list):
        movements = list(artifact.get("movements"))
    if isinstance(artifact, dict) and isinstance(artifact.get("promised_events"), list):
        promised_events = list(artifact.get("promised_events"))
    for entry in entries:
        position = entry["section"] - 1
        entry["movement_assigned"] = (
            movements[position] if position < len(movements) else None
        )
    return {
        "section_count": len(entries),
        "total_word_count": total,
        "sections": entries,
        "longest_section_position": longest,
        "movements_assigned": movements,
        "promised_events": promised_events,
    }


def generate_ground_truth(faults):
    """One ground-truth string per fault, from the definitions."""
    lines = []
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_section":
            lines.append(
                "drop_section: section %s is omitted from the delivered "
                "text, so a written section does not appear."
                % fault.get("section")
            )
        elif kind == "swap_sections":
            lines.append(
                "swap_sections: sections %s and %s exchange positions in "
                "the delivered text."
                % (fault.get("first"), fault.get("second"))
            )
        elif kind == "duplicate_section":
            lines.append(
                "duplicate_section: section %s appears twice in the "
                "delivered text." % fault.get("section")
            )
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return lines


def plant_faults(artifact):
    """Return the fault set for the given work, deterministically,
    using only defined kinds and only elements that exist."""
    count = len(artifact.get("sections", [])) if isinstance(artifact, dict) else 0
    faults = []
    if count >= 1:
        faults.append({"kind": "drop_section", "section": 1})
    if count >= 2:
        faults.append({"kind": "swap_sections", "first": 1, "second": 2})
        faults.append({"kind": "duplicate_section", "section": 2})
    return faults


def perceive(render_record):
    path = render_record["path"]
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    promised_events = []
    if isinstance(render_record.get("promised_events"), list):
        promised_events = list(render_record.get("promised_events") or [])
    if not promised_events:
        record = render_record.get("structure_record")
        if isinstance(record, dict):
            promised_events = list(record.get("promised_events") or [])
    return perceive_text(text, promised_events)


def _require_int(revision, key):
    value = revision.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("%s requires an integer %r" % (revision.get("op"), key))
    return value


def _require_text(revision, key="text"):
    text = revision.get(key)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("%r must be a non-empty string" % key)
    return text


def _split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def apply_revision(artifact, revision):
    """Return a revised deep copy; input never modified."""
    if not isinstance(revision, dict) or "op" not in revision:
        raise ValueError("revision must be an object with an 'op'")
    op = revision["op"]
    revised = copy.deepcopy(artifact)
    sections = revised.get("sections")
    if not isinstance(sections, list):
        raise ValueError("story has no sections")

    if op == "replace_section":
        number = _require_int(revision, "section")
        if number < 1 or number > len(sections):
            raise ValueError("section %r is out of range" % (number,))
        sections[number - 1]["text"] = _require_text(revision)
    elif op == "insert_section":
        after = _require_int(revision, "after")
        if after < 0 or after > len(sections):
            raise ValueError("'after' is out of range")
        title = revision.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("'title' must be a non-empty string")
        sections.insert(after, {"title": title, "text": _require_text(revision)})
    elif op == "remove_section":
        number = _require_int(revision, "section")
        if number < 1 or number > len(sections):
            raise ValueError("section %r is out of range" % (number,))
        del sections[number - 1]
    elif op == "change_sentence":
        number = _require_int(revision, "section")
        if number < 1 or number > len(sections):
            raise ValueError("section %r is out of range" % (number,))
        sentence_number = _require_int(revision, "sentence")
        sentences = _split_sentences(sections[number - 1].get("text", ""))
        if sentence_number < 1 or sentence_number > len(sentences):
            raise ValueError("sentence %r is out of range" % (sentence_number,))
        sentences[sentence_number - 1] = _require_text(revision)
        sections[number - 1]["text"] = " ".join(sentences)
    else:
        raise ValueError("unknown revision op: %r" % (op,))

    errors = validate(revised)
    if errors:
        raise ValueError(
            "resulting story failed validation: %s" % "; ".join(errors)
        )
    return revised


def apply_revisions(artifact, revisions):
    current = copy.deepcopy(artifact)
    for revision in revisions or []:
        current = apply_revision(current, revision)
    return current
