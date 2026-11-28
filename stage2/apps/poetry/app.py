"""stage2/apps/poetry/app.py - the poetry application.

Text only; no audio instrument is involved.  Sensoryne receives
the delivered text produced on the composing side.  Inside
Sensoryne this application keeps the artifact format and
validation, revisions, the text instrument adapter, the input
summary and structure record, delivery-fault definitions and
ground-truth text, and the prompts.  Delivery faults are
definitions recorded here and applied by the composing side;
the delivered text Sensoryne receives already carries them.
"""

import copy
import json

from ...decision import DECISION_SHAPE_TEXT
from .instrument import perceive_text

NAME = "poetry"

REVISION_SHAPE_TEXT = """Revisions are operations applied to the poem, in order. Line numbers are 1-based across the whole poem. Stanza and line references are checked, text must be non-empty, and the resulting poem is re-validated before it becomes the new version. A rejected revision leaves the current poem unchanged.
- {"op": "replace_line", "line": 3, "text": "..."}: replace the whole line.
- {"op": "insert_line", "after": 3, "text": "..."}: insert a new line after the given line (0 inserts at the beginning).
- {"op": "remove_line", "line": 3}: remove the line.
- {"op": "change_word", "line": 3, "word": 2, "text": "..."}: replace the word at the 1-based word position in the line."""


def validate(artifact):
    """Structure only: an object with a stanzas list, at
    least one stanza, each stanza a list of lines, each line
    a non-empty string."""
    errors = []
    if not isinstance(artifact, dict) or "stanzas" not in artifact:
        return ["poem must be an object with a 'stanzas' list"]
    stanzas = artifact.get("stanzas")
    if not isinstance(stanzas, list) or not stanzas:
        return ["poem must contain at least one stanza"]
    for index, stanza in enumerate(stanzas, start=1):
        if not isinstance(stanza, list) or not stanza:
            errors.append("stanza %d must be a non-empty list of lines" % index)
            continue
        for line_index, line in enumerate(stanza, start=1):
            if not isinstance(line, str) or not line.strip():
                errors.append("stanza %d line %d is empty" % (index, line_index))
    return errors


def _objective_text(objective):
    if isinstance(objective, dict):
        return json.dumps(objective, indent=2)
    return str(objective)


def composition_prompt(objective):
    return """Compose a poem meeting the objective's form. Answer with the artifact JSON only, in this format and nothing else:
{"stanzas": [["line", "..."], ["..."], ["..."]]}

The objective states the form in words and as data: the total line count, the stanza count, the syllables per line, and the rhyme pairs (lines numbered from 1 across the poem). The poem is composed toward that form.

Objective:
%s

Answer with one poem JSON object in the format {"stanzas": [["line", "..."]]} and nothing else.""" % _objective_text(objective)


def composition_retry_prompt(objective, errors, rejected):
    return """Your previous poem did not pass validation. Please provide a corrected poem.

Objective:
%s

Failed checks:
%s

Rejected poem:
%s

Answer with one corrected poem JSON object in the format {"stanzas": [["line", "..."]]} and nothing else.""" % (
        _objective_text(objective),
        "\n".join("- " + str(e) for e in errors),
        json.dumps(rejected, indent=2),
    )


def control_prompt(objective, artifact):
    return """You are evaluating a poem you composed. You cannot perceive the delivered text in this condition; judge from the poem alone.

Objective:
%s

Poem:
%s

State any faults you find in the poem itself, order revisions if needed, and judge whether the work meets the objective.

%s

%s""" % (
        _objective_text(objective),
        json.dumps(artifact, indent=2),
        DECISION_SHAPE_TEXT,
        REVISION_SHAPE_TEXT,
    )


def perceptual_prompt(objective, evidence, input_summary, structure_record):
    return """You are evaluating a poem you composed, given evidence from an independent text instrument about the delivered text, the input summary, and the structure record of what you wrote.

Objective:
%s

INPUT (what was composed): the poem exactly as written, below.

Input summary:
%s

Structure record (the poem's lines numbered across the whole poem, grouped by stanza, with the form restated):
%s

OUTPUT (what was delivered): the delivered text, with the instrument's full evidence about it below. Judge the delivered poem against what was written and the objective.

Evidence:
%s

The evidence's comparison section is Sensoryne's judgment of the delivered text against the written poem and the objective, and the decision must state how it answers that judgment.

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
    """The poem exactly as written."""
    return copy.deepcopy(artifact)


def structure_record(artifact):
    stanzas = []
    number = 0
    for stanza_index, stanza in enumerate(artifact.get("stanzas", []), start=1):
        entries = []
        for line in stanza:
            number += 1
            entries.append({"line": number, "text": line})
        stanzas.append(
            {"stanza": stanza_index, "line_count": len(entries), "lines": entries}
        )
    return {
        "line_count": number,
        "stanza_count": len(stanzas),
        "form": copy.deepcopy(artifact.get("form")),
        "stanzas": stanzas,
    }


def generate_ground_truth(faults):
    """One ground-truth string per fault, from the definitions."""
    lines = []
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_line":
            lines.append(
                "drop_line: line %s is omitted from the delivered text, "
                "so a written line does not appear." % fault.get("line")
            )
        elif kind == "duplicate_line":
            lines.append(
                "duplicate_line: line %s appears twice in the delivered "
                "text." % fault.get("line")
            )
        elif kind == "swap_lines":
            lines.append(
                "swap_lines: lines %s and %s exchange positions in the "
                "delivered text."
                % (fault.get("first"), fault.get("second"))
            )
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return lines


def plant_faults(artifact):
    """Return the fault set for the given work, deterministically.

    Uses only the fault kinds this application defines.  Every
    referenced line exists in the work; kinds that do not apply
    to a work too small for them are omitted rather than
    inventing an element.  Faults are definitions only, applied
    on the composing side.
    """
    total = 0
    for stanza in artifact.get("stanzas", []) if isinstance(artifact, dict) else []:
        if isinstance(stanza, list):
            total += len(stanza)
    faults = []
    if total >= 1:
        faults.append({"kind": "drop_line", "line": 1})
    if total >= 2:
        faults.append({"kind": "duplicate_line", "line": 2})
        faults.append({"kind": "swap_lines", "first": 1, "second": 2})
    return faults


def perceive(render_record):
    path = render_record["path"]
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    return perceive_text(text)


def _locate(stanzas, line_number):
    if not isinstance(line_number, int) or isinstance(line_number, bool):
        raise ValueError("line must be an integer")
    remaining = line_number
    for stanza_index, stanza in enumerate(stanzas):
        if 1 <= remaining <= len(stanza):
            return stanza_index, remaining - 1
        remaining -= len(stanza)
    raise ValueError("line %r is out of range" % (line_number,))


def _require_text(revision):
    text = revision.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return text


def apply_revision(artifact, revision):
    """Return a revised deep copy; input never modified."""
    if not isinstance(revision, dict) or "op" not in revision:
        raise ValueError("revision must be an object with an 'op'")
    op = revision["op"]
    revised = copy.deepcopy(artifact)
    stanzas = revised.get("stanzas")
    if not isinstance(stanzas, list):
        raise ValueError("poem has no stanzas")
    total = sum(len(s) for s in stanzas if isinstance(s, list))

    if op == "replace_line":
        stanza_index, line_index = _locate(stanzas, revision.get("line"))
        stanzas[stanza_index][line_index] = _require_text(revision)
    elif op == "insert_line":
        after = revision.get("after")
        if not isinstance(after, int) or isinstance(after, bool):
            raise ValueError("insert_line requires an integer 'after'")
        if after < 0 or after > total:
            raise ValueError("'after' is out of range")
        text = _require_text(revision)
        if after == 0:
            if not stanzas:
                raise ValueError("poem has no stanzas")
            stanzas[0].insert(0, text)
        else:
            stanza_index, line_index = _locate(stanzas, after)
            stanzas[stanza_index].insert(line_index + 1, text)
    elif op == "remove_line":
        stanza_index, line_index = _locate(stanzas, revision.get("line"))
        del stanzas[stanza_index][line_index]
    elif op == "change_word":
        stanza_index, line_index = _locate(stanzas, revision.get("line"))
        word = revision.get("word")
        if not isinstance(word, int) or isinstance(word, bool):
            raise ValueError("change_word requires an integer 'word'")
        words = stanzas[stanza_index][line_index].split()
        if word < 1 or word > len(words):
            raise ValueError("word position %r is out of range" % (word,))
        words[word - 1] = _require_text(revision)
        stanzas[stanza_index][line_index] = " ".join(words)
    else:
        raise ValueError("unknown revision op: %r" % (op,))

    errors = validate(revised)
    if errors:
        raise ValueError(
            "resulting poem failed validation: %s" % "; ".join(errors)
        )
    return revised


def judge_brief():
    return """Judge this poem on its form kept or broken, its line craft, its imagery, and its close. Ground every claim in facts in the pack: the written work, the comparison findings, the objective coverage, and the maker's decisions. Where the record carries no fact, say the record does not show it; never invent one. Judge the work itself, not delivery alone: the delivery findings are facts to weigh, and the question is whether the work is good. Record the verdict with bottom_line, judgment, strengths, and weaknesses."""


def apply_revisions(artifact, revisions):
    current = copy.deepcopy(artifact)
    for revision in revisions or []:
        current = apply_revision(current, revision)
    return current
