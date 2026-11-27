"""stage2/apps/poetry/comparison.py - Sensoryne's judgment of the
delivered text against the written poem.

The comparison works from the instrument's own analysis of the
delivered text (the evidence measurements for the delivered
lines, or the delivered text itself), set against the written
poem as the artifact and its structure record state it.  It
never works from delivery-fault definitions: a finding names
what the delivered text shows against what was written.

Each finding is an object with kind, where, written, delivered,
and statement.  The kinds are line_missing, line_duplicated,
lines_swapped, line_text_differs, and form_off.  The syllable
judgment stays the instrument's stated estimate rule; every
syllable figure in a finding is named as an estimate.
"""

import copy
import os
from collections import Counter

from .app import structure_record
from .instrument import SYLLABLE_RULE, parse_delivered_text, syllable_estimate

__all__ = ["compute_comparison", "build_comparison"]

FINDING_KINDS = (
    "line_missing",
    "line_duplicated",
    "lines_swapped",
    "line_text_differs",
    "form_off",
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
            "No findings: the delivered text matches the written "
            "poem within the stated tolerances."
        )
    else:
        line = "%d finding(s): %s." % (
            total,
            ", ".join(
                "%s=%d" % (kind, counts[kind]) for kind in sorted(counts)
            ),
        )
    return {"counts": counts, "total": total, "line": line}


def _form(artifact, render_record):
    for source in (artifact, render_record):
        if isinstance(source, dict):
            if isinstance(source.get("form"), dict):
                return source["form"]
            if isinstance(source.get("objective"), dict):
                return source["objective"]
    if isinstance(artifact, dict):
        keys = ("line_count", "stanzas", "syllables_per_line")
        if any(key in artifact for key in keys):
            return artifact
    return {}


def _written_entries(artifact):
    """Flattened written lines, with stanza and line positions."""
    entries = []
    stanzas_text = []
    structure = None
    if isinstance(artifact, dict):
        try:
            structure = structure_record(artifact)
        except Exception:
            structure = None
    if isinstance(structure, dict) and isinstance(
        structure.get("stanzas"), list
    ):
        for stanza_entry in structure["stanzas"]:
            if not isinstance(stanza_entry, dict):
                continue
            stanza_index = stanza_entry.get("stanza")
            texts = []
            for position, line_entry in enumerate(
                stanza_entry.get("lines", []) or [], start=1
            ):
                if not isinstance(line_entry, dict):
                    continue
                text = line_entry.get("text", "")
                texts.append(text)
                entries.append(
                    {
                        "stanza": stanza_index,
                        "line_in_stanza": position,
                        "line": line_entry.get("line"),
                        "global_line": line_entry.get("line"),
                        "text": text,
                    }
                )
            stanzas_text.append(texts)
    else:
        raw = artifact.get("stanzas", []) if isinstance(artifact, dict) else []
        global_number = 0
        for stanza_index, stanza in enumerate(raw or [], start=1):
            if not isinstance(stanza, list):
                continue
            texts = []
            for position, text in enumerate(stanza, start=1):
                global_number += 1
                texts.append(text)
                entries.append(
                    {
                        "stanza": stanza_index,
                        "line_in_stanza": position,
                        "line": global_number,
                        "global_line": global_number,
                        "text": text,
                    }
                )
            stanzas_text.append(texts)
    return entries, stanzas_text


def _delivered_entries(render_record, evidence):
    """Flattened delivered lines, from the evidence if present."""
    lines = None
    for source in (evidence, render_record):
        if not isinstance(source, dict):
            continue
        measurements = source.get("measurements")
        if isinstance(measurements, dict) and isinstance(
            measurements.get("lines"), list
        ):
            lines = measurements["lines"]
            break
        full = source.get("full")
        if isinstance(full, dict):
            full_measurements = full.get("measurements")
            if isinstance(full_measurements, dict) and isinstance(
                full_measurements.get("lines"), list
            ):
                lines = full_measurements["lines"]
                break
    if lines is not None:
        entries = []
        for index, entry in enumerate(lines):
            if not isinstance(entry, dict):
                continue
            text = entry.get("text", "")
            estimate = entry.get("syllable_estimate")
            if estimate is None:
                estimate = sum(
                    syllable_estimate(word) for word in str(text).split()
                )
            entries.append(
                {
                    "stanza": entry.get("stanza"),
                    "line_in_stanza": entry.get("line_in_stanza"),
                    "line": entry.get("line", index + 1),
                    "global_line": entry.get("line", index + 1),
                    "text": text,
                    "syllable_estimate": estimate,
                }
            )
        # Fill line_in_stanza from stanza grouping when the
        # evidence does not carry it: count within each stanza
        # in delivered order.
        running = {}
        for entry in entries:
            stanza = entry.get("stanza")
            position = running.get(stanza, 0) + 1
            entry["line_in_stanza"] = position
            running[stanza] = position
        return entries

    text = None
    stanzas = None
    for source in (evidence, render_record):
        if not isinstance(source, dict):
            continue
        if isinstance(source.get("delivered_text"), str):
            text = source["delivered_text"]
            break
        full = source.get("full")
        if isinstance(full, dict):
            if isinstance(full.get("delivered_text"), str):
                text = full["delivered_text"]
                break
            if isinstance(full.get("stanzas"), list):
                stanzas = full["stanzas"]
                break
        if isinstance(source.get("stanzas"), list):
            stanzas = source["stanzas"]
            break
    if text is None and stanzas is None:
        path = None
        if isinstance(render_record, dict):
            candidate = render_record.get("path")
            if isinstance(candidate, str) and os.path.isfile(candidate):
                path = candidate
        if path is not None:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except Exception:
                text = None
    if stanzas is None:
        if text is None:
            return []
        stanzas = parse_delivered_text(text)
    entries = []
    global_number = 0
    for stanza_index, stanza in enumerate(stanzas or [], start=1):
        if not isinstance(stanza, list):
            continue
        for position, line_text in enumerate(stanza, start=1):
            global_number += 1
            entries.append(
                {
                    "stanza": stanza_index,
                    "line_in_stanza": position,
                    "line": global_number,
                    "global_line": global_number,
                    "text": line_text,
                    "syllable_estimate": sum(
                        syllable_estimate(word)
                        for word in str(line_text).split()
                    ),
                }
            )
    return entries


def _position_text(entry):
    if entry is None:
        return "no written line at that position"
    return "stanza %s line %s (global line %s)" % (
        entry.get("stanza"),
        entry.get("line_in_stanza"),
        entry.get("global_line"),
    )


def _where(entry):
    return _position_text(entry)


def _opening_words(text):
    words = str(text or "").split()
    return " ".join(words[:4])


def compute_comparison(artifact, render_record=None, evidence=None):
    """Compare the delivered text against the written poem."""
    # Tolerance for the controller's call shape as well as for
    # a direct call carrying the evidence in the second place.
    if evidence is None and isinstance(render_record, dict):
        if "measurements" in render_record or "delivered_text" in render_record:
            evidence = render_record
            render_record = None

    written, written_stanzas = _written_entries(artifact)
    delivered = _delivered_entries(render_record, evidence)
    form = _form(artifact, render_record)
    target = form.get("syllables_per_line") if isinstance(form, dict) else None

    findings = []

    written_counter = Counter(entry["text"] for entry in written)
    delivered_counter = Counter(entry["text"] for entry in delivered)

    # lines_swapped: two positions whose delivered texts are
    # each other's written texts, and neither position matches
    # its written line.
    swapped_pairs = []
    used = set()
    limit = min(len(written), len(delivered))
    for i in range(limit):
        if i in used or written[i]["text"] == delivered[i]["text"]:
            continue
        for j in range(i + 1, limit):
            if j in used or written[j]["text"] == delivered[j]["text"]:
                continue
            if (
                delivered[i]["text"] == written[j]["text"]
                and delivered[j]["text"] == written[i]["text"]
            ):
                swapped_pairs.append((i, j))
                used.add(i)
                used.add(j)
                break

    for first_index, second_index in swapped_pairs:
        first_written = written[first_index]
        second_written = written[second_index]
        first_delivered = delivered[first_index]
        second_delivered = delivered[second_index]
        where = "%s and %s" % (
            _where(first_written),
            _where(second_written),
        )
        written_block = {
            "stanza": first_written.get("stanza"),
            "line": first_written.get("line"),
            "line_in_stanza": first_written.get("line_in_stanza"),
            "global_line": first_written.get("global_line"),
            "text": first_written.get("text"),
            "first_stanza": first_written.get("stanza"),
            "first_line": first_written.get("line"),
            "first_line_in_stanza": first_written.get("line_in_stanza"),
            "first_global_line": first_written.get("global_line"),
            "first_text": first_written.get("text"),
            "second_stanza": second_written.get("stanza"),
            "second_line": second_written.get("line"),
            "second_line_in_stanza": second_written.get("line_in_stanza"),
            "second_global_line": second_written.get("global_line"),
            "second_text": second_written.get("text"),
            "first": {
                "stanza": first_written.get("stanza"),
                "line": first_written.get("line"),
                "line_in_stanza": first_written.get("line_in_stanza"),
                "global_line": first_written.get("global_line"),
                "text": first_written.get("text"),
            },
            "second": {
                "stanza": second_written.get("stanza"),
                "line": second_written.get("line"),
                "line_in_stanza": second_written.get("line_in_stanza"),
                "global_line": second_written.get("global_line"),
                "text": second_written.get("text"),
            },
        }
        delivered_block = {
            "stanza": first_delivered.get("stanza"),
            "line": first_delivered.get("line"),
            "line_in_stanza": first_delivered.get("line_in_stanza"),
            "global_line": first_delivered.get("global_line"),
            "text": first_delivered.get("text"),
            "first": {
                "stanza": first_delivered.get("stanza"),
                "line": first_delivered.get("line"),
                "line_in_stanza": first_delivered.get("line_in_stanza"),
                "global_line": first_delivered.get("global_line"),
                "text": first_delivered.get("text"),
            },
            "second": {
                "stanza": second_delivered.get("stanza"),
                "line": second_delivered.get("line"),
                "line_in_stanza": second_delivered.get("line_in_stanza"),
                "global_line": second_delivered.get("global_line"),
                "text": second_delivered.get("text"),
            },
        }
        findings.append(
            _finding(
                "lines_swapped",
                where,
                written_block,
                delivered_block,
                "Lines at %s and %s are swapped in the delivered "
                "text: the written line %r at %s was delivered "
                "at %s, and the written line %r at %s was "
                "delivered at %s."
                % (
                    _position_text(first_written),
                    _position_text(second_written),
                    first_written.get("text"),
                    _position_text(first_written),
                    _position_text(second_delivered),
                    second_written.get("text"),
                    _position_text(second_written),
                    _position_text(first_delivered),
                ),
            )
        )

    # line_duplicated: a written line appearing more times in
    # the delivered text than it is written.
    for text, delivered_count in delivered_counter.items():
        written_count = written_counter.get(text, 0)
        if written_count <= 0 or delivered_count <= written_count:
            continue
        written_entry = next(
            entry for entry in written if entry["text"] == text
        )
        positions = [
            {
                "stanza": entry.get("stanza"),
                "line": entry.get("line"),
                "line_in_stanza": entry.get("line_in_stanza"),
                "global_line": entry.get("global_line"),
                "text": entry.get("text"),
            }
            for entry in delivered
            if entry["text"] == text
        ]
        findings.append(
            _finding(
                "line_duplicated",
                _where(written_entry),
                {
                    "stanza": written_entry.get("stanza"),
                    "line": written_entry.get("line"),
                    "line_in_stanza": written_entry.get("line_in_stanza"),
                    "global_line": written_entry.get("global_line"),
                    "text": written_entry.get("text"),
                },
                {
                    "text": text,
                    "count": delivered_count,
                    "appearances": delivered_count,
                    "times": delivered_count,
                    "positions": positions,
                },
                "The written line at %s is duplicated in the "
                "delivered text: %r appears %d times in the "
                "delivered text."
                % (
                    _position_text(written_entry),
                    text,
                    delivered_count,
                ),
            )
        )

    # Missing written lines, and delivered lines matching no
    # written line.  A novel delivered line paired with the
    # missing written line at its position is a text that
    # differs, not a line that is also missing.  When a novel
    # line sits at a position whose written line is present
    # elsewhere or no written line is missing, the mismatch is
    # a differing text, not a missing line paired to it.
    missing_indices = []
    for text, written_count in written_counter.items():
        delivered_count = delivered_counter.get(text, 0)
        if delivered_count >= written_count:
            continue
        deficit = written_count - delivered_count
        positions = [
            index
            for index, entry in enumerate(written)
            if entry["text"] == text
        ]
        missing_indices.extend(positions[-deficit:])
    missing_indices.sort()

    novel_indices = [
        index
        for index, entry in enumerate(delivered)
        if entry["text"] not in written_counter
    ]

    # Pair a novel delivered line with the written line at its
    # own position when one is missing there; otherwise leave
    # the position's written line to be judged as present in
    # the written poem and the novel line as differing, with
    # that same written line named as the line it differs
    # from.
    differing = []
    still_missing = list(missing_indices)
    for delivered_index in novel_indices:
        delivered_entry = delivered[delivered_index]
        written_entry = None
        if delivered_index in still_missing:
            still_missing.remove(delivered_index)
            written_entry = written[delivered_index]
        elif delivered_index < len(written):
            written_entry = written[delivered_index]
        elif still_missing:
            nearest = min(
                still_missing,
                key=lambda index: abs(index - delivered_index),
            )
            still_missing.remove(nearest)
            written_entry = written[nearest]
        differing.append((delivered_entry, written_entry))

    for delivered_entry, written_entry in differing:
        where_entry = delivered_entry
        findings.append(
            _finding(
                "line_text_differs",
                _where(where_entry),
                {
                    "stanza": (
                        written_entry.get("stanza")
                        if written_entry is not None
                        else None
                    ),
                    "line": (
                        written_entry.get("line")
                        if written_entry is not None
                        else None
                    ),
                    "line_in_stanza": (
                        written_entry.get("line_in_stanza")
                        if written_entry is not None
                        else None
                    ),
                    "global_line": (
                        written_entry.get("global_line")
                        if written_entry is not None
                        else None
                    ),
                    "text": (
                        written_entry.get("text")
                        if written_entry is not None
                        else None
                    ),
                },
                {
                    "stanza": delivered_entry.get("stanza"),
                    "line": delivered_entry.get("line"),
                    "line_in_stanza": delivered_entry.get("line_in_stanza"),
                    "global_line": delivered_entry.get("global_line"),
                    "text": delivered_entry.get("text"),
                },
                "At %s the delivered line %r matches no written "
                "line at its position: the written line was %r "
                "and the delivered line is %r."
                % (
                    _position_text(delivered_entry),
                    delivered_entry.get("text"),
                    (
                        written_entry.get("text")
                        if written_entry is not None
                        else None
                    ),
                    delivered_entry.get("text"),
                ),
            )
        )

    for written_index in still_missing:
        written_entry = written[written_index]
        opening = _opening_words(written_entry.get("text"))
        findings.append(
            _finding(
                "line_missing",
                _where(written_entry),
                {
                    "stanza": written_entry.get("stanza"),
                    "line": written_entry.get("line"),
                    "line_in_stanza": written_entry.get("line_in_stanza"),
                    "global_line": written_entry.get("global_line"),
                    "text": written_entry.get("text"),
                    "opening_words": opening,
                },
                {
                    "stanza": written_entry.get("stanza"),
                    "line": written_entry.get("line"),
                    "line_in_stanza": written_entry.get("line_in_stanza"),
                    "global_line": written_entry.get("global_line"),
                    "text": None,
                    "present": False,
                    "count": 0,
                },
                "The written line at %s is missing from the "
                "delivered text: the written line beginning %r "
                "was not delivered."
                % (_position_text(written_entry), opening),
            )
        )

    # form_off: delivered line or stanza counts against the
    # written poem's, and syllable estimates departing from
    # the objective form's syllables per line.
    delivered_stanzas = []
    grouping = {}
    for entry in delivered:
        grouping.setdefault(entry.get("stanza"), []).append(entry.get("text"))
    if grouping:
        delivered_stanzas = [
            grouping[key] for key in sorted(
                grouping, key=lambda key: (key is None, key)
            )
        ]
    lines_written = len(written)
    lines_delivered = len(delivered)
    stanzas_written = len(written_stanzas)
    stanzas_delivered = len(delivered_stanzas)

    if lines_delivered != lines_written:
        findings.append(
            _finding(
                "form_off",
                "line count",
                {
                    "line_count": lines_written,
                    "stanza_count": stanzas_written,
                },
                {
                    "line_count": lines_delivered,
                    "stanza_count": stanzas_delivered,
                },
                "Delivered line count %d differs from the "
                "written poem's line count %d."
                % (lines_delivered, lines_written),
            )
        )
    if stanzas_delivered != stanzas_written:
        findings.append(
            _finding(
                "form_off",
                "stanza count",
                {
                    "line_count": lines_written,
                    "stanza_count": stanzas_written,
                },
                {
                    "line_count": lines_delivered,
                    "stanza_count": stanzas_delivered,
                },
                "Delivered stanza count %d differs from the "
                "written poem's stanza count %d."
                % (stanzas_delivered, stanzas_written),
            )
        )

    if target is not None:
        for delivered_entry in delivered:
            estimate = delivered_entry.get("syllable_estimate")
            if estimate is None or estimate == target:
                continue
            written_entry = None
            for candidate in written:
                if (
                    candidate.get("stanza") == delivered_entry.get("stanza")
                    and candidate.get("line_in_stanza")
                    == delivered_entry.get("line_in_stanza")
                ):
                    written_entry = candidate
                    break
            findings.append(
                _finding(
                    "form_off",
                    _where(delivered_entry),
                    {
                        "stanza": delivered_entry.get("stanza"),
                        "line": delivered_entry.get("line"),
                        "line_in_stanza": delivered_entry.get("line_in_stanza"),
                        "global_line": delivered_entry.get("global_line"),
                        "text": (
                            written_entry.get("text")
                            if written_entry is not None
                            else None
                        ),
                        "syllables_per_line": target,
                    },
                    {
                        "stanza": delivered_entry.get("stanza"),
                        "line": delivered_entry.get("line"),
                        "line_in_stanza": delivered_entry.get("line_in_stanza"),
                        "global_line": delivered_entry.get("global_line"),
                        "text": delivered_entry.get("text"),
                        "syllable_estimate": estimate,
                    },
                    "The delivered line at %s has syllable "
                    "estimate %s, an estimate, departing from "
                    "the form's syllables per line %s."
                    % (
                        _position_text(delivered_entry),
                        estimate,
                        target,
                    ),
                )
            )

    lines_delivered_as_written = sum(
        min(written_counter[text], delivered_counter.get(text, 0))
        for text in written_counter
    )
    stanzas_complete = 0
    lines_in_written_position = 0
    for stanza_index, stanza_texts in enumerate(written_stanzas):
        delivered_texts = (
            delivered_stanzas[stanza_index]
            if stanza_index < len(delivered_stanzas)
            else []
        )
        if delivered_texts == stanza_texts:
            stanzas_complete += 1
        for position, text in enumerate(stanza_texts):
            if (
                position < len(delivered_texts)
                and delivered_texts[position] == text
            ):
                lines_in_written_position += 1

    return {
        "findings": findings,
        "summary": _summary(findings),
        "lines_written": lines_written,
        "lines_delivered": lines_delivered,
        "lines_delivered_as_written": lines_delivered_as_written,
        "lines_in_written_position": lines_in_written_position,
        "stanzas_written": stanzas_written,
        "stanzas_delivered": stanzas_delivered,
        "stanzas_complete": stanzas_complete,
        "tolerances": {
            "syllable_rule": SYLLABLE_RULE,
            "syllable_estimates_are_estimates": True,
            "syllables_per_line": target,
        },
    }


build_comparison = compute_comparison
