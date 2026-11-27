"""stage2/apps/story/comparison.py - Sensoryne's judgment of the
delivered text against the written story.

The comparison works from the text analyzer's own analysis of the
delivered text (the evidence measurements for the delivered
sections and promised events, or the delivered text itself), set
against the written story as the artifact and its structure
record state it: the artifact's sections in order, the share
the written structure assigns each section, and the events the
objective promises.  It never works from delivery-fault
definitions: a finding names what the delivered text shows
against what was written.

Each finding is an object with kind, where, written, delivered,
and statement.  The kinds are event_not_delivered,
section_missing, section_share_off, and section_out_of_order.
The section-share judgment compares a delivered section's
measured share of the delivered text against the share the
written structure assigns that section; both shares are stated
in the finding, and a finding is made only where the departure
exceeds the stated tolerance.
"""

import copy
import os
import re

from .app import structure_record
from .instrument import analyze_text

__all__ = [
    "SECTION_SHARE_TOLERANCE",
    "FINDING_KINDS",
    "compute_comparison",
    "build_comparison",
]

SECTION_SHARE_TOLERANCE = 0.05

FINDING_KINDS = (
    "event_not_delivered",
    "section_missing",
    "section_share_off",
    "section_out_of_order",
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
            "story within the stated tolerances."
        )
    else:
        line = "%d finding(s): %s." % (
            total,
            ", ".join(
                "%s=%d" % (kind, counts[kind]) for kind in sorted(counts)
            ),
        )
    return {"counts": counts, "total": total, "line": line}


def _words(text):
    return re.findall(r"\S+", text or "")


def _objective(artifact):
    if isinstance(artifact, dict) and isinstance(artifact.get("objective"), dict):
        return artifact["objective"]
    return {}


def _promised_events(artifact, render_record, evidence):
    obj = _objective(artifact)
    events = obj.get("promised_events")
    if isinstance(events, list) and events:
        return list(events)
    if isinstance(artifact, dict) and isinstance(artifact.get("promised_events"), list) and artifact.get("promised_events"):
        return list(artifact["promised_events"])
    if isinstance(render_record, dict) and isinstance(render_record.get("promised_events"), list) and render_record.get("promised_events"):
        return list(render_record["promised_events"])
    for source in (evidence, render_record):
        if isinstance(source, dict) and isinstance(source.get("measurements"), dict):
            meas = source["measurements"]
            found = meas.get("promised_events_found")
            not_found = meas.get("promised_events_not_found")
            if isinstance(found, list) or isinstance(not_found, list):
                combined = list(found or []) + list(not_found or [])
                if combined:
                    return combined
    return list(events or []) if isinstance(events, list) else []


def _written_sections(artifact):
    raw_sections = []
    if isinstance(artifact, dict) and isinstance(artifact.get("sections"), list):
        raw_sections = artifact["sections"]
    structure = None
    try:
        structure = structure_record(artifact)
    except Exception:
        structure = None
    entries = []
    if isinstance(structure, dict) and isinstance(structure.get("sections"), list) and structure["sections"]:
        for index, entry in enumerate(structure["sections"]):
            if not isinstance(entry, dict):
                continue
            raw_text = ""
            if index < len(raw_sections) and isinstance(raw_sections[index], dict):
                raw_text = raw_sections[index].get("text", "") or ""
            title = entry.get("title")
            word_count = entry.get("word_count")
            if word_count is None:
                word_count = len(_words(raw_text))
            entries.append(
                {
                    "section": entry.get("section", index + 1),
                    "title": title,
                    "name": title,
                    "section_title": title,
                    "position": entry.get("section", index + 1),
                    "section_position": entry.get("section", index + 1),
                    "text": raw_text,
                    "word_count": word_count,
                    "share_of_whole": entry.get("share_of_whole", 0.0),
                    "written_share": entry.get("share_of_whole", 0.0),
                    "movement": entry.get("movement_assigned"),
                    "movement_assigned": entry.get("movement_assigned"),
                }
            )
    else:
        for index, section in enumerate(raw_sections):
            if not isinstance(section, dict):
                continue
            title = section.get("title")
            text = section.get("text", "") or ""
            entries.append(
                {
                    "section": index + 1,
                    "title": title,
                    "name": title,
                    "section_title": title,
                    "position": index + 1,
                    "section_position": index + 1,
                    "text": text,
                    "word_count": len(_words(text)),
                    "share_of_whole": 0.0,
                    "written_share": 0.0,
                    "movement": None,
                    "movement_assigned": None,
                }
            )
    total = sum(entry["word_count"] for entry in entries)
    for entry in entries:
        share = (entry["word_count"] / total) if total else 0.0
        entry["share_of_whole"] = share
        entry["written_share"] = share
        entry["share"] = share
    return entries


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


def _full_sections_of(evidence, render_record):
    for source in (evidence, render_record):
        if not isinstance(source, dict):
            continue
        full = source.get("full")
        if isinstance(full, dict) and isinstance(full.get("sections"), list):
            return full["sections"]
        if isinstance(source.get("sections"), list) and source is not render_record:
            # Evidence-side sections carry delivered title/text.
            sections = source.get("sections")
            if sections and isinstance(sections[0], dict) and "title" in sections[0]:
                return sections
    return None


def _delivered_data(render_record, evidence, promised_events):
    """Return (delivered_sections, delivered_text, found_events)."""
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

    sections = []
    # Primary: the analyzer's own per-section measurements.
    meas_sections = None
    if isinstance(measurements, dict) and isinstance(measurements.get("sections"), list) and measurements["sections"]:
        meas_sections = measurements["sections"]
    if meas_sections is not None:
        full_sections = _full_sections_of(evidence, render_record) or []
        total = measurements.get("total_word_count")
        if not isinstance(total, (int, float)) or isinstance(total, bool):
            total = None
        for index, entry in enumerate(meas_sections):
            if not isinstance(entry, dict):
                continue
            title = entry.get("title")
            word_count = entry.get("word_count", 0)
            try:
                word_count = int(word_count)
            except Exception:
                word_count = len(_words(entry.get("text", "") or ""))
            text = ""
            if index < len(full_sections) and isinstance(full_sections[index], dict):
                text = full_sections[index].get("text", "") or ""
                if title is None:
                    title = full_sections[index].get("title")
            share = entry.get("share_of_whole")
            if share is None:
                share = 0.0
            sections.append(
                {
                    "section": entry.get("section", index + 1),
                    "title": title,
                    "name": title,
                    "section_title": title,
                    "text": text,
                    "word_count": word_count,
                    "share_of_whole": float(share) if isinstance(share, (int, float)) and not isinstance(share, bool) else 0.0,
                    "delivered_share": float(share) if isinstance(share, (int, float)) and not isinstance(share, bool) else 0.0,
                }
            )
        if total is None:
            total = sum(entry["word_count"] for entry in sections)
        # Correct shares to the summed delivered word count when
        # a share was absent; measured shares are kept as stated.
        if total and any(entry["share_of_whole"] == 0.0 and entry["word_count"] for entry in sections):
            for entry in sections:
                entry["share_of_whole"] = entry["word_count"] / total
                entry["delivered_share"] = entry["share_of_whole"]
        if delivered_text is None and full_sections:
            texts = [
                str(s.get("text", "")) for s in full_sections if isinstance(s, dict)
            ]
            if texts and any(texts):
                delivered_text = " ".join(texts)
    else:
        full_sections = _full_sections_of(evidence, render_record)
        if full_sections is not None:
            all_text = "\n\n".join(
                "## %s\n%s" % (s.get("title", ""), s.get("text", ""))
                for s in full_sections
                if isinstance(s, dict)
            )
            analysis = analyze_text(all_text, promised_events)
            sections = [
                {
                    "section": entry["section"],
                    "title": entry["title"],
                    "name": entry["title"],
                    "section_title": entry["title"],
                    "text": "",
                    "word_count": entry["word_count"],
                    "share_of_whole": entry["share_of_whole"],
                    "delivered_share": entry["share_of_whole"],
                }
                for entry in analysis["measurements"]["sections"]
            ]
            if delivered_text is None:
                delivered_text = all_text
        elif delivered_text is not None:
            analysis = analyze_text(delivered_text, promised_events)
            sections = [
                {
                    "section": entry["section"],
                    "title": entry["title"],
                    "name": entry["title"],
                    "section_title": entry["title"],
                    "text": "",
                    "word_count": entry["word_count"],
                    "share_of_whole": entry["share_of_whole"],
                    "delivered_share": entry["share_of_whole"],
                }
                for entry in analysis["measurements"]["sections"]
            ]
        else:
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
                if text is not None:
                    delivered_text = text
                    analysis = analyze_text(text, promised_events)
                    sections = [
                        {
                            "section": entry["section"],
                            "title": entry["title"],
                            "name": entry["title"],
                            "section_title": entry["title"],
                            "text": "",
                            "word_count": entry["word_count"],
                            "share_of_whole": entry["share_of_whole"],
                            "delivered_share": entry["share_of_whole"],
                        }
                        for entry in analysis["measurements"]["sections"]
                    ]

    if delivered_text is not None:
        lowered = str(delivered_text).lower()
        found = [event for event in promised_events if str(event).lower() in lowered]
    elif isinstance(measurements, dict) and isinstance(measurements.get("promised_events_found"), list):
        found = list(measurements["promised_events_found"])
    else:
        found = []
    return sections, delivered_text, found


def _title_key(title):
    return str(title).strip().lower() if title is not None else ""


def compute_comparison(artifact, render_record=None, evidence=None):
    """Compare the delivered text against the written story."""
    if evidence is None and isinstance(render_record, dict):
        if "measurements" in render_record or "delivered_text" in render_record:
            evidence = render_record
            render_record = None

    promised_events = _promised_events(artifact, render_record, evidence)
    written = _written_sections(artifact)
    delivered, _delivered_text, found_events = _delivered_data(
        render_record, evidence, promised_events
    )

    findings = []

    # event_not_delivered: the promised event's own words, and
    # the section it was promised in (the written section whose
    # text carries the event).
    for event in promised_events:
        if event in found_events:
            continue
        promised_entry = None
        for entry in written:
            if str(event).lower() in str(entry.get("text", "")).lower():
                promised_entry = entry
                break
        if promised_entry is not None:
            where = "section %s (%s)" % (
                promised_entry.get("section"),
                promised_entry.get("title"),
            )
            statement = (
                "Promised event %r promised in section %s (%s) was "
                "not delivered in the delivered text."
                % (
                    event,
                    promised_entry.get("section"),
                    promised_entry.get("title"),
                )
            )
        else:
            where = "promised event %r" % (event,)
            statement = (
                "Promised event %r was not delivered in the "
                "delivered text." % (event,)
            )
        findings.append(
            _finding(
                "event_not_delivered",
                where,
                {
                    "event": event,
                    "promised_event": event,
                    "text": event,
                    "section": promised_entry.get("section") if promised_entry else None,
                    "section_position": promised_entry.get("section") if promised_entry else None,
                    "position": promised_entry.get("section") if promised_entry else None,
                    "section_title": promised_entry.get("title") if promised_entry else None,
                    "title": promised_entry.get("title") if promised_entry else None,
                    "name": promised_entry.get("title") if promised_entry else None,
                },
                {
                    "event": event,
                    "promised_event": event,
                    "text": event,
                    "present": False,
                    "found": False,
                    "section": None,
                    "section_title": None,
                },
                statement,
            )
        )

    delivered_titles = {_title_key(entry.get("title")) for entry in delivered}

    # section_missing: the written section's name and position.
    for entry in written:
        if _title_key(entry.get("title")) in delivered_titles:
            continue
        where = "section %s (%s)" % (entry.get("section"), entry.get("title"))
        findings.append(
            _finding(
                "section_missing",
                where,
                {
                    "section": entry.get("section"),
                    "position": entry.get("section"),
                    "section_position": entry.get("section"),
                    "title": entry.get("title"),
                    "name": entry.get("title"),
                    "section_title": entry.get("title"),
                    "word_count": entry.get("word_count"),
                    "share_of_whole": entry.get("share_of_whole"),
                    "written_share": entry.get("share_of_whole"),
                    "movement": entry.get("movement_assigned"),
                },
                {
                    "section": entry.get("section"),
                    "position": entry.get("section"),
                    "section_position": entry.get("section"),
                    "title": entry.get("title"),
                    "name": entry.get("title"),
                    "section_title": entry.get("title"),
                    "present": False,
                    "found": False,
                    "word_count": 0,
                    "share_of_whole": 0.0,
                    "delivered_share": 0.0,
                },
                "Written section %s (%s) is missing from the "
                "delivered text." % (entry.get("section"), entry.get("title")),
            )
        )

    # section_share_off: a delivered section's measured share
    # against the written share, both stated, beyond tolerance.
    total_delivered = sum(entry["word_count"] for entry in delivered)
    delivered_words_by_title = {}
    for entry in delivered:
        key = _title_key(entry.get("title"))
        delivered_words_by_title[key] = (
            delivered_words_by_title.get(key, 0) + entry["word_count"]
        )
    for entry in written:
        key = _title_key(entry.get("title"))
        if key not in delivered_titles:
            continue
        written_share = float(entry.get("share_of_whole") or 0.0)
        delivered_words = delivered_words_by_title.get(key, 0)
        delivered_share = (
            (delivered_words / total_delivered) if total_delivered else 0.0
        )
        departure = abs(delivered_share - written_share)
        if departure <= SECTION_SHARE_TOLERANCE:
            continue
        where = "section %s (%s)" % (entry.get("section"), entry.get("title"))
        findings.append(
            _finding(
                "section_share_off",
                where,
                {
                    "section": entry.get("section"),
                    "position": entry.get("section"),
                    "section_position": entry.get("section"),
                    "title": entry.get("title"),
                    "name": entry.get("title"),
                    "section_title": entry.get("title"),
                    "word_count": entry.get("word_count"),
                    "share_of_whole": written_share,
                    "written_share": written_share,
                    "share": written_share,
                },
                {
                    "section": entry.get("section"),
                    "position": entry.get("section"),
                    "section_position": entry.get("section"),
                    "title": entry.get("title"),
                    "name": entry.get("title"),
                    "section_title": entry.get("title"),
                    "word_count": delivered_words,
                    "share_of_whole": delivered_share,
                    "delivered_share": delivered_share,
                    "share": delivered_share,
                    "departure": departure,
                    "section_share_tolerance": SECTION_SHARE_TOLERANCE,
                },
                "Section %s (%s) was delivered at share %s of the "
                "delivered text against its written share %s, "
                "departing by %s, more than the %s section share "
                "tolerance."
                % (
                    entry.get("section"),
                    entry.get("title"),
                    delivered_share,
                    written_share,
                    departure,
                    SECTION_SHARE_TOLERANCE,
                ),
            )
        )

    # section_out_of_order: the two sections and their written
    # order. First delivered occurrence of each written section
    # only, so a duplicate does not invent an order.
    written_by_title = {}
    for entry in written:
        key = _title_key(entry.get("title"))
        if key not in written_by_title:
            written_by_title[key] = entry
    present_in_delivered_order = []
    seen_sections = set()
    for entry in delivered:
        key = _title_key(entry.get("title"))
        written_entry = written_by_title.get(key)
        if written_entry is None:
            continue
        if written_entry.get("section") in seen_sections:
            continue
        seen_sections.add(written_entry.get("section"))
        present_in_delivered_order.append(written_entry)

    for i in range(len(present_in_delivered_order)):
        for j in range(i + 1, len(present_in_delivered_order)):
            first_delivered = present_in_delivered_order[i]
            second_delivered = present_in_delivered_order[j]
            if first_delivered.get("section") <= second_delivered.get("section"):
                continue
            # first_delivered was written after second_delivered.
            earlier_written = second_delivered
            later_written = first_delivered
            where = "section %s (%s) and section %s (%s)" % (
                earlier_written.get("section"),
                earlier_written.get("title"),
                later_written.get("section"),
                later_written.get("title"),
            )
            findings.append(
                _finding(
                    "section_out_of_order",
                    where,
                    {
                        "first_section": earlier_written.get("section"),
                        "first_title": earlier_written.get("title"),
                        "first_name": earlier_written.get("title"),
                        "first_position": earlier_written.get("section"),
                        "second_section": later_written.get("section"),
                        "second_title": later_written.get("title"),
                        "second_name": later_written.get("title"),
                        "second_position": later_written.get("section"),
                        "section": earlier_written.get("section"),
                        "title": earlier_written.get("title"),
                        "sections": [
                            earlier_written.get("title"),
                            later_written.get("title"),
                        ],
                        "written_order": [
                            earlier_written.get("title"),
                            later_written.get("title"),
                        ],
                        "first": {
                            "section": earlier_written.get("section"),
                            "title": earlier_written.get("title"),
                            "name": earlier_written.get("title"),
                        },
                        "second": {
                            "section": later_written.get("section"),
                            "title": later_written.get("title"),
                            "name": later_written.get("title"),
                        },
                    },
                    {
                        "first_section": later_written.get("section"),
                        "first_title": later_written.get("title"),
                        "first_name": later_written.get("title"),
                        "first_position": later_written.get("section"),
                        "second_section": earlier_written.get("section"),
                        "second_title": earlier_written.get("title"),
                        "second_name": earlier_written.get("title"),
                        "second_position": earlier_written.get("section"),
                        "section": later_written.get("section"),
                        "title": later_written.get("title"),
                        "sections": [
                            later_written.get("title"),
                            earlier_written.get("title"),
                        ],
                        "delivered_order": [
                            later_written.get("title"),
                            earlier_written.get("title"),
                        ],
                        "first": {
                            "section": later_written.get("section"),
                            "title": later_written.get("title"),
                            "name": later_written.get("title"),
                        },
                        "second": {
                            "section": earlier_written.get("section"),
                            "title": earlier_written.get("title"),
                            "name": earlier_written.get("title"),
                        },
                    },
                    "Sections %r (written section %s) and %r "
                    "(written section %s) are out of order in the "
                    "delivered text: %r was written after %r but "
                    "delivered before it."
                    % (
                        earlier_written.get("title"),
                        earlier_written.get("section"),
                        later_written.get("title"),
                        later_written.get("section"),
                        later_written.get("title"),
                        earlier_written.get("title"),
                    ),
                )
            )

    promised_delivered = sum(
        1 for event in promised_events if event in found_events
    )
    delivered_positions = [
        entry.get("section") for entry in present_in_delivered_order
    ]
    sorted_positions = sorted(delivered_positions)
    sections_in_order = sum(
        1
        for delivered_rank, position in enumerate(delivered_positions)
        if position == sorted_positions[delivered_rank]
    )

    return {
        "findings": findings,
        "summary": _summary(findings),
        "promised_events_written": len(promised_events),
        "promised_events_delivered": promised_delivered,
        "promised_events_total": len(promised_events),
        "sections_written": len(written),
        "sections_delivered": len(delivered),
        "sections_present": len(present_in_delivered_order),
        "sections_delivered_in_written_order": sections_in_order,
        "sections_in_written_order": sections_in_order,
        "tolerances": {
            "section_share_tolerance": SECTION_SHARE_TOLERANCE,
            "section_share_tolerance_abs": SECTION_SHARE_TOLERANCE,
        },
    }


build_comparison = compute_comparison
