"""stage2/apps/story/instrument.py - text analyzer over delivered text.

Reports measurements only, with its methods stated.  It never
calls anything wrong, broken, or good.
"""

import re
from collections import OrderedDict

METHOD = (
    "delivered text split into sections by headings of the form "
    "'## <title>'; per section, word count, sentence count "
    "(sentences split on . ! ?), dialogue share (words inside "
    "double quotation marks as a share of the section's words), "
    "and share of the whole; promised events checked as "
    "case-insensitive substrings of the delivered text"
)


def split_sections(text):
    sections = []
    current = None
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            if current is not None:
                sections.append(current)
            current = {"title": line[3:].strip(), "lines": []}
        elif current is not None and line:
            current["lines"].append(line)
    if current is not None:
        sections.append(current)
    return [
        {"title": s["title"], "text": "\n".join(s["lines"])}
        for s in sections
    ]


def _words(text):
    return re.findall(r"\S+", text)


def _sentences(text):
    parts = [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]
    return parts


def _dialogue_words(text):
    quoted = re.findall(r'"([^"]*)"', text)
    return sum(len(_words(q)) for q in quoted)


def analyze_text(text, promised_events=None):
    sections = split_sections(text)
    promised_events = list(promised_events or [])
    total_words = 0
    per_section = []
    for index, section in enumerate(sections, start=1):
        words = _words(section["text"])
        sentences = _sentences(section["text"])
        dialogue = _dialogue_words(section["text"])
        total_words += len(words)
        per_section.append(
            {
                "section": index,
                "title": section["title"],
                "word_count": len(words),
                "sentence_count": len(sentences),
                "dialogue_words": dialogue,
                "dialogue_share": (dialogue / len(words)) if words else 0.0,
                "share_of_whole": 0.0,
            }
        )
    if total_words:
        for entry in per_section:
            entry["share_of_whole"] = entry["word_count"] / total_words
    lowered = str(text).lower()
    found = []
    not_found = []
    for event in promised_events:
        if str(event).lower() in lowered:
            found.append(event)
        else:
            not_found.append(event)
    if per_section:
        summary = (
            "%d sections delivered, %d words in total. "
            "Per-section word counts: %s. "
            "Promised events present: %s. Promised events absent: %s."
            % (
                len(per_section),
                total_words,
                ", ".join(
                    "section %d (%s) %d words"
                    % (e["section"], e["title"], e["word_count"])
                    for e in per_section
                ),
                ", ".join(str(e) for e in found) if found else "none",
                ", ".join(str(e) for e in not_found) if not_found else "none",
            )
        )
    else:
        summary = "0 sections delivered, 0 words in total."
    measurements = {
        "instrument": "text analyzer",
        "method": METHOD,
        "section_count": len(per_section),
        "section_order": [e["title"] for e in per_section],
        "total_word_count": total_words,
        "sections": per_section,
        "promised_events_found": found,
        "promised_events_not_found": not_found,
    }
    full = {
        "instrument": "text analyzer",
        "method": METHOD,
        "delivered_text": text,
        "sections": sections,
        "measurements": measurements,
        "interpretation": None,
    }
    return {
        "summary": summary,
        "measurements": measurements,
        "full": full,
        "delivered_text": text,
        "instrument": "text analyzer",
        "interpretation": None,
    }


def perceive_text(text, promised_events=None):
    return analyze_text(text, promised_events)
