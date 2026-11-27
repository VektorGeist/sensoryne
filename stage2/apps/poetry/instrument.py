"""stage2/apps/poetry/instrument.py - mechanical text instrument.

Analyzes delivered text only.  It reports measurements and
labels every syllable count an estimate.  It never calls
anything wrong, broken, or good.
"""

import re
from collections import Counter, OrderedDict

SYLLABLE_RULE = (
    "count vowel groups in each word, subtract one for a silent "
    "final e, floor of one syllable per word"
)

_VOWELS = "aeiouy"


def syllable_estimate(word):
    """Estimate syllables for one word by the stated rule."""
    cleaned = re.sub(r"[^a-z]", "", str(word).lower())
    if not cleaned:
        return 1
    groups = 0
    in_group = False
    for char in cleaned:
        if char in _VOWELS:
            if not in_group:
                groups += 1
                in_group = True
        else:
            in_group = False
    if cleaned.endswith("e") and groups > 1:
        groups -= 1
    return max(1, groups)


def normalized_ending(end_word):
    """Substring from the end word's last vowel to its end."""
    cleaned = re.sub(r"[^a-z]", "", str(end_word).lower())
    if not cleaned:
        return ""
    for index in range(len(cleaned) - 1, -1, -1):
        if cleaned[index] in _VOWELS:
            return cleaned[index:]
    return cleaned


def parse_delivered_text(text):
    """Parse stanza-separated delivered text into stanzas."""
    stanzas = []
    current = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                stanzas.append(current)
                current = []
            continue
        current.append(line)
    if current:
        stanzas.append(current)
    return stanzas


def analyze_text(text):
    stanzas = parse_delivered_text(text)
    lines = []
    global_number = 0
    word_positions = OrderedDict()
    word_counts = Counter()

    for stanza_index, stanza in enumerate(stanzas, start=1):
        for line_text in stanza:
            global_number += 1
            words = line_text.split()
            end_word = words[-1] if words else ""
            syllables = sum(syllable_estimate(word) for word in words) if words else 0
            lines.append(
                {
                    "line": global_number,
                    "stanza": stanza_index,
                    "text": line_text,
                    "word_count": len(words),
                    "character_count": len(line_text),
                    "syllable_estimate": syllables,
                    "syllable_estimate_is_estimate": True,
                    "end_word": end_word,
                    "normalized_ending": normalized_ending(end_word),
                }
            )
            for word in words:
                cleaned = re.sub(r"[^a-z']", "", word.lower()).strip("'")
                if not cleaned:
                    continue
                word_counts[cleaned] += 1
                word_positions.setdefault(cleaned, [])
                if global_number not in word_positions[cleaned]:
                    word_positions[cleaned].append(global_number)

    ending_groups = OrderedDict()
    for entry in lines:
        ending = entry["normalized_ending"]
        if ending:
            ending_groups.setdefault(ending, [])
            ending_groups[ending].append(entry["line"])
    shared_endings = {
        ending: line_numbers
        for ending, line_numbers in ending_groups.items()
        if len(line_numbers) > 1
    }
    repeated_words = [
        {
            "word": word,
            "count": count,
            "lines": word_positions[word],
        }
        for word, count in sorted(word_counts.items())
        if count > 1
    ]
    estimates = [entry["syllable_estimate"] for entry in lines]
    lines_per_stanza = [len(stanza) for stanza in stanzas]

    if lines:
        summary_parts = [
            "%d lines delivered in %d stanzas." % (len(lines), len(stanzas))
        ]
        if estimates:
            summary_parts.append(
                "Syllable estimates range from %d to %d per line."
                % (min(estimates), max(estimates))
            )
        if shared_endings:
            shared_text = "; ".join(
                "lines %s share the ending %r"
                % (", ".join(str(n) for n in numbers), ending)
                for ending, numbers in shared_endings.items()
            )
            summary_parts.append("Shared endings: %s." % shared_text)
        else:
            summary_parts.append("No lines share an ending.")
        if repeated_words:
            repeated_text = "; ".join(
                "%r appears %d times on lines %s"
                % (
                    item["word"],
                    item["count"],
                    ", ".join(str(n) for n in item["lines"]),
                )
                for item in repeated_words
            )
            summary_parts.append("Repeated words: %s." % repeated_text)
        else:
            summary_parts.append("No word repeats.")
        summary = " ".join(summary_parts)
    else:
        summary = "0 lines delivered in 0 stanzas."

    measurements = {
        "instrument": "mechanical text analyzer",
        "syllable_rule": SYLLABLE_RULE,
        "syllable_estimates_are_estimates": True,
        "line_count": len(lines),
        "stanza_count": len(stanzas),
        "lines_per_stanza": lines_per_stanza,
        "syllable_estimate_range": (
            [min(estimates), max(estimates)] if estimates else []
        ),
        "lines": lines,
        "shared_endings": shared_endings,
        "repeated_words": repeated_words,
    }
    full = {
        "instrument": "mechanical text analyzer",
        "method": "delivered text parsed into stanzas by blank lines; per-line measurements computed mechanically",
        "syllable_rule": SYLLABLE_RULE,
        "syllable_estimates_are_estimates": True,
        "delivered_text": text,
        "stanzas": stanzas,
        "measurements": measurements,
        "interpretation": None,
    }
    return {
        "summary": summary,
        "measurements": measurements,
        "full": full,
        "delivered_text": text,
        "instrument": "mechanical text analyzer",
        "interpretation": None,
    }


def perceive_text(text):
    return analyze_text(text)
