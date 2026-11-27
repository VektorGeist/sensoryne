"""poetry_renderer.py - composing-side poetry renderer.

renderer(artifact, out_path, faults): writes the delivered text
page to out_path with delivery faults applied to the delivered
copy only; the written poem is unchanged.  Faults are applied
inside the stanza lists, so every line keeps its stanza
membership and no fault changes the stanza structure.
"""

import copy
import os


def _locate(stanzas, line_number):
    """Return (stanza_index, index_in_stanza) for a 1-based line
    number counted across the whole poem, or None if out of range."""
    if not isinstance(line_number, int) or isinstance(line_number, bool):
        return None
    remaining = line_number
    for stanza_index, stanza in enumerate(stanzas):
        if 1 <= remaining <= len(stanza):
            return stanza_index, remaining - 1
        remaining -= len(stanza)
    return None


def _apply_faults(stanzas, faults):
    stanzas = copy.deepcopy(stanzas)
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_line":
            location = _locate(stanzas, fault.get("line"))
            if location is not None:
                stanza_index, line_index = location
                del stanzas[stanza_index][line_index]
        elif kind == "duplicate_line":
            location = _locate(stanzas, fault.get("line"))
            if location is not None:
                stanza_index, line_index = location
                stanzas[stanza_index].insert(line_index + 1, stanzas[stanza_index][line_index])
        elif kind == "swap_lines":
            first = _locate(stanzas, fault.get("first"))
            second = _locate(stanzas, fault.get("second"))
            if first is not None and second is not None:
                first_stanza, first_index = first
                second_stanza, second_index = second
                stanzas[first_stanza][first_index], stanzas[second_stanza][second_index] = (
                    stanzas[second_stanza][second_index],
                    stanzas[first_stanza][first_index],
                )
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return stanzas


def renderer(artifact, out_path, faults):
    delivered = _apply_faults(artifact.get("stanzas", []), faults)
    text = "\n\n".join("\n".join(stanza) for stanza in delivered) + "\n"
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    total = sum(len(stanza) for stanza in delivered)
    return {
        "path": str(out_path),
        "kind": "text",
        "lines": total,
        "stanzas": len(delivered),
        "faults_applied": list(faults or []),
    }
