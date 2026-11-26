"""story_renderer.py - composing-side story renderer.

renderer(artifact, out_path, faults): writes the delivered text
to out_path with delivery faults applied to the delivered copy
only; the written story is unchanged.  The render record
carries the objective's promised_events and movements alongside
its other keys, so the application's perceive can check the
promised events against the delivered text; the code
application's renderer does the same with its structure record
and tests.
"""

import copy
import os


def _apply_faults(sections, faults):
    sections = copy.deepcopy(sections)
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_section":
            number = fault.get("section")
            if isinstance(number, int) and 1 <= number <= len(sections):
                del sections[number - 1]
        elif kind == "duplicate_section":
            number = fault.get("section")
            if isinstance(number, int) and 1 <= number <= len(sections):
                sections.insert(number, copy.deepcopy(sections[number - 1]))
        elif kind == "swap_sections":
            first, second = fault.get("first"), fault.get("second")
            if (
                isinstance(first, int) and isinstance(second, int)
                and 1 <= first <= len(sections) and 1 <= second <= len(sections)
            ):
                sections[first - 1], sections[second - 1] = (
                    sections[second - 1],
                    sections[first - 1],
                )
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return sections


def _objective_data(artifact):
    objective = artifact.get("objective") if isinstance(artifact, dict) else None
    promised_events = []
    movements = []
    if isinstance(objective, dict):
        promised_events = list(objective.get("promised_events") or [])
        movements = list(objective.get("movements") or [])
    if isinstance(artifact, dict):
        if not promised_events and isinstance(artifact.get("promised_events"), list):
            promised_events = list(artifact.get("promised_events"))
        if not movements and isinstance(artifact.get("movements"), list):
            movements = list(artifact.get("movements"))
    return promised_events, movements


def renderer(artifact, out_path, faults):
    delivered = _apply_faults(artifact.get("sections", []), faults)
    text = "\n\n".join(
        "## %s\n%s" % (section.get("title", ""), section.get("text", ""))
        for section in delivered
    ) + "\n"
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    words = sum(len(str(s.get("text", "")).split()) for s in delivered)
    promised_events, movements = _objective_data(artifact)
    return {
        "path": str(out_path),
        "kind": "text",
        "sections": len(delivered),
        "words": words,
        "promised_events": promised_events,
        "movements": movements,
        "faults_applied": copy.deepcopy(list(faults or [])),
    }
