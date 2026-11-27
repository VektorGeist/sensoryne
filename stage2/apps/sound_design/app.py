"""stage2/apps/sound_design/app.py - sound design application."""
import copy, json

from ...decision import DECISION_SHAPE_TEXT
from .instrument import perceive_audio

NAME = "sound_design"
KINDS = ("noise burst", "filtered noise", "sine cluster", "percussive hit", "silence")
REVISION_SHAPE_TEXT = """Revisions addressed by event id:
- {"op": "set_gain", "id": "hit", "gain": 0.8}: set gain.
- {"op": "set_start", "id": "hit", "start": 1.25}: set start.
- {"op": "set_duration", "id": "hit", "duration": 0.5}: set duration.
- {"op": "add_event", "event": {"id": "tail", "kind": "noise burst", "start": 4.0, "duration": 0.5, "gain": 0.4, "filter_cutoff": 4000, "bandwidth": 1000}}: add one event.
- {"op": "remove_event", "id": "tail"}: remove one event."""

def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float("inf"), float("-inf"))

def _num(d, key, required=False):
    if key not in d or d.get(key) is None:
        if required:
            raise ValueError("%s requires %r" % (d.get("op"), key))
        return None
    if not _finite(d[key]):
        raise ValueError("%r must be finite" % key)
    return float(d[key])

def _param(event, key, default=None):
    if isinstance(event.get(key), (int, float)) and not isinstance(event.get(key), bool):
        return float(event[key])
    params = event.get("params")
    if isinstance(params, dict) and isinstance(params.get(key), (int, float)) and not isinstance(params.get(key), bool):
        return float(params[key])
    return default

def _check_event(event):
    errors = []
    if not isinstance(event, dict):
        return ["event must be an object"]
    if not isinstance(event.get("id"), str) or not event["id"].strip():
        errors.append("event id must be a non-empty string")
    if event.get("kind") not in KINDS:
        errors.append("event kind must be one of noise burst, filtered noise, sine cluster, percussive hit, silence")
    if not _finite(event.get("start")) or float(event.get("start", -1)) < 0:
        errors.append("event id %r start must be finite and not negative" % event.get("id"))
    if not _finite(event.get("duration")) or float(event.get("duration", 0)) <= 0:
        errors.append("event id %r duration must be positive" % event.get("id"))
    if not _finite(event.get("gain")) or float(event.get("gain", -1)) < 0:
        errors.append("event id %r gain must be finite and not negative" % event.get("id"))
    for key in ("filter_cutoff", "bandwidth"):
        value = _param(event, key)
        if value is not None and not _finite(value):
            errors.append("event id %r %s must be finite" % (event.get("id"), key))
        elif value is not None and value < 0:
            errors.append("event id %r %s must not be negative" % (event.get("id"), key))
        elif key == "filter_cutoff" and value is not None and value == 0:
            errors.append("event id %r filter_cutoff must be positive" % event.get("id"))
    return errors

def validate(artifact):
    if not isinstance(artifact, dict) or "events" not in artifact:
        return ["sound work must hold an events list"]
    errors = []
    if not _finite(artifact.get("duration")) or float(artifact.get("duration") or 0) <= 0:
        errors.append("sound work duration must be positive")
    events = artifact.get("events")
    if not isinstance(events, list) or not events:
        return errors + ["sound work must hold at least one event"]
    seen = set()
    for event in events:
        errors.extend(_check_event(event))
        if isinstance(event, dict) and event.get("id"):
            if event["id"] in seen:
                errors.append("duplicate event id %r" % event["id"])
            seen.add(event["id"])
    return errors

def _objective_text(objective):
    return json.dumps(objective, indent=2) if isinstance(objective, dict) else str(objective)

def composition_prompt(objective):
    return """Create a non-pitched sound work meeting the objective. Answer with the artifact JSON only:
{"duration": 6.0, "events": [{"id": "hit", "kind": "percussive hit", "start": 0.0, "duration": 0.4, "gain": 0.8, "filter_cutoff": 180, "bandwidth": 0}, {"id": "wash", "kind": "filtered noise", "start": 1.0, "duration": 3.0, "gain": 0.3, "filter_cutoff": 1200, "bandwidth": 800}]}

Objective:
%s

Kinds are noise burst, filtered noise, sine cluster, percussive hit, silence. Every event needs id, kind, start, duration, gain, and kind parameters filter_cutoff and bandwidth where they apply. Answer with one sound work JSON object and nothing else.""" % _objective_text(objective)

def composition_retry_prompt(objective, errors, rejected):
    return """Previous sound work failed validation. Corrected sound work only.

Objective:
%s

Failed checks:
%s

Rejected:
%s

Answer with one corrected sound work JSON object and nothing else.""" % (_objective_text(objective), "\n".join("- " + str(e) for e in errors), json.dumps(rejected, indent=2))

def control_prompt(objective, artifact):
    return """Evaluating a sound work you composed. You cannot perceive the rendered audio in this condition; judge from the sound work alone.

Objective:
%s

Sound work:
%s

State faults, order revisions if needed, judge whether the work meets the objective.

%s

%s""" % (_objective_text(objective), json.dumps(artifact, indent=2), DECISION_SHAPE_TEXT, REVISION_SHAPE_TEXT)

def perceptual_prompt(objective, evidence, input_summary, structure_record):
    return """Evaluating a sound work you composed, given evidence from an independent non-pitch audio analyzer about the rendered audio, the input summary, and the structure record.

Objective:
%s

INPUT (written):
%s

Structure record:
%s

OUTPUT (rendered audio, evidence):
%s

The evidence's comparison section is Sensoryne's judgment of the rendered audio against the written event list and the objective, and the decision must state how it answers that judgment.

Judge the rendered audio against what was written and the objective.

%s

%s""" % (_objective_text(objective), json.dumps(input_summary, indent=2), json.dumps(structure_record, indent=2), json.dumps(evidence, indent=2), DECISION_SHAPE_TEXT, REVISION_SHAPE_TEXT)

def input_summary(artifact):
    return copy.deepcopy(artifact)

def structure_record(artifact):
    events = artifact.get("events", []) if isinstance(artifact, dict) else []
    entries = []
    total = 0.0
    for event in events:
        if not isinstance(event, dict):
            continue
        start = float(event.get("start", 0)); dur = float(event.get("duration", 0))
        total += dur
        entries.append({"id": event.get("id"), "kind": event.get("kind"), "start": start, "end": start + dur, "duration": dur, "gain": event.get("gain"), "filter_cutoff": _param(event, "filter_cutoff"), "bandwidth": _param(event, "bandwidth")})
    overlaps = []
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            overlap = min(a["end"], b["end"]) - max(a["start"], b["start"])
            if overlap > 0:
                overlaps.append({"first": a["id"], "second": b["id"], "overlap_seconds": float(overlap)})
    return {"duration": artifact.get("duration") if isinstance(artifact, dict) else None, "event_count": len(entries), "events": entries, "declared_event_duration_total": float(total), "declared_overlaps": overlaps}

def generate_ground_truth(faults):
    lines = []
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_event":
            lines.append("drop_event: event %r is omitted from the rendered audio." % fault.get("id"))
        elif kind == "shift_event_start":
            lines.append("shift_event_start: event %r is rendered at start %s instead of %s." % (fault.get("id"), fault.get("new_start"), fault.get("from")))
        elif kind == "raise_event_gain":
            lines.append("raise_event_gain: event %r is rendered with gain multiplied by %s." % (fault.get("id"), fault.get("factor")))
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return lines

def plant_faults(artifact):
    events = artifact.get("events", []) if isinstance(artifact, dict) else []
    if not events:
        return []
    total = float(artifact.get("duration") or 0)
    audible = [i for i, e in enumerate(events) if isinstance(e, dict) and e.get("kind") != "silence"]
    drop_idx = audible[0] if audible else 0
    shift_idx = None
    gain_idx = None
    for i, e in enumerate(events):
        if not isinstance(e, dict) or i == drop_idx or e.get("kind") == "silence":
            continue
        start = float(e.get("start", 0)); dur = float(e.get("duration", 0))
        if shift_idx is None and start + 0.25 + dur <= total:
            shift_idx = i
        if gain_idx is None and float(e.get("gain", 0)) > 0:
            gain_idx = i
    if shift_idx is None:
        for i, e in enumerate(events):
            if isinstance(e, dict) and i != drop_idx and e.get("kind") != "silence" and float(e.get("start", 0)) + 0.25 + float(e.get("duration", 0)) <= total:
                shift_idx = i
                break
    if gain_idx is None:
        for i, e in enumerate(events):
            if isinstance(e, dict) and i != drop_idx and e.get("kind") != "silence" and float(e.get("gain", 0)) > 0:
                gain_idx = i
                break
    faults = [{"kind": "drop_event", "id": events[drop_idx].get("id")}]
    if shift_idx is not None:
        e = events[shift_idx]
        faults.append({"kind": "shift_event_start", "id": e.get("id"), "from": float(e.get("start", 0)), "new_start": float(e.get("start", 0)) + 0.25})
    if gain_idx is not None:
        faults.append({"kind": "raise_event_gain", "id": events[gain_idx].get("id"), "factor": 1.8})
    return faults

def perceive(render_record):
    return perceive_audio(render_record["path"], render_record.get("delivered_events"), render_record.get("duration"))

def _find(revised, event_id):
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("an event id is required")
    for event in revised.get("events", []):
        if isinstance(event, dict) and event.get("id") == event_id:
            return event
    raise ValueError("event id %r does not exist" % (event_id,))

def apply_revision(artifact, revision):
    if not isinstance(revision, dict) or "op" not in revision:
        raise ValueError("revision must be an object with an 'op'")
    op = revision["op"]
    revised = copy.deepcopy(artifact)
    if op == "set_gain":
        event = _find(revised, revision.get("id")); gain = _num(revision, "gain", True)
        if gain < 0:
            raise ValueError("gain must not be negative")
        event["gain"] = gain
    elif op == "set_start":
        event = _find(revised, revision.get("id")); start = _num(revision, "start", True)
        if start < 0:
            raise ValueError("start must not be negative")
        event["start"] = start
    elif op == "set_duration":
        event = _find(revised, revision.get("id")); duration = _num(revision, "duration", True)
        if duration <= 0:
            raise ValueError("duration must be positive")
        event["duration"] = duration
    elif op == "add_event":
        event = copy.deepcopy(revision.get("event"))
        errors = _check_event(event)
        if errors:
            raise ValueError("; ".join(errors))
        if any(isinstance(e, dict) and e.get("id") == event.get("id") for e in revised.get("events", [])):
            raise ValueError("event id %r already exists" % event.get("id"))
        revised.setdefault("events", []).append(event)
    elif op == "remove_event":
        event = _find(revised, revision.get("id")); revised["events"].remove(event)
    else:
        raise ValueError("unknown revision op: %r" % (op,))
    errors = validate(revised)
    if errors:
        raise ValueError("resulting sound work failed validation: %s" % "; ".join(errors))
    return revised

def judge_brief():
    return """Judge this sound work on its events, its timing, and the resulting sound picture. Ground every claim in facts in the pack: the written work, the comparison findings, the objective coverage, and the maker's decisions. Where the record carries no fact, say the record does not show it; never invent one. Judge the work itself, not delivery alone: the delivery findings are facts to weigh, and the question is whether the work is good. Record the verdict with bottom_line, judgment, strengths, and weaknesses."""


def apply_revisions(artifact, revisions):
    current = copy.deepcopy(artifact)
    for revision in revisions or []:
        current = apply_revision(current, revision)
    return current
