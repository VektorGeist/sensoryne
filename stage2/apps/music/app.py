"""stage2/apps/music/app.py - music application, instrumental and vocal. Artifact names instrument; validation is structural only."""
import copy, json, math
from ...decision import DECISION_SHAPE_TEXT
from .perceive import perceive_wav
from .instruments import get_instrument, NOTES

NAME = "music"
_OFFSET = {"piano": 0, "guitar": -12, "violin": 0, "viola": 0, "cello": 0, "double bass": -12, "harp": 0, "flute": 0, "oboe": 0, "clarinet": -2, "bassoon": 0, "trumpet": -2, "french horn": -7, "tenor trombone": 0, "soprano saxophone": -2, "alto saxophone": -9, "tenor saxophone": -14, "baritone saxophone": -21}

REVISION_SHAPE_TEXT = """Revisions applied to the score, in order:
- {"op": "set_part_gain", "part": "melody", "value": 0.0}: set a part's gain.
- {"op": "scale_velocity", "factor": 0.5, "part": "melody", "from": 9.0, "to": 12.0, "midi_min": 40, "midi_max": 80}: multiply matching velocities; part/from/to/midi_min/midi_max optional, all given must match.
- {"op": "add_note", "part": "melody", "start": 8.0, "midi": 65, "vel": 0.45, "dur": 1.9}: add one note.
- {"op": "remove_notes", "part": "melody", "from": 9.0, "to": 12.0}: remove notes in window; part optional, window required.
Vocal revisions, addressed to lyrics only:
- {"op": "replace_line", "line": 1, "text": ["la", "la"]}: replace a lyric line.
- {"op": "add_line", "text": ["la"]}: add a lyric line at the end, or {"op": "add_line", "after": 1, "text": ["la"]}.
- {"op": "remove_line", "line": 1}: remove a lyric line.
Invalid parts, lines, values, or results failing validation are rejected, leaving the artifact unchanged."""

def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))

def _midi_name(m):
    for e in NOTES:
        if e["midi"] == m:
            return e["name"]
    return None

def _is_vocal(a):
    return isinstance(a, dict) and a.get("kind") == "vocal"

def _validate_vocal(artifact):
    errors = []
    lyrics = artifact.get("lyrics")
    if not isinstance(lyrics, list) or not lyrics:
        return ["vocal artifact must hold a non-empty lyrics list"]
    for i, line in enumerate(lyrics, start=1):
        if not isinstance(line, list) or not line:
            errors.append("lyric line %d must be a non-empty list of syllables or words" % i)
        elif not all(isinstance(s, str) and s.strip() for s in line):
            errors.append("lyric line %d holds an empty syllable or word" % i)
    ref = artifact.get("render") or artifact.get("render_path") or artifact.get("audio")
    if ref is not None and not isinstance(ref, str):
        errors.append("vocal render reference must be a string")
    return errors

def score_total_length(s):
    t = 0.0
    for p in s.get("parts", []):
        for n in p.get("notes", []):
            d = n.get("dur")
            t = max(t, float(n["start"]) + (float(d) if d is not None else 0.0))
    return t

def validate(artifact):
    if not isinstance(artifact, dict):
        return ["artifact must be an object"]
    if _is_vocal(artifact):
        return _validate_vocal(artifact)
    if "parts" not in artifact or not isinstance(artifact.get("parts"), list):
        return ["score must be an object with a 'parts' list"]
    errors = []
    for part in artifact.get("parts", []):
        if not isinstance(part, dict):
            errors.append("part must be an object")
            continue
        name = part.get("name")
        gain = part.get("gain", 1.0)
        if "gain" in part and (not _finite(gain) or float(gain) < 0):
            errors.append("part %r has an invalid gain" % name)
        notes = part.get("notes", [])
        if notes is not None and not isinstance(notes, list):
            errors.append("part %r notes must be a list" % name)
            continue
        for n in notes or []:
            if not isinstance(n, dict):
                errors.append("part %r note must be an object" % name); break
            if not _finite(n.get("start")) or float(n["start"]) < 0:
                errors.append("part %r note invalid start" % name); break
            if not isinstance(n.get("midi"), int) or isinstance(n.get("midi"), bool) or not 0 <= n["midi"] <= 127:
                errors.append("part %r note invalid midi" % name); break
            v = n.get("vel")
            if not _finite(v) or not 0.0 <= float(v) <= 1.0:
                errors.append("part %r note invalid velocity" % name); break
            d = n.get("dur")
            if d is not None and (not _finite(d) or float(d) <= 0):
                errors.append("part %r note invalid duration" % name); break
    return errors

def composition_prompt(objective):
    return """Compose a work for Sensoryne. Answer with one artifact JSON object and nothing else.

Instrumental form: {"instrument": "piano", "parts": [{"name": "your part name", "gain": 1.0, "notes": [{"start": 0.0, "midi": 72, "vel": 0.4, "dur": 0.9}]}]}
Vocal form: {"kind": "vocal", "lyrics": [["la", "la"], ["lu", "na"]]}

Objective: %s

Format: an instrument name in "instrument", one or more parts with composer-chosen names in "parts", each part holding a "notes" list with notes as start, midi, vel, dur.
Vocal format: lyrics as a list of lines, each line a list of syllables or words in order.
Answer with one artifact JSON object and nothing else.""" % (objective,)

def composition_retry_prompt(objective, errors, rejected):
    return """Previous artifact failed validation. Corrected artifact only.

Objective: %s

Failed checks:
%s

Rejected:
%s

Answer with one corrected artifact JSON object and nothing else.""" % (objective, "\n".join("- " + str(e) for e in errors), json.dumps(rejected, indent=2))

def control_prompt(objective, artifact):
    return """Evaluating a work you composed; you cannot hear the render, judge from the artifact alone.

Objective: %s

Artifact:
%s

State faults, order revisions if needed, judge objective.

%s

%s""" % (objective, json.dumps(artifact, indent=2), DECISION_SHAPE_TEXT, REVISION_SHAPE_TEXT)

def perceptual_prompt(objective, evidence, input_summary, structure_record):
    return """Evaluating a work you composed, with evidence from Sensoryne hearing, input summary, and structure record.

Objective: %s

INPUT (written):
%s

Structure record:
%s

OUTPUT (rendered, evidence):
%s

The evidence's comparison section is Sensoryne's judgment of the render against the written work and the objective, and the decision must state how it answers that judgment.

Judge output vs input/objective, state faults, revisions, completeness.

%s

%s""" % (objective, json.dumps(input_summary, indent=2), json.dumps(structure_record, indent=2), json.dumps(evidence, indent=2), DECISION_SHAPE_TEXT, REVISION_SHAPE_TEXT)

def input_summary(artifact):
    if _is_vocal(artifact):
        return copy.deepcopy(artifact)
    out = []
    for p in artifact.get("parts", []):
        for n in p.get("notes", []):
            out.append({"part": p.get("name"), "start": float(n["start"]), "midi": int(n["midi"]), "dur": n.get("dur")})
    return out

def structure_record(artifact):
    if _is_vocal(artifact):
        lines = []
        number = 0
        for line in artifact.get("lyrics", []):
            number += 1
            lines.append({"line": number, "syllables": list(line), "syllable_count": len(line)})
        return {"kind": "vocal", "line_count": number, "syllable_count": sum(x["syllable_count"] for x in lines), "lines": lines}
    inst_name = artifact.get("instrument") or "piano"
    try:
        inst = get_instrument(inst_name)
    except Exception:
        inst = {"name": inst_name}
    off = _OFFSET.get(inst_name, 0)
    parts = []
    total = 0
    for p in artifact.get("parts", []):
        entries = []
        for n in p.get("notes", []):
            written = int(n["midi"])
            sounding = written + off
            sname = _midi_name(sounding)
            # Sounding frequency from the sounding midi
            # number directly (440 * 2**((midi-69)/12)),
            # never through the note name: several names in
            # the table are enharmonic respellings whose
            # name parsing would shift the pitch by a
            # semitone.
            sounding_freq = 440.0 * (2.0 ** ((sounding - 69) / 12.0)) if isinstance(sounding, (int, float)) else None
            entries.append({"start": float(n["start"]), "midi": written, "dur": n.get("dur"), "written_name": _midi_name(written), "sounding_midi": sounding, "sounding_name": sname, "sounding_freq_hz": sounding_freq})
        total += len(entries)
        parts.append({"part": p.get("name"), "note_count": len(entries), "notes": entries})
    return {"instrument": inst_name, "transposition": inst.get("transposition"), "written_range": inst.get("written") or inst.get("sounding"), "sounding_range": inst.get("sounding") or inst.get("written"), "part_count": len(parts), "note_count": total, "parts": parts}

def generate_ground_truth(faults):
    lines = []
    for f in faults or []:
        k = f.get("kind")
        if k == "part_gain":
            lines.append("part_gain: %s part louder than written (velocities x %s), may mask others." % (f.get("part"), f.get("factor")))
        elif k == "swell":
            lines.append("swell: all parts in [%s,%s) louder than written (x %s), not one part." % (f.get("from"), f.get("to"), f.get("factor")))
        elif k == "drop_note":
            lines.append("drop_note: first note in %s at %s midi %s omitted every render." % (f.get("part"), f.get("start"), f.get("midi")))
        else:
            raise ValueError("unknown fault kind: %r" % (k,))
    return lines

def plant_faults(score):
    if _is_vocal(score):
        return []
    parts = [p for p in score.get("parts", []) if isinstance(p, dict)] if isinstance(score, dict) else []
    if not parts:
        return []
    try:
        total = score_total_length(score)
    except Exception:
        total = 0.0
    faults = []
    candidates = []
    for p in parts:
        midis = [int(n["midi"]) for n in p.get("notes", []) if isinstance(n, dict) and isinstance(n.get("midi"), int) and not isinstance(n.get("midi"), bool)]
        if midis:
            candidates.append((min(midis), p.get("name")))
    if candidates:
        lowest_midi, lowest_part = min(candidates, key=lambda item: item[0])
        faults.append({"kind": "part_gain", "part": lowest_part, "factor": 2.4})
    if total and total > 0:
        faults.append({"kind": "swell", "from": float(total) * 0.45, "to": float(total) * 0.6, "factor": 1.9})
        half = float(total) / 2.0
        target = None
        target_part = None
        for p in parts:
            for n in p.get("notes", []):
                if isinstance(n, dict) and _finite(n.get("start")) and float(n["start"]) >= half:
                    target = n
                    target_part = p
                    break
            if target is not None:
                break
        if target is not None:
            faults.append({"kind": "drop_note", "part": target_part.get("name"), "start": float(target["start"]), "midi": int(target["midi"])})
    return faults

def perceive(render_record):
    artifact = None
    if isinstance(render_record, dict):
        artifact = render_record.get("_artifact") or render_record.get("artifact")
        if artifact is None and isinstance(render_record.get("delivered_events"), list):
            return _perceive_audio_events(render_record["path"], render_record.get("delivered_events"), render_record.get("duration"))
    return perceive_wav(render_record["path"], artifact=artifact)

def _clamp(v):
    if not _finite(v):
        raise ValueError("finite required")
    return float(min(1.0, max(0.0, float(v))))

def _req_part(s, part):
    if not any(isinstance(p, dict) and p.get("name") == part for p in s.get("parts", [])):
        raise ValueError("invalid part name: %r" % (part,))

def _num(r, key, req=False):
    if key not in r or r.get(key) is None:
        if req:
            raise ValueError("%s requires %r" % (r.get("op"), key))
        return None
    if not _finite(r[key]):
        raise ValueError("%r finite required" % key)
    return float(r[key])

def _apply_vocal_revision(revised, revision):
    op = revision["op"]
    lyrics = revised.get("lyrics")
    if not isinstance(lyrics, list):
        raise ValueError("vocal artifact has no lyrics list")
    def _text():
        text = revision.get("text")
        if not isinstance(text, list) or not text or not all(isinstance(s, str) and s.strip() for s in text):
            raise ValueError("line text must be a non-empty list of syllables or words")
        return list(text)
    if op == "replace_line":
        line = revision.get("line")
        if not isinstance(line, int) or isinstance(line, bool) or not 1 <= line <= len(lyrics):
            raise ValueError("line out of range")
        lyrics[line - 1] = _text()
    elif op == "add_line":
        text = _text()
        after = revision.get("after")
        if after is None:
            lyrics.append(text)
        elif isinstance(after, int) and not isinstance(after, bool) and 0 <= after <= len(lyrics):
            lyrics.insert(after, text)
        else:
            raise ValueError("after out of range")
    elif op == "remove_line":
        line = revision.get("line")
        if not isinstance(line, int) or isinstance(line, bool) or not 1 <= line <= len(lyrics):
            raise ValueError("line out of range")
        del lyrics[line - 1]
    else:
        raise ValueError("unknown revision op: %r" % (op,))
    errors = validate(revised)
    if errors:
        raise ValueError("resulting vocal artifact failed validation: %s" % "; ".join(errors))
    return revised

def apply_revision(artifact, revision):
    if not isinstance(revision, dict) or "op" not in revision:
        raise ValueError("revision needs op")
    op = revision["op"]
    revised = copy.deepcopy(artifact)
    if _is_vocal(artifact):
        if op not in ("replace_line", "add_line", "remove_line"):
            raise ValueError("unknown vocal revision op: %r" % (op,))
        return _apply_vocal_revision(revised, revision)
    by = {p.get("name"): p for p in revised.get("parts", []) if isinstance(p, dict)}
    if op == "set_part_gain":
        part = revision.get("part"); _req_part(revised, part); v = _num(revision, "value", True)
        if v < 0:
            raise ValueError("gain negative")
        by[part]["gain"] = v
    elif op == "scale_velocity":
        f = _num(revision, "factor", True)
        if f < 0:
            raise ValueError("factor negative")
        part = revision.get("part")
        if part is not None:
            _req_part(revised, part)
        sf = _num(revision, "from"); st = _num(revision, "to"); mn = _num(revision, "midi_min"); mx = _num(revision, "midi_max")
        if mn is not None and mx is not None and mn > mx:
            raise ValueError("midi_min>max")
        for p in revised.get("parts", []):
            if part and p.get("name") != part:
                continue
            for n in p.get("notes", []):
                if sf is not None and float(n["start"]) < sf: continue
                if st is not None and float(n["start"]) >= st: continue
                if mn is not None and int(n["midi"]) < mn: continue
                if mx is not None and int(n["midi"]) > mx: continue
                n["vel"] = _clamp(float(n["vel"]) * f)
    elif op == "add_note":
        part = revision.get("part"); _req_part(revised, part); st = _num(revision, "start", True)
        if st < 0:
            raise ValueError("start negative")
        if not isinstance(revision.get("midi"), int) or isinstance(revision.get("midi"), bool) or not 0 <= revision["midi"] <= 127:
            raise ValueError("midi invalid")
        v = _num(revision, "vel", True)
        if not 0.0 <= v <= 1.0:
            raise ValueError("vel range")
        d = revision.get("dur")
        if d is not None and (not _finite(d) or float(d) <= 0):
            raise ValueError("dur invalid")
        by[part].setdefault("notes", []).append({"start": st, "midi": revision["midi"], "vel": v, "dur": float(d) if d is not None else None})
    elif op == "remove_notes":
        sf = _num(revision, "from", True); st = _num(revision, "to", True)
        if st < sf:
            raise ValueError("to before from")
        part = revision.get("part")
        if part is not None:
            _req_part(revised, part)
        for p in revised.get("parts", []):
            if part and p.get("name") != part:
                continue
            p["notes"] = [n for n in p.get("notes", []) if not (sf <= float(n["start"]) < st)]
    else:
        raise ValueError("unknown revision op: %r" % (op,))
    errors = validate(revised)
    if errors:
        raise ValueError("resulting score failed validation: %s" % "; ".join(errors))
    return revised

def judge_brief():
    return """Judge this music on melody, harmony, structure, arc, dynamics, and whether the piece achieves what its objective declares. First name the kind of piece from the objective in the pack (symphony, sonata, concerto, piano piece, etude, song, miniature, or whatever the objective declares) and judge the work as that kind of thing. A symphony is judged on its movements or large sections, on thematic material stated and developed across the whole span, on contrast and return, and on whether its length is earned, meaning the material sustains the span rather than filling it. A work that declares itself a symphony and delivers a short single-arc piece has not met its kind, whatever its other merits. The other side holds with equal force: a piece that declares a small kind, a piano miniature, a study, or a short piece, is judged at its own scale and is neither penalized for not being a symphony nor credited as one. Ground every claim in facts in the pack: the written work, the comparison findings, the objective coverage, and the maker's decisions. Where the record carries no fact, say the record does not show it; never invent one. Judge the work itself, not delivery alone: the delivery findings are facts to weigh, and the question is whether the work is good. Record the verdict with bottom_line, judgment, strengths, and weaknesses."""

def apply_revisions(artifact, revisions):
    cur = copy.deepcopy(artifact)
    for r in revisions or []:
        cur = apply_revision(cur, r)
    return cur

# ---- merged Music & Sound: written events (folded from the former sound work form) ----
from .instrument import perceive_audio as _perceive_audio_events

KINDS = ("noise burst", "filtered noise", "sine cluster", "percussive hit", "silence")
_EVENT_REVISION_SHAPE_TEXT = """Revisions addressed by event id:
- {"op": "set_gain", "id": "hit", "gain": 0.8}: set gain.
- {"op": "set_start", "id": "hit", "start": 1.25}: set start.
- {"op": "set_duration", "id": "hit", "duration": 0.5}: set duration.
- {"op": "add_event", "event": {"id": "tail", "kind": "noise burst", "start": 4.0, "duration": 0.5, "gain": 0.4, "filter_cutoff": 4000, "bandwidth": 1000}}: add one event.
- {"op": "remove_event", "id": "tail"}: remove one event."""
REVISION_SHAPE_TEXT = REVISION_SHAPE_TEXT + "\n" + _EVENT_REVISION_SHAPE_TEXT

_score_validate = validate
_score_input_summary = input_summary
_score_structure_record = structure_record
_score_generate_ground_truth = generate_ground_truth
_score_plant_faults = plant_faults
_score_perceive = perceive
_score_apply_revision = apply_revision
_score_composition_prompt = composition_prompt
_score_control_prompt = control_prompt
_score_perceptual_prompt = perceptual_prompt
_score_judge_brief = judge_brief

def _has_parts(a):
    return isinstance(a, dict) and isinstance(a.get("parts"), list)

def _has_events(a):
    return isinstance(a, dict) and isinstance(a.get("events"), list)

def _ev_num(d, key, required=False):
    if key not in d or d.get(key) is None:
        if required:
            raise ValueError("%s requires %r" % (d.get("op"), key))
        return None
    if not _finite(d[key]):
        raise ValueError("%r must be finite" % key)
    return float(d[key])

def _ev_param(event, key, default=None):
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
        value = _ev_param(event, key)
        if value is not None and not _finite(value):
            errors.append("event id %r %s must be finite" % (event.get("id"), key))
        elif value is not None and value < 0:
            errors.append("event id %r %s must not be negative" % (event.get("id"), key))
        elif key == "filter_cutoff" and value is not None and value == 0:
            errors.append("event id %r filter_cutoff must be positive" % event.get("id"))
    return errors

def _validate_events(artifact):
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

def validate(artifact):
    if not isinstance(artifact, dict):
        return _score_validate(artifact)
    if _is_vocal(artifact):
        errors = list(_score_validate(artifact))
        if "events" in artifact:
            errors.extend(_validate_events(artifact))
        return errors
    has_parts = "parts" in artifact
    has_events = "events" in artifact
    if has_parts and has_events:
        return list(_score_validate(artifact)) + list(_validate_events(artifact))
    if has_events and not has_parts:
        return _validate_events(artifact)
    return _score_validate(artifact)

def _objective_text(objective):
    return json.dumps(objective, indent=2) if isinstance(objective, dict) else str(objective)

def composition_prompt(objective):
    return """Compose a work for Sensoryne. Answer with one artifact JSON object and nothing else. The work may be a score, a written event list, or both in one work.

Score form: {"instrument": "piano", "parts": [{"name": "your part name", "gain": 1.0, "notes": [{"start": 0.0, "midi": 72, "vel": 0.4, "dur": 0.9}]}]}
Event form: {"duration": 6.0, "events": [{"id": "hit", "kind": "percussive hit", "start": 0.0, "duration": 0.4, "gain": 0.8, "filter_cutoff": 180, "bandwidth": 0}, {"id": "wash", "kind": "filtered noise", "start": 1.0, "duration": 3.0, "gain": 0.3, "filter_cutoff": 1200, "bandwidth": 800}]}
Both form: one object carrying the score fields ("instrument", "parts") together with the event fields ("duration", "events").
Vocal form: {"kind": "vocal", "lyrics": [["la", "la"], ["lu", "na"]]}

Objective: %s

Format: for a score, an instrument name in "instrument", one or more parts with composer-chosen names in "parts", each part holding a "notes" list with notes as start, midi, vel, dur. For an event list, "duration" and "events" with kinds noise burst, filtered noise, sine cluster, percussive hit, silence; every event needs id, kind, start, duration, gain, and kind parameters filter_cutoff and bandwidth where they apply. Vocal format: lyrics as a list of lines, each line a list of syllables or words in order.
Answer with one artifact JSON object and nothing else.""" % _objective_text(objective)

def control_prompt(objective, artifact):
    return """Evaluating a work you composed; you cannot hear the render, judge from the artifact alone. The work may carry a written score, a written event list, or both.

Objective: %s

Artifact:
%s

State faults, order revisions if needed, judge objective.

%s

%s""" % (_objective_text(objective), json.dumps(artifact, indent=2), DECISION_SHAPE_TEXT, REVISION_SHAPE_TEXT)

def perceptual_prompt(objective, evidence, input_summary, structure_record):
    return """Evaluating a work you composed, with evidence from Sensoryne hearing, input summary, and structure record. The work may carry a written score, a written event list, or both.

Objective: %s

INPUT (written):
%s

Structure record:
%s

OUTPUT (rendered, evidence):
%s

The evidence's comparison section is Sensoryne's judgment of the render against the written score and the written event list and the objective, and the decision must state how it answers that judgment.

Judge output vs input/objective, state faults, revisions, completeness.

%s

%s""" % (_objective_text(objective), json.dumps(input_summary, indent=2), json.dumps(structure_record, indent=2), json.dumps(evidence, indent=2), DECISION_SHAPE_TEXT, REVISION_SHAPE_TEXT)

def input_summary(artifact):
    if _is_vocal(artifact):
        return _score_input_summary(artifact)
    if _has_parts(artifact) and _has_events(artifact):
        return {"score": _score_input_summary(artifact), "events": copy.deepcopy(artifact.get("events")), "duration": artifact.get("duration")}
    if _has_events(artifact) and not _has_parts(artifact):
        return copy.deepcopy(artifact)
    return _score_input_summary(artifact)

def _event_structure_record(artifact):
    events = artifact.get("events", []) if isinstance(artifact, dict) else []
    entries = []
    total = 0.0
    for event in events:
        if not isinstance(event, dict):
            continue
        start = float(event.get("start", 0)); dur = float(event.get("duration", 0))
        total += dur
        entries.append({"id": event.get("id"), "kind": event.get("kind"), "start": start, "end": start + dur, "duration": dur, "gain": event.get("gain"), "filter_cutoff": _ev_param(event, "filter_cutoff"), "bandwidth": _ev_param(event, "bandwidth")})
    overlaps = []
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            overlap = min(a["end"], b["end"]) - max(a["start"], b["start"])
            if overlap > 0:
                overlaps.append({"first": a["id"], "second": b["id"], "overlap_seconds": float(overlap)})
    return {"duration": artifact.get("duration") if isinstance(artifact, dict) else None, "event_count": len(entries), "events": entries, "declared_event_duration_total": float(total), "declared_overlaps": overlaps}

def structure_record(artifact):
    if _is_vocal(artifact):
        return _score_structure_record(artifact)
    if _has_parts(artifact) and _has_events(artifact):
        merged = dict(_score_structure_record(artifact))
        merged.update(_event_structure_record(artifact))
        return merged
    if _has_events(artifact) and not _has_parts(artifact):
        return _event_structure_record(artifact)
    return _score_structure_record(artifact)

def _event_generate_ground_truth(faults):
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

def generate_ground_truth(faults):
    lines = []
    for f in faults or []:
        k = f.get("kind") if isinstance(f, dict) else None
        if k in ("part_gain", "swell", "drop_note"):
            lines.extend(_score_generate_ground_truth([f]))
        elif k in ("drop_event", "shift_event_start", "raise_event_gain"):
            lines.extend(_event_generate_ground_truth([f]))
        else:
            raise ValueError("unknown fault kind: %r" % (k,))
    return lines

def _event_plant_faults(artifact):
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

def plant_faults(artifact):
    if _is_vocal(artifact):
        return []
    faults = []
    if _has_parts(artifact):
        faults.extend(_score_plant_faults(artifact))
    if _has_events(artifact):
        faults.extend(_event_plant_faults(artifact))
    return faults

def perceive(render_record):
    artifact = None
    if isinstance(render_record, dict):
        artifact = render_record.get("_artifact") or render_record.get("artifact")
        if artifact is None and isinstance(render_record.get("delivered_events"), list):
            return _perceive_audio_events(render_record["path"], render_record.get("delivered_events"), render_record.get("duration"))
    has_parts = isinstance(artifact, dict) and isinstance(artifact.get("parts"), list)
    has_events = isinstance(artifact, dict) and isinstance(artifact.get("events"), list)
    if has_events and not has_parts:
        duration = render_record.get("duration")
        if duration is None and isinstance(artifact, dict):
            duration = artifact.get("duration")
        return _perceive_audio_events(render_record["path"], render_record.get("delivered_events"), duration)
    if has_parts and has_events:
        hearing = _score_perceive(render_record)
        duration = render_record.get("duration")
        if duration is None:
            duration = artifact.get("duration")
        events_ev = _perceive_audio_events(render_record["path"], render_record.get("delivered_events"), duration)
        merged_measurements = dict(events_ev.get("measurements") or {})
        merged_measurements.update(hearing.get("measurements") or {})
        merged_measurements["events"] = (events_ev.get("measurements") or {}).get("events")
        merged_measurements["silence_share"] = (events_ev.get("measurements") or {}).get("silence_share")
        merged_measurements["event_count"] = (events_ev.get("measurements") or {}).get("event_count")
        merged_measurements["loudness_contour"] = (events_ev.get("measurements") or {}).get("loudness_contour")
        merged_measurements["spectral_centroid_hz"] = (events_ev.get("measurements") or {}).get("median_spectral_centroid_hz")
        full = dict(hearing.get("full") or {})
        full.update(events_ev.get("full") or {})
        full["measurements"] = merged_measurements
        full["render_path"] = str(render_record["path"])
        full["instrument"] = "Sensoryne hearing"
        full["interpretation"] = None
        summary = str(hearing.get("summary") or "") + " " + str(events_ev.get("summary") or "")
        return {"summary": summary.strip(), "measurements": merged_measurements, "token_pack": hearing.get("token_pack"), "full": full, "render_path": str(render_record["path"]), "instrument": "Sensoryne hearing", "interpretation": None}
    return _score_perceive(render_record)

def _ev_find(revised, event_id):
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("an event id is required")
    for event in revised.get("events", []):
        if isinstance(event, dict) and event.get("id") == event_id:
            return event
    raise ValueError("event id %r does not exist" % (event_id,))

def _event_apply_revision(artifact, revision):
    if not isinstance(revision, dict) or "op" not in revision:
        raise ValueError("revision must be an object with an 'op'")
    op = revision["op"]
    revised = copy.deepcopy(artifact)
    if op == "set_gain":
        event = _ev_find(revised, revision.get("id")); gain = _ev_num(revision, "gain", True)
        if gain < 0:
            raise ValueError("gain must not be negative")
        event["gain"] = gain
    elif op == "set_start":
        event = _ev_find(revised, revision.get("id")); start = _ev_num(revision, "start", True)
        if start < 0:
            raise ValueError("start must not be negative")
        event["start"] = start
    elif op == "set_duration":
        event = _ev_find(revised, revision.get("id")); duration = _ev_num(revision, "duration", True)
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
        event = _ev_find(revised, revision.get("id")); revised["events"].remove(event)
    else:
        raise ValueError("unknown revision op: %r" % (op,))
    errors = validate(revised)
    if errors:
        raise ValueError("resulting sound work failed validation: %s" % "; ".join(errors))
    return revised

_EVENT_OPS = ("set_gain", "set_start", "set_duration", "add_event", "remove_event")
_SCORE_OPS = ("set_part_gain", "scale_velocity", "add_note", "remove_notes")

def apply_revision(artifact, revision):
    if not isinstance(revision, dict) or "op" not in revision:
        raise ValueError("revision needs op")
    op = revision["op"]
    if _is_vocal(artifact):
        return _score_apply_revision(artifact, revision)
    if op in _EVENT_OPS:
        return _event_apply_revision(artifact, revision)
    if op in _SCORE_OPS:
        return _score_apply_revision(artifact, revision)
    raise ValueError("unknown revision op: %r" % (op,))

def judge_brief():
    return """Judge this music on melody, harmony, structure, arc, dynamics, and whether the piece achieves what its objective declares. First name the kind of piece from the objective in the pack (symphony, sonata, concerto, piano piece, etude, song, miniature, or whatever the objective declares) and judge the work as that kind of thing. A symphony is judged on its movements or large sections, on thematic material stated and developed across the whole span, on contrast and return, and on whether its length is earned, meaning the material sustains the span rather than filling it. A work that declares itself a symphony and delivers a short single-arc piece has not met its kind, whatever its other merits. The other side holds with equal force: a piece that declares a small kind, a piano miniature, a study, or a short piece, is judged at its own scale and is neither penalized for not being a symphony nor credited as one. Where the work carries written events, judge it also on its events, its timing, and the resulting sound picture: whether the named sounds are realized at their written onsets and whether the sound picture achieves what the objective declares. Ground every claim in facts in the pack: the written work, the written score and event list, the comparison findings, the objective coverage, and the maker's decisions. Where the record carries no fact, say the record does not show it; never invent one. Judge the work itself, not delivery alone: the delivery findings are facts to weigh, and the question is whether the work is good. Record the verdict with bottom_line, judgment, strengths, and weaknesses."""

