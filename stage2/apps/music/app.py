"""stage2/apps/music/app.py - music application, instrumental and vocal. Artifact names instrument; validation is structural only."""
import copy, json, math
from ...decision import DECISION_SHAPE_TEXT
from .perceive import perceive_wav
from .instruments import get_instrument, note_frequency, NOTES

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

def apply_revisions(artifact, revisions):
    cur = copy.deepcopy(artifact)
    for r in revisions or []:
        cur = apply_revision(cur, r)
    return cur
