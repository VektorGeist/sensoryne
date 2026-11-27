"""stage2/apps/animation/app.py - short animation application.

A planted time shift moves an interior keyframe, never the
first or the last: shifting the start would be invisible to a
renderer that holds the first keyframe's state until the next
one, and shifting the end would change the track's end time.
The shift fault therefore moves the keyframe at index 1 to
the midpoint of the track, which changes the realized
positions across the middle of the scene while leaving the
first and last keyframes in place.
"""
import copy, json

from ...decision import DECISION_SHAPE_TEXT
from .instrument import perceive_frames

NAME = "animation"
EASINGS = ("linear", "ease-in", "ease-out", "ease-in-out")
REVISION_SHAPE_TEXT = """Revisions addressed by element id and keyframe time:
- {"op": "set_keyframe", "id": "ball", "time": 1.0, "keyframe": {"x": 100, "y": 50, "width": 20, "height": 20, "opacity": 1.0, "easing": "linear"}}: set the keyframe at that time.
- {"op": "add_keyframe", "id": "ball", "keyframe": {"time": 1.0, "x": 100, "y": 50, "width": 20, "height": 20, "opacity": 1.0, "easing": "linear"}}: add a keyframe.
- {"op": "remove_keyframe", "id": "ball", "time": 1.0}: remove the keyframe at that time.
- {"op": "set_easing", "id": "ball", "time": 1.0, "easing": "ease-in-out"}: set the easing of the segment starting at that time."""

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

def _check_keyframe(k):
    errors = []
    if not isinstance(k, dict):
        return ["keyframe must be an object"]
    for key in ("time", "x", "y", "width", "height", "opacity"):
        if not _finite(k.get(key)):
            errors.append("keyframe %s must be finite" % key)
    if _finite(k.get("width")) and float(k.get("width")) < 0:
        errors.append("keyframe width must not be negative")
    if _finite(k.get("height")) and float(k.get("height")) < 0:
        errors.append("keyframe height must not be negative")
    if _finite(k.get("opacity")) and not 0.0 <= float(k.get("opacity")) <= 1.0:
        errors.append("keyframe opacity must be between 0 and 1")
    if k.get("easing") not in EASINGS:
        errors.append("keyframe easing must be one of linear, ease-in, ease-out, ease-in-out")
    return errors

def validate(artifact):
    if not isinstance(artifact, dict) or "elements" not in artifact:
        return ["scene must hold an elements list"]
    errors = []
    if not _finite(artifact.get("duration")) or float(artifact.get("duration") or 0) <= 0:
        errors.append("scene duration must be positive")
    if not _finite(artifact.get("frame_rate")) or float(artifact.get("frame_rate") or 0) <= 0:
        errors.append("scene frame_rate must be positive")
    canvas = artifact.get("canvas")
    if not isinstance(canvas, dict) or not _finite(canvas.get("width")) or float(canvas.get("width") or 0) <= 0 or not _finite(canvas.get("height")) or float(canvas.get("height") or 0) <= 0:
        errors.append("scene canvas must hold positive width and height")
    elements = artifact.get("elements")
    if not isinstance(elements, list) or not elements:
        return errors + ["scene must hold at least one element"]
    seen = set()
    for el in elements:
        if not isinstance(el, dict):
            errors.append("element must be an object")
            continue
        if not isinstance(el.get("id"), str) or not el["id"].strip():
            errors.append("element id must be a non-empty string")
        elif el["id"] in seen:
            errors.append("duplicate element id %r" % el["id"])
        else:
            seen.add(el["id"])
        track = el.get("keyframes") or el.get("track")
        if not isinstance(track, list) or not track:
            errors.append("element id %r must hold a keyframe track" % el.get("id"))
            continue
        for k in track:
            errors.extend(_check_keyframe(k))
    return errors

def _objective_text(objective):
    return json.dumps(objective, indent=2) if isinstance(objective, dict) else str(objective)

def composition_prompt(objective):
    return """Create a short animation scene meeting the objective. Answer with the artifact JSON only:
{"duration": 2.0, "frame_rate": 12, "canvas": {"width": 320, "height": 240}, "loop": false, "elements": [{"id": "ball", "keyframes": [{"time": 0.0, "x": 10, "y": 100, "width": 20, "height": 20, "opacity": 1.0, "easing": "linear"}, {"time": 2.0, "x": 280, "y": 100, "width": 20, "height": 20, "opacity": 1.0, "easing": "linear"}]}]}

Objective:
%s

Every element needs an id and a keyframe track with time, position, size, opacity, and an easing per segment (linear, ease-in, ease-out, ease-in-out). Answer with one scene JSON object and nothing else.""" % _objective_text(objective)

def composition_retry_prompt(objective, errors, rejected):
    return """Previous scene failed validation. Corrected scene only.

Objective:
%s

Failed checks:
%s

Rejected:
%s

Answer with one corrected scene JSON object and nothing else.""" % (_objective_text(objective), "\n".join("- " + str(e) for e in errors), json.dumps(rejected, indent=2))

def control_prompt(objective, artifact):
    return """Evaluating an animation scene you composed. You cannot perceive the rendered frames in this condition; judge from the scene alone.

Objective:
%s

Scene:
%s

State faults, order revisions if needed, judge whether the work meets the objective.

%s

%s""" % (_objective_text(objective), json.dumps(artifact, indent=2), DECISION_SHAPE_TEXT, REVISION_SHAPE_TEXT)

def perceptual_prompt(objective, evidence, input_summary, structure_record):
    return """Evaluating an animation scene you composed, given evidence from an independent frame analyzer about the rendered frames, the input summary, and the structure record.

Objective:
%s

INPUT (written):
%s

Structure record:
%s

OUTPUT (rendered frames, evidence):
%s

The evidence's comparison section is Sensoryne's judgment of the delivered animation against the written timeline and the objective, and the decision must state how it answers that judgment.

Judge the rendered frames against what was written and the objective.

%s

%s""" % (_objective_text(objective), json.dumps(input_summary, indent=2), json.dumps(structure_record, indent=2), json.dumps(evidence, indent=2), DECISION_SHAPE_TEXT, REVISION_SHAPE_TEXT)

def input_summary(artifact):
    return copy.deepcopy(artifact)

def structure_record(artifact):
    elements = artifact.get("elements", []) if isinstance(artifact, dict) else []
    entries = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        track = sorted((el.get("keyframes") or el.get("track") or []), key=lambda k: float(k.get("time", 0)) if isinstance(k, dict) else 0)
        entries.append({"id": el.get("id"), "keyframe_count": len(track), "times": [k.get("time") for k in track if isinstance(k, dict)], "easings": [k.get("easing") for k in track if isinstance(k, dict)], "start": track[0] if track else None, "end": track[-1] if track else None})
    return {"duration": artifact.get("duration") if isinstance(artifact, dict) else None, "frame_rate": artifact.get("frame_rate") if isinstance(artifact, dict) else None, "canvas": copy.deepcopy(artifact.get("canvas")) if isinstance(artifact, dict) else None, "loop": artifact.get("loop") if isinstance(artifact, dict) else None, "element_count": len(entries), "elements": entries}

def generate_ground_truth(faults):
    lines = []
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_keyframe":
            lines.append("drop_keyframe: keyframe at time %s of element %r is omitted from the rendered frames." % (fault.get("time"), fault.get("id")))
        elif kind == "shift_keyframe_time":
            lines.append("shift_keyframe_time: keyframe at time %s of element %r is rendered at time %s." % (fault.get("time"), fault.get("id"), fault.get("new_time")))
        elif kind == "swap_easings":
            lines.append("swap_easings: the easings of elements %r and %r are exchanged in the rendered frames." % (fault.get("first"), fault.get("second")))
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return lines

def plant_faults(artifact):
    elements = artifact.get("elements", []) if isinstance(artifact, dict) else []
    faults = []
    multi = None
    for el in elements:
        if isinstance(el, dict) and len(el.get("keyframes") or el.get("track") or []) >= 3:
            multi = el
            break
    if multi is not None:
        track = sorted(multi.get("keyframes") or multi.get("track") or [], key=lambda k: float(k.get("time", 0)))
        interior = track[1]
        last_time = float(track[-1].get("time", 0))
        midpoint = float(interior.get("time", 0)) + (last_time - float(interior.get("time", 0))) / 2.0
        faults.append({"kind": "drop_keyframe", "id": multi.get("id"), "time": float(track[-1].get("time", 0))})
        faults.append({"kind": "shift_keyframe_time", "id": multi.get("id"), "time": float(interior.get("time", 0)), "new_time": midpoint})
    if len(elements) >= 2:
        faults.append({"kind": "swap_easings", "first": elements[0].get("id"), "second": elements[1].get("id")})
    return faults

def perceive(render_record):
    paths = render_record.get("frame_paths") or render_record.get("frames") or []
    return perceive_frames(paths, render_record.get("scene"), render_record.get("delivered_elements"), render_record.get("realized_tracks"), bool(render_record.get("loop")))

def _find(revised, element_id):
    if not isinstance(element_id, str) or not element_id:
        raise ValueError("an element id is required")
    for el in revised.get("elements", []):
        if isinstance(el, dict) and el.get("id") == element_id:
            return el
    raise ValueError("element id %r does not exist" % (element_id,))

def _track(el):
    if isinstance(el.get("keyframes"), list):
        return el["keyframes"]
    if isinstance(el.get("track"), list):
        return el["track"]
    el["keyframes"] = []
    return el["keyframes"]

def _find_keyframe(track, time):
    for k in track:
        if isinstance(k, dict) and _finite(k.get("time")) and float(k["time"]) == float(time):
            return k
    return None

def apply_revision(artifact, revision):
    if not isinstance(revision, dict) or "op" not in revision:
        raise ValueError("revision must be an object with an 'op'")
    op = revision["op"]
    revised = copy.deepcopy(artifact)
    if op == "set_keyframe":
        el = _find(revised, revision.get("id")); track = _track(el); time = _num(revision, "time", True)
        keyframe = copy.deepcopy(revision.get("keyframe"))
        if not isinstance(keyframe, dict):
            raise ValueError("set_keyframe requires a keyframe object")
        keyframe.setdefault("time", time)
        errors = _check_keyframe(keyframe)
        if errors or float(keyframe.get("time")) != time:
            raise ValueError("; ".join(errors) or "keyframe time must match the revision time")
        existing = _find_keyframe(track, time)
        if existing is None:
            track.append(keyframe)
        else:
            track[track.index(existing)] = keyframe
        track.sort(key=lambda k: float(k.get("time", 0)))
    elif op == "add_keyframe":
        el = _find(revised, revision.get("id")); track = _track(el)
        keyframe = copy.deepcopy(revision.get("keyframe"))
        errors = _check_keyframe(keyframe)
        if errors:
            raise ValueError("; ".join(errors))
        if _find_keyframe(track, float(keyframe["time"])) is not None:
            raise ValueError("a keyframe already exists at that time")
        track.append(keyframe); track.sort(key=lambda k: float(k.get("time", 0)))
    elif op == "remove_keyframe":
        el = _find(revised, revision.get("id")); track = _track(el); time = _num(revision, "time", True)
        existing = _find_keyframe(track, time)
        if existing is None:
            raise ValueError("no keyframe at that time")
        track.remove(existing)
    elif op == "set_easing":
        el = _find(revised, revision.get("id")); track = _track(el); time = _num(revision, "time", True)
        easing = revision.get("easing")
        if easing not in EASINGS:
            raise ValueError("easing must be one of linear, ease-in, ease-out, ease-in-out")
        existing = _find_keyframe(track, time)
        if existing is None:
            raise ValueError("no keyframe at that time")
        existing["easing"] = easing
    else:
        raise ValueError("unknown revision op: %r" % (op,))
    errors = validate(revised)
    if errors:
        raise ValueError("resulting scene failed validation: %s" % "; ".join(errors))
    return revised

def judge_brief():
    return """Judge this animation on the timeline and the motion as written against what was delivered. Ground every claim in facts in the pack: the written work, the comparison findings, the objective coverage, and the maker's decisions. Where the record carries no fact, say the record does not show it; never invent one. Judge the work itself, not delivery alone: the delivery findings are facts to weigh, and the question is whether the work is good. Record the verdict with bottom_line, judgment, strengths, and weaknesses."""


def apply_revisions(artifact, revisions):
    current = copy.deepcopy(artifact)
    for revision in revisions or []:
        current = apply_revision(current, revision)
    return current
