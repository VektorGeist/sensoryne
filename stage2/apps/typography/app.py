"""stage2/apps/typography/app.py - the typography application.

Sensoryne receives the delivered page produced on the composing
side.  Inside Sensoryne this application keeps the artifact
format and validation, revisions, the raster analyzer adapter,
the input summary and structure record, delivery-fault
definitions and ground-truth text, and the prompts.  Delivery
faults are definitions recorded here and applied by the
composing side; the delivered page Sensoryne receives already
carries them.
"""
import copy, json, math
from ...decision import DECISION_SHAPE_TEXT
from .instrument import perceive_page

NAME = "typography"
KINDS = ("heading", "body", "caption", "image")
ALIGNMENTS = ("left", "center", "right")
REVISION_SHAPE_TEXT = """Revisions addressed by element id:
- {"op": "set_position", "id": "title", "x": 50, "y": 40}: set position.
- {"op": "set_font_size", "id": "title", "font_size": 36}: set font size.
- {"op": "set_alignment", "id": "title", "alignment": "center"}: set alignment.
- {"op": "add_element", "element": {"id": "note", "kind": "caption", "text": "...", "font_size": 12, "x": 50, "y": 900, "width": 700, "height": 20, "alignment": "left"}}: add one element.
- {"op": "remove_element", "id": "note"}: remove one element."""

def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))

def _num(d, key, required=False):
    if key not in d or d.get(key) is None:
        if required:
            raise ValueError("%s requires %r" % (d.get("op"), key))
        return None
    if not _finite(d[key]):
        raise ValueError("%r must be finite" % key)
    return float(d[key])

def _check_element(el):
    errors = []
    if not isinstance(el, dict):
        return ["element must be an object"]
    if not isinstance(el.get("id"), str) or not el["id"].strip():
        errors.append("element id must be a non-empty string")
    if el.get("kind") not in KINDS:
        errors.append("element kind must be one of heading, body, caption, image")
    if el.get("kind") != "image" and (not isinstance(el.get("text"), str) or not el["text"].strip()):
        errors.append("text element id %r must hold non-empty text" % el.get("id"))
    if not _finite(el.get("font_size")) or float(el.get("font_size") or 0) <= 0:
        errors.append("element id %r font_size must be positive" % el.get("id"))
    for key in ("x", "y", "width", "height"):
        if not _finite(el.get(key)):
            errors.append("element id %r %s must be finite" % (el.get("id"), key))
    if _finite(el.get("width")) and float(el.get("width")) <= 0:
        errors.append("element id %r width must be positive" % el.get("id"))
    if _finite(el.get("height")) and float(el.get("height")) <= 0:
        errors.append("element id %r height must be positive" % el.get("id"))
    if el.get("alignment") not in ALIGNMENTS:
        errors.append("element id %r alignment must be left, center, or right" % el.get("id"))
    return errors

def validate(artifact):
    if not isinstance(artifact, dict) or "elements" not in artifact:
        return ["page artifact must hold an elements list"]
    errors = []
    page = artifact.get("page")
    if not isinstance(page, dict) or not _finite(page.get("width")) or float(page.get("width") or 0) <= 0 or not _finite(page.get("height")) or float(page.get("height") or 0) <= 0:
        errors.append("page must hold positive width and height")
    elements = artifact.get("elements")
    if not isinstance(elements, list) or not elements:
        return errors + ["page must hold at least one element"]
    seen = set()
    for el in elements:
        errors.extend(_check_element(el))
        if isinstance(el, dict) and el.get("id"):
            if el["id"] in seen:
                errors.append("duplicate element id %r" % el["id"])
            seen.add(el["id"])
    return errors

def _objective_text(objective):
    return json.dumps(objective, indent=2) if isinstance(objective, dict) else str(objective)

def composition_prompt(objective):
    return """Create a page layout meeting the objective. Answer with the artifact JSON only:
{"page": {"width": 800, "height": 1000, "background": "#ffffff"}, "elements": [{"id": "title", "kind": "heading", "text": "...", "font_size": 32, "x": 50, "y": 40, "width": 700, "height": 50, "alignment": "left"}]}

Objective:
%s

Use kinds heading, body, caption, image. Every element needs id, kind, font_size, x, y, width, height, alignment. Answer with one page JSON object and nothing else.""" % _objective_text(objective)

def composition_retry_prompt(objective, errors, rejected):
    return """Previous page failed validation. Corrected page only.

Objective:
%s

Failed checks:
%s

Rejected:
%s

Answer with one corrected page JSON object and nothing else.""" % (_objective_text(objective), "\n".join("- " + str(e) for e in errors), json.dumps(rejected, indent=2))

def control_prompt(objective, artifact):
    return """Evaluating a page you composed. You cannot perceive the drawn page in this condition; judge from the page alone.

Objective:
%s

Page:
%s

State faults, order revisions if needed, judge whether the work meets the objective.

%s

%s""" % (_objective_text(objective), json.dumps(artifact, indent=2), DECISION_SHAPE_TEXT, REVISION_SHAPE_TEXT)

def perceptual_prompt(objective, evidence, input_summary, structure_record):
    return """Evaluating a page you composed, given evidence from an independent raster analyzer about the drawn page, the input summary, and the structure record.

Objective:
%s

INPUT (written):
%s

Structure record:
%s

OUTPUT (drawn page, evidence):
%s

The evidence's comparison section is Sensoryne's judgment of the delivered layout against the written declarations and the objective, and the decision must state how it answers that judgment.

Judge the drawn page against what was written and the objective.

%s

%s""" % (_objective_text(objective), json.dumps(input_summary, indent=2), json.dumps(structure_record, indent=2), json.dumps(evidence, indent=2), DECISION_SHAPE_TEXT, REVISION_SHAPE_TEXT)

def input_summary(artifact):
    return copy.deepcopy(artifact)

def structure_record(artifact):
    elements = artifact.get("elements", []) if isinstance(artifact, dict) else []
    entries = [{"id": e.get("id"), "kind": e.get("kind"), "box": {"x": e.get("x"), "y": e.get("y"), "width": e.get("width"), "height": e.get("height")}, "font_size": e.get("font_size"), "alignment": e.get("alignment")} for e in elements if isinstance(e, dict)]
    def median(kind):
        vals = sorted(float(e.get("font_size") or 0) for e in elements if isinstance(e, dict) and e.get("kind") == kind)
        return float(vals[len(vals)//2]) if vals else None
    heading = median("heading"); body = median("body"); caption = median("caption")
    return {"page": copy.deepcopy(artifact.get("page")) if isinstance(artifact, dict) else None, "element_count": len(entries), "elements": entries, "left_edges": sorted({e.get("x") for e in entries}), "type_hierarchy": {"heading_median": heading, "body_median": body, "caption_median": caption, "heading_body_ratio": (heading/body if heading and body else None), "body_caption_ratio": (body/caption if body and caption else None)}}

def generate_ground_truth(faults):
    lines = []
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "shift_element":
            lines.append("shift_element: element %r is moved by dx=%s, dy=%s off its written position." % (fault.get("id"), fault.get("dx"), fault.get("dy")))
        elif kind in ("shrink_heading", "shrink_element"):
            lines.append("shrink_element: element %r is drawn at font size %s instead of its written size." % (fault.get("id"), fault.get("font_size")))
        elif kind == "move_into_margin":
            lines.append("move_into_margin: element %r is moved to x=%s, y=%s in the page margin band, off the shared alignment edge." % (fault.get("id"), fault.get("x"), fault.get("y")))
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return lines

def plant_faults(artifact):
    elements = artifact.get("elements", []) if isinstance(artifact, dict) else []
    faults = []
    heading = next((e for e in elements if isinstance(e, dict) and e.get("kind") == "heading"), None)
    body_sizes = [float(e.get("font_size")) for e in elements if isinstance(e, dict) and e.get("kind") == "body" and _finite(e.get("font_size"))]
    if heading is not None and body_sizes:
        faults.append({"kind": "shrink_heading", "id": heading.get("id"), "font_size": body_sizes[0]})
    if elements:
        faults.append({"kind": "shift_element", "id": elements[0].get("id"), "dx": 30, "dy": 0})
    if len(elements) >= 2:
        faults.append({"kind": "move_into_margin", "id": elements[1].get("id"), "x": 2, "y": 2})
    return faults

def perceive(render_record):
    return perceive_page(render_record["path"], render_record.get("page"), render_record.get("delivered_elements"), render_record.get("realized_boxes"))

def _find(revised, element_id):
    if not isinstance(element_id, str) or not element_id:
        raise ValueError("an element id is required")
    for el in revised.get("elements", []):
        if isinstance(el, dict) and el.get("id") == element_id:
            return el
    raise ValueError("element id %r does not exist" % (element_id,))

def apply_revision(artifact, revision):
    if not isinstance(revision, dict) or "op" not in revision:
        raise ValueError("revision must be an object with an 'op'")
    op = revision["op"]
    revised = copy.deepcopy(artifact)
    if op == "set_position":
        el = _find(revised, revision.get("id")); el["x"] = _num(revision, "x", True); el["y"] = _num(revision, "y", True)
    elif op == "set_font_size":
        el = _find(revised, revision.get("id")); size = _num(revision, "font_size", True)
        if size <= 0:
            raise ValueError("font_size must be positive")
        el["font_size"] = size
    elif op == "set_alignment":
        el = _find(revised, revision.get("id")); alignment = revision.get("alignment")
        if alignment not in ALIGNMENTS:
            raise ValueError("alignment must be left, center, or right")
        el["alignment"] = alignment
    elif op == "add_element":
        element = copy.deepcopy(revision.get("element"))
        errors = _check_element(element)
        if errors:
            raise ValueError("; ".join(errors))
        if any(isinstance(e, dict) and e.get("id") == element.get("id") for e in revised.get("elements", [])):
            raise ValueError("element id %r already exists" % element.get("id"))
        revised.setdefault("elements", []).append(element)
    elif op == "remove_element":
        el = _find(revised, revision.get("id")); revised["elements"].remove(el)
    else:
        raise ValueError("unknown revision op: %r" % (op,))
    errors = validate(revised)
    if errors:
        raise ValueError("resulting page failed validation: %s" % "; ".join(errors))
    return revised

def judge_brief():
    return """Judge this page on its layout, its hierarchy, and its readability. Ground every claim in facts in the pack: the written work, the comparison findings, the objective coverage, and the maker's decisions. Where the record carries no fact, say the record does not show it; never invent one. Judge the work itself, not delivery alone: the delivery findings are facts to weigh, and the question is whether the work is good. Record the verdict with bottom_line, judgment, strengths, and weaknesses."""


def apply_revisions(artifact, revisions):
    current = copy.deepcopy(artifact)
    for revision in revisions or []:
        current = apply_revision(current, revision)
    return current
