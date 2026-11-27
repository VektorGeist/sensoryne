"""stage2/apps/drawing/app.py - the drawing application.

Sensoryne receives the rendered picture produced on the
composing side.  Inside Sensoryne this application keeps the
artifact format and validation, revisions, the raster analyzer
adapter, the input summary and structure record, delivery-fault
definitions and ground-truth text, and the prompts.  Delivery
faults are definitions recorded here and applied by the
composing side in the rendering; the rendered picture Sensoryne
receives already carries them.
"""

import copy
import json
import re
import xml.etree.ElementTree as ET

from ...decision import DECISION_SHAPE_TEXT
from .instrument import perceive_image

NAME = "drawing"
SVG_RENDERER = "cairosvg"

REVISION_SHAPE_TEXT = """Revisions are operations applied to the drawing, addressed by element id, in order. Each revision is validated against the parsed SVG, the result is re-parsed and re-validated, and a rejected revision leaves the current drawing unchanged.
- {"op": "set_attribute", "id": "sun", "attribute": "cx", "value": "120"}: set one attribute on the element with that id.
- {"op": "replace_element", "id": "sun", "svg": "<circle id=\\"sun\\" .../>" }: replace the element with that id with the given SVG element.
- {"op": "add_element", "svg": "<circle id=\\"moon\\" .../>" }: add one SVG element.
- {"op": "remove_element", "id": "sun"}: remove the element with that id."""


def _local_name(tag):
    return str(tag).rsplit("}", 1)[-1]


def _parse_svg(svg):
    return ET.fromstring(svg)


def _collect(svg):
    root = _parse_svg(svg)
    elements = []
    for element in root.iter():
        element_id = element.get("id")
        if element_id:
            elements.append(
                {
                    "id": element_id,
                    "kind": _local_name(element.tag),
                    "tag": _local_name(element.tag),
                    "attributes": dict(element.attrib),
                }
            )
    return root, elements


def validate(artifact):
    """Structure only: an object with an svg string, the
    SVG parses as XML, and the root element is svg."""
    errors = []
    if not isinstance(artifact, dict) or "svg" not in artifact:
        return ["drawing must be an object with an 'svg' string"]
    svg = artifact.get("svg")
    if not isinstance(svg, str) or not svg.strip():
        return ["drawing 'svg' must be a non-empty string"]
    try:
        root, elements = _collect(svg)
    except ET.ParseError as exc:
        return ["SVG does not parse as XML: %s" % exc]
    if _local_name(root.tag) != "svg":
        errors.append("root element of the SVG must be 'svg'")
    return errors


def _objective_text(objective):
    if isinstance(objective, dict):
        return json.dumps(objective, indent=2)
    return str(objective)


def composition_prompt(objective):
    return """Create a drawing meeting the objective's intent. Answer with the artifact JSON only, in this format and nothing else:
{"svg": "<svg ...>...</svg>", "elements": [{"id": "...", "kind": "shape or text", "description": "what it is and where it sits"}]}

The objective states the picture's intent in words, plus the intended elements list.

Objective:
%s

Answer with one drawing JSON object in the format {"svg": "<svg ...>...</svg>", "elements": [...]} and nothing else.""" % _objective_text(objective)


def composition_retry_prompt(objective, errors, rejected):
    return """Your previous drawing did not pass validation. Please provide a corrected drawing.

Objective:
%s

Failed checks:
%s

Rejected drawing:
%s

Answer with one corrected drawing JSON object in the format {"svg": "<svg ...>...</svg>", "elements": [...]} and nothing else.""" % (
        _objective_text(objective),
        "\n".join("- " + str(e) for e in errors),
        json.dumps(rejected, indent=2),
    )


def control_prompt(objective, artifact):
    return """You are evaluating a drawing you composed. You cannot perceive the rendered picture in this condition; judge from the drawing alone.

Objective:
%s

Drawing:
%s

State any faults you find in the drawing itself, order revisions if needed, and judge whether the work meets the objective.

%s

%s""" % (
        _objective_text(objective),
        json.dumps(artifact, indent=2),
        DECISION_SHAPE_TEXT,
        REVISION_SHAPE_TEXT,
    )


def perceptual_prompt(objective, evidence, input_summary, structure_record):
    return """You are evaluating a drawing you composed, given evidence from an independent raster analyzer about the rendered picture, the input summary, and the structure record of what you wrote.

Objective:
%s

INPUT (what was composed): the SVG source and intended elements as written, below.

Input summary:
%s

Structure record (every element by id and kind, with position and size from its attributes, canvas size, and palette of colors used):
%s

OUTPUT (what was rendered): the rendered picture, with the analyzer's full evidence about it below. The analyzer reports measurements only; it does not say what the picture shows. Judge the rendered picture against what was written and the objective.

Evidence:
%s

The evidence's comparison section is Sensoryne's judgment of the delivered drawing against the written declarations and the objective, and the decision must state how it answers that judgment.

State any faults you find, order revisions if needed, and judge whether the rendered work meets the objective.

%s

%s""" % (
        _objective_text(objective),
        json.dumps(input_summary, indent=2),
        json.dumps(structure_record, indent=2),
        json.dumps(evidence, indent=2),
        DECISION_SHAPE_TEXT,
        REVISION_SHAPE_TEXT,
    )


def input_summary(artifact):
    """The SVG source and intended elements exactly as written."""
    return copy.deepcopy(artifact)


def _palette(root):
    colors = []
    seen = set()
    for element in root.iter():
        for key, value in element.attrib.items():
            if key == "fill" or key == "stroke" or key.endswith("color"):
                value = str(value).strip()
                if value and value not in seen:
                    seen.add(value)
                    colors.append(value)
        style = element.get("style", "")
        for match in re.findall(r"(?:fill|stroke)\s*:\s*([^;]+)", style):
            value = match.strip()
            if value and value not in seen:
                seen.add(value)
                colors.append(value)
    return colors


def structure_record(artifact):
    try:
        root, elements = _collect(artifact.get("svg", ""))
    except Exception:
        return {
            "elements": [],
            "canvas_size": None,
            "viewBox": None,
            "palette": [],
        }
    entries = []
    for element in elements:
        attributes = element["attributes"]
        entries.append(
            {
                "id": element["id"],
                "kind": element["kind"],
                "position": {
                    key: attributes[key]
                    for key in ("x", "y", "cx", "cy", "x1", "y1", "x2", "y2")
                    if key in attributes
                },
                "size": {
                    key: attributes[key]
                    for key in ("width", "height", "r", "rx", "ry")
                    if key in attributes
                },
                "attributes": attributes,
            }
        )
    return {
        "elements": entries,
        "canvas_size": root.get("viewBox"),
        "viewBox": root.get("viewBox"),
        "palette": _palette(root),
    }


def generate_ground_truth(faults):
    """One ground-truth string per fault, from the definitions."""
    lines = []
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_element":
            lines.append(
                "drop_element: element %r is omitted from the rendered "
                "picture." % fault.get("id")
            )
        elif kind == "shift_element":
            lines.append(
                "shift_element: element %r is moved by dx=%s, dy=%s in "
                "the rendered picture."
                % (fault.get("id"), fault.get("dx"), fault.get("dy"))
            )
        elif kind == "recolor":
            lines.append(
                "recolor: element %r is rendered with color %r instead "
                "of its written color."
                % (fault.get("id"), fault.get("color"))
            )
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return lines


def plant_faults(artifact):
    """Return the fault set for the given work, deterministically,
    using only defined kinds and only element ids that exist."""
    faults = []
    try:
        _root, elements = _collect(artifact.get("svg", ""))
    except Exception:
        return faults
    ids = [entry["id"] for entry in elements]
    if len(ids) >= 1:
        faults.append({"kind": "drop_element", "id": ids[0]})
    if len(ids) >= 2:
        faults.append({"kind": "shift_element", "id": ids[1], "dx": 20, "dy": 10})
        faults.append({"kind": "recolor", "id": ids[1], "color": "#ff0000"})
    return faults


def perceive(render_record):
    return perceive_image(render_record["path"])


def _find_by_id(root, element_id):
    for element in root.iter():
        if element.get("id") == element_id:
            return element
    return None


def _find_parent(root, target):
    for parent in root.iter():
        for child in list(parent):
            if child is target:
                return parent
    return None


def _require_id(revision):
    element_id = revision.get("id")
    if not isinstance(element_id, str) or not element_id:
        raise ValueError("%s requires a non-empty 'id'" % revision.get("op"))
    return element_id


def apply_revision(artifact, revision):
    """Return a revised deep copy; input never modified."""
    if not isinstance(revision, dict) or "op" not in revision:
        raise ValueError("revision must be an object with an 'op'")
    op = revision["op"]
    revised = copy.deepcopy(artifact)
    svg = revised.get("svg")
    if not isinstance(svg, str):
        raise ValueError("drawing has no SVG")
    try:
        root = _parse_svg(svg)
    except ET.ParseError as exc:
        raise ValueError("SVG does not parse as XML: %s" % exc)

    if op == "set_attribute":
        element_id = _require_id(revision)
        element = _find_by_id(root, element_id)
        if element is None:
            raise ValueError("element id %r does not exist" % element_id)
        attribute = revision.get("attribute")
        if not isinstance(attribute, str) or not attribute:
            raise ValueError("set_attribute requires an 'attribute'")
        if "value" not in revision:
            raise ValueError("set_attribute requires a 'value'")
        element.set(attribute, str(revision["value"]))
    elif op == "replace_element":
        element_id = _require_id(revision)
        element = _find_by_id(root, element_id)
        if element is None:
            raise ValueError("element id %r does not exist" % element_id)
        replacement_svg = revision.get("svg")
        if not isinstance(replacement_svg, str) or not replacement_svg.strip():
            raise ValueError("replace_element requires an 'svg' element")
        try:
            replacement = _parse_svg(replacement_svg)
        except ET.ParseError as exc:
            raise ValueError("replacement SVG does not parse as XML: %s" % exc)
        parent = _find_parent(root, element)
        if parent is None:
            raise ValueError("cannot replace the root element")
        index = list(parent).index(element)
        parent.remove(element)
        parent.insert(index, replacement)
    elif op == "add_element":
        addition_svg = revision.get("svg")
        if not isinstance(addition_svg, str) or not addition_svg.strip():
            raise ValueError("add_element requires an 'svg' element")
        try:
            addition = _parse_svg(addition_svg)
        except ET.ParseError as exc:
            raise ValueError("added SVG does not parse as XML: %s" % exc)
        root.append(addition)
    elif op == "remove_element":
        element_id = _require_id(revision)
        element = _find_by_id(root, element_id)
        if element is None:
            raise ValueError("element id %r does not exist" % element_id)
        parent = _find_parent(root, element)
        if parent is None:
            raise ValueError("cannot remove the root element")
        parent.remove(element)
    else:
        raise ValueError("unknown revision op: %r" % (op,))

    revised["svg"] = ET.tostring(root, encoding="unicode")
    try:
        _parse_svg(revised["svg"])
    except ET.ParseError as exc:
        raise ValueError("resulting SVG does not parse as XML: %s" % exc)
    errors = validate(revised)
    if errors:
        raise ValueError(
            "resulting drawing failed validation: %s" % "; ".join(errors)
        )
    return revised


def judge_brief():
    return """Judge this drawing on its composition and whether the declared scene is realized. Ground every claim in facts in the pack: the written work, the comparison findings, the objective coverage, and the maker's decisions. Where the record carries no fact, say the record does not show it; never invent one. Judge the work itself, not delivery alone: the delivery findings are facts to weigh, and the question is whether the work is good. Record the verdict with bottom_line, judgment, strengths, and weaknesses."""


def apply_revisions(artifact, revisions):
    current = copy.deepcopy(artifact)
    for revision in revisions or []:
        current = apply_revision(current, revision)
    return current
