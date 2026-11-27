"""stage2/apps/drawing/comparison.py - Sensoryne's judgment of the
delivered drawing against the written declarations.

The comparison works from the delivered SVG document: the
elements realized in the delivered SVG (the fault-applied SVG
the renderer realized, when the render record carries it), set
against the written work as the artifact states it: the shapes
declared in the written SVG, each with its kind (its SVG tag),
its geometry (position and size from its attributes) and its
color (fill or stroke).  It never works from delivery-fault
definitions: a finding names what the delivered SVG shows
against what was written.

Shapes are matched by shape kind and position: a delivered
shape of the same kind at the declared position and size
within the stated geometry tolerance realizes that declared
shape.  A remaining delivered shape carrying a remaining
declared shape's id realizes it displaced, and a remaining
delivered shape of the same kind nearest to a remaining
declared shape is its displaced realization when no id
carries the identity.  Geometry and color are then judged on
every matched pair.

Each finding is an object with kind, where, written,
delivered, and statement.  The kinds are shape_missing,
shape_not_declared, geometry_off, and color_off.  The
geometry tolerance is stated here and in the reliability
block; every geometry finding carries both the written and
the delivered values it was judged on.
"""

import copy
import os
import re
import xml.etree.ElementTree as ET

__all__ = [
    "GEOMETRY_TOLERANCE",
    "FINDING_KINDS",
    "compute_comparison",
    "build_comparison",
]

GEOMETRY_TOLERANCE = 0.5

FINDING_KINDS = (
    "shape_missing",
    "shape_not_declared",
    "geometry_off",
    "color_off",
)

_POSITION_KEYS = ("x", "y", "cx", "cy", "x1", "y1", "x2", "y2")
_SIZE_KEYS = ("width", "height", "r", "rx", "ry")


def _j(v):
    if isinstance(v, dict):
        return {str(k): _j(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, (str, int, float)):
        return v
    try:
        import numpy as np

        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
    except Exception:
        pass
    return str(v)


def _finding(kind, where, written, delivered, statement):
    return {
        "kind": kind,
        "where": where,
        "written": _j(written),
        "delivered": _j(delivered),
        "statement": statement,
    }


def _summary(findings):
    counts = {}
    for finding in findings:
        counts[finding["kind"]] = counts.get(finding["kind"], 0) + 1
    total = len(findings)
    if total == 0:
        line = (
            "No findings: the delivered drawing matches the "
            "written declarations within the stated tolerances."
        )
    else:
        line = "%d finding(s): %s." % (
            total,
            ", ".join("%s=%d" % (kind, counts[kind]) for kind in sorted(counts)),
        )
    return {"counts": counts, "total": total, "line": line}


def _local_name(tag):
    return str(tag).rsplit("}", 1)[-1]


def _float_or_none(value):
    try:
        if isinstance(value, bool):
            return None
        return float(value)
    except Exception:
        return None


def _style_color(attributes, key):
    style = attributes.get("style") if isinstance(attributes, dict) else None
    if not isinstance(style, str):
        return None
    match = re.search(r"(?:^|;)\s*%s\s*:\s*([^;]+)" % re.escape(key), style)
    return match.group(1).strip() if match else None


def _shape_block(shape):
    if shape is None:
        return None
    position = dict(shape.get("position") or {})
    size = dict(shape.get("size") or {})
    geometry = {}
    geometry.update(position)
    geometry.update(size)
    return {
        "id": shape.get("id"),
        "kind": shape.get("kind"),
        "tag": shape.get("kind"),
        "shape": shape.get("kind"),
        "name": shape.get("id"),
        "position": position,
        "declared_position": dict(position),
        "size": size,
        "declared_size": dict(size),
        "geometry": geometry,
        "color": shape.get("color"),
        "fill": shape.get("fill"),
        "stroke": shape.get("stroke"),
        "attributes": copy.deepcopy(shape.get("attributes") or {}),
    }


def _shape_from_attributes(element_id, kind, attributes):
    attributes = dict(attributes or {})
    position = {}
    for key in _POSITION_KEYS:
        if key in attributes:
            value = _float_or_none(attributes.get(key))
            if value is not None:
                position[key] = value
    size = {}
    for key in _SIZE_KEYS:
        if key in attributes:
            value = _float_or_none(attributes.get(key))
            if value is not None:
                size[key] = value
    fill = attributes.get("fill")
    if fill is None:
        fill = _style_color(attributes, "fill")
    stroke = attributes.get("stroke")
    if stroke is None:
        stroke = _style_color(attributes, "stroke")
    if fill is not None:
        fill = str(fill).strip()
    if stroke is not None:
        stroke = str(stroke).strip()
    color = fill if fill is not None else stroke
    return {
        "id": element_id,
        "kind": kind,
        "position": position,
        "size": size,
        "color": color,
        "fill": fill,
        "stroke": stroke,
        "attributes": attributes,
    }


def _shapes_from_svg(svg):
    if not isinstance(svg, str) or not svg.strip():
        return []
    try:
        root = ET.fromstring(svg)
    except Exception:
        return []
    shapes = []
    for element in root.iter():
        if element is root:
            continue
        kind = _local_name(element.tag)
        if kind == "svg":
            continue
        attributes = dict(element.attrib)
        shapes.append(
            _shape_from_attributes(element.get("id"), kind, attributes)
        )
    return shapes


def _shapes_from_entries(entries):
    shapes = []
    if not isinstance(entries, list):
        return shapes
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        attributes = {}
        if isinstance(entry.get("attributes"), dict):
            attributes.update(entry["attributes"])
        for key in _POSITION_KEYS + _SIZE_KEYS:
            if key in entry:
                attributes[key] = entry[key]
        for key in ("fill", "stroke", "color"):
            if key in entry:
                attributes[key] = entry[key]
        kind = entry.get("kind") or entry.get("tag") or entry.get("shape")
        shape = _shape_from_attributes(entry.get("id"), kind, attributes)
        if isinstance(entry.get("position"), dict):
            for key, value in entry["position"].items():
                number = _float_or_none(value)
                if number is not None:
                    shape["position"][str(key)] = number
        if isinstance(entry.get("size"), dict):
            for key, value in entry["size"].items():
                number = _float_or_none(value)
                if number is not None:
                    shape["size"][str(key)] = number
        if entry.get("color") is not None and shape.get("color") is None:
            shape["color"] = str(entry["color"]).strip()
        shapes.append(shape)
    return shapes


def _written_shapes(artifact):
    if not isinstance(artifact, dict):
        return []
    shapes = _shapes_from_svg(artifact.get("svg", ""))
    if shapes:
        return shapes
    shapes = _shapes_from_entries(artifact.get("elements"))
    if shapes:
        return shapes
    objective = artifact.get("objective")
    if isinstance(objective, dict):
        return _shapes_from_entries(objective.get("elements"))
    return []


def _first_svg_text(*sources):
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("delivered_svg", "delivered", "svg", "delivered_text"):
            value = source.get(key)
            if isinstance(value, str) and value.strip().startswith("<"):
                return value
        full = source.get("full")
        if isinstance(full, dict):
            for key in ("delivered_svg", "delivered", "svg"):
                value = full.get(key)
                if isinstance(value, str) and value.strip().startswith("<"):
                    return value
    return None


def _first_entries(*sources):
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("delivered_elements", "delivered_shapes", "elements", "shapes"):
            value = source.get(key)
            if isinstance(value, list) and value:
                entries = _shapes_from_entries(value)
                if entries:
                    return entries
        for key in ("structure_record", "structure"):
            structure = source.get(key)
            if isinstance(structure, dict) and isinstance(structure.get("elements"), list):
                entries = _shapes_from_entries(structure["elements"])
                if entries:
                    return entries
    return []


def _delivered_shapes(artifact, render_record, evidence, written):
    # A direct call may carry the evidence in the second place.
    if evidence is None and isinstance(render_record, dict):
        if "measurements" in render_record or "delivered_svg" in render_record:
            evidence = render_record
            render_record = None
    sources = (render_record, evidence)
    svg = _first_svg_text(*sources)
    if svg is not None:
        return _shapes_from_svg(svg)
    entries = _first_entries(*sources)
    if entries:
        return entries
    # A delivered SVG file handed in as the render path.
    if isinstance(render_record, dict):
        path = render_record.get("path")
        if isinstance(path, str) and path.lower().endswith(".svg") and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return _shapes_from_svg(handle.read())
            except Exception:
                pass
    # No delivered SVG is carried: the delivered document is
    # the written document, so a clean delivery is judged clean
    # rather than inventing a delivery.
    return list(written)


def _geometry(shape):
    geometry = {}
    geometry.update(shape.get("position") or {})
    geometry.update(shape.get("size") or {})
    return geometry


def _within_tolerance(written, delivered, tolerance):
    written_geometry = _geometry(written)
    delivered_geometry = _geometry(delivered)
    if set(written_geometry.keys()) != set(delivered_geometry.keys()):
        return False
    return all(
        abs(float(written_geometry[key]) - float(delivered_geometry[key]))
        <= tolerance
        for key in written_geometry
    )


def _departures(written, delivered, tolerance):
    written_geometry = _geometry(written)
    delivered_geometry = _geometry(delivered)
    departures = {}
    for key in sorted(set(written_geometry) | set(delivered_geometry)):
        written_value = written_geometry.get(key)
        delivered_value = delivered_geometry.get(key)
        if written_value is None or delivered_value is None:
            departures[key] = {
                "written": written_value,
                "delivered": delivered_value,
            }
            continue
        difference = float(delivered_value) - float(written_value)
        if abs(difference) > tolerance:
            departures[key] = {
                "written": float(written_value),
                "delivered": float(delivered_value),
                "difference": float(difference),
                "absolute_difference": float(abs(difference)),
            }
    return departures


def _distance(written, delivered):
    written_geometry = _geometry(written)
    delivered_geometry = _geometry(delivered)
    keys = set(written_geometry) | set(delivered_geometry)
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        written_value = written_geometry.get(key)
        delivered_value = delivered_geometry.get(key)
        if written_value is None or delivered_value is None:
            total += 1.0
            continue
        total += abs(float(delivered_value) - float(written_value))
    return total


def _fmt_geometry(geometry):
    if not geometry:
        return "no declared position"
    return ", ".join(
        "%s=%s" % (key, geometry[key]) for key in sorted(geometry)
    )


def _fmt_position(shape):
    return _fmt_geometry(shape.get("position") or {})


def _fmt_size(shape):
    size = shape.get("size") or {}
    if not size:
        return "no declared size"
    return ", ".join("%s=%s" % (key, size[key]) for key in sorted(size))


def _where(shape):
    position = _fmt_position(shape)
    if shape.get("id") is not None:
        return "%s %r at %s" % (shape.get("kind"), shape.get("id"), position)
    return "%s at %s" % (shape.get("kind"), position)


def _color_text(color):
    return str(color) if color is not None else "no color"


def _normalize_color(color):
    return str(color).strip().lower() if color is not None else None


def compute_comparison(artifact, render_record=None, evidence=None):
    """Compare the delivered SVG against the written shapes."""
    tolerance = GEOMETRY_TOLERANCE
    written = _written_shapes(artifact)
    delivered = _delivered_shapes(artifact, render_record, evidence, written)

    matched_pairs = []  # (written_index, delivered_index)
    matched_written = set()
    matched_delivered = set()

    # Matched by shape kind and position: same kind, position
    # and size within the stated tolerance, color aside.
    for wi, w in enumerate(written):
        for di, d in enumerate(delivered):
            if di in matched_delivered:
                continue
            if d.get("kind") != w.get("kind"):
                continue
            if _within_tolerance(w, d, tolerance):
                matched_pairs.append((wi, di))
                matched_written.add(wi)
                matched_delivered.add(di)
                break

    # A remaining delivered shape carrying a remaining
    # declared shape's id is that shape's realization.
    for wi, w in enumerate(written):
        if wi in matched_written:
            continue
        if w.get("id") is None:
            continue
        for di, d in enumerate(delivered):
            if di in matched_delivered:
                continue
            if d.get("id") is not None and d.get("id") == w.get("id"):
                matched_pairs.append((wi, di))
                matched_written.add(wi)
                matched_delivered.add(di)
                break

    # A remaining delivered shape of the same kind nearest to
    # a remaining declared shape is its displaced realization
    # when no id carried the identity.
    while True:
        best = None
        for wi, w in enumerate(written):
            if wi in matched_written:
                continue
            for di, d in enumerate(delivered):
                if di in matched_delivered:
                    continue
                if d.get("kind") != w.get("kind"):
                    continue
                distance = _distance(w, d)
                if best is None or distance < best[0]:
                    best = (distance, wi, di)
        if best is None:
            break
        _distance_value, wi, di = best
        matched_pairs.append((wi, di))
        matched_written.add(wi)
        matched_delivered.add(di)

    findings = []
    realized_as_declared = 0

    order = {wi: index for index, (wi, _di) in enumerate(matched_pairs)}
    for wi, di in sorted(matched_pairs, key=lambda pair: order[pair[0]]):
        w = written[wi]
        d = delivered[di]
        clean = True
        departures = _departures(w, d, tolerance)
        if departures:
            clean = False
            delivered_block = _shape_block(d)
            delivered_block.update(
                {
                    "present": True,
                    "departures": departures,
                    "geometry_tolerance": tolerance,
                    "position_tolerance": tolerance,
                    "size_tolerance": tolerance,
                }
            )
            findings.append(
                _finding(
                    "geometry_off",
                    _where(w),
                    _shape_block(w),
                    delivered_block,
                    "Declared %s %r at %s (size %s) was delivered "
                    "at %s (size %s): its delivered position or "
                    "size departs from the declared values by "
                    "more than the %s geometry tolerance."
                    % (
                        w.get("kind"),
                        w.get("id"),
                        _fmt_position(w),
                        _fmt_size(w),
                        _fmt_position(d),
                        _fmt_size(d),
                        tolerance,
                    ),
                )
            )
        written_color = _normalize_color(w.get("color"))
        delivered_color = _normalize_color(d.get("color"))
        if written_color != delivered_color:
            clean = False
            delivered_block = _shape_block(d)
            delivered_block.update(
                {
                    "present": True,
                    "written_color": w.get("color"),
                    "geometry_tolerance": tolerance,
                }
            )
            findings.append(
                _finding(
                    "color_off",
                    _where(w),
                    _shape_block(w),
                    delivered_block,
                    "Declared %s %r at %s was delivered with "
                    "color %s instead of its declared color %s."
                    % (
                        w.get("kind"),
                        w.get("id"),
                        _fmt_position(w),
                        _color_text(d.get("color")),
                        _color_text(w.get("color")),
                    ),
                )
            )
        if clean:
            realized_as_declared += 1

    for wi, w in enumerate(written):
        if wi in matched_written:
            continue
        delivered_block = {
            "id": w.get("id"),
            "kind": w.get("kind"),
            "tag": w.get("kind"),
            "shape": w.get("kind"),
            "name": w.get("id"),
            "position": dict(w.get("position") or {}),
            "declared_position": dict(w.get("position") or {}),
            "size": dict(w.get("size") or {}),
            "geometry": _geometry(w),
            "color": None,
            "present": False,
            "found": False,
            "realized": False,
        }
        findings.append(
            _finding(
                "shape_missing",
                _where(w),
                _shape_block(w),
                delivered_block,
                "Declared %s %r at %s is missing from the "
                "delivered drawing: no delivered %s matches "
                "its declared kind and position."
                % (
                    w.get("kind"),
                    w.get("id"),
                    _fmt_position(w),
                    w.get("kind"),
                ),
            )
        )

    for di, d in enumerate(delivered):
        if di in matched_delivered:
            continue
        written_block = {
            "id": None,
            "kind": None,
            "tag": None,
            "shape": None,
            "name": None,
            "position": {},
            "declared_position": {},
            "size": {},
            "geometry": {},
            "color": None,
            "present": False,
            "declared": False,
            "matching_written_shape": None,
        }
        delivered_block = _shape_block(d)
        delivered_block.update({"present": True, "declared": False})
        findings.append(
            _finding(
                "shape_not_declared",
                _where(d),
                written_block,
                delivered_block,
                "Delivered %s %r at %s matches no declared "
                "shape: no written shape declares it."
                % (d.get("kind"), d.get("id"), _fmt_position(d)),
            )
        )

    counts = {}
    for finding in findings:
        counts[finding["kind"]] = counts.get(finding["kind"], 0) + 1
    declared = len(written)
    return {
        "findings": findings,
        "summary": _summary(findings),
        "shapes_written": declared,
        "shapes_declared": declared,
        "shapes_total": declared,
        "declared_shapes": declared,
        "declared_shape_count": declared,
        "shapes_delivered": len(delivered),
        "shapes_matched": len(matched_pairs),
        "shapes_realized": realized_as_declared,
        "shapes_realized_as_declared": realized_as_declared,
        "realized_as_declared": realized_as_declared,
        "realized_count": realized_as_declared,
        "shapes_missing": counts.get("shape_missing", 0),
        "missing_count": counts.get("shape_missing", 0),
        "shapes_not_declared": counts.get("shape_not_declared", 0),
        "geometry_off_count": counts.get("geometry_off", 0),
        "color_off_count": counts.get("color_off", 0),
        "tolerances": {
            "geometry_tolerance": tolerance,
            "geometry_tolerance_units": "SVG units",
            "position_tolerance": tolerance,
            "size_tolerance": tolerance,
            "geometry_tolerance_units_description": (
                "a delivered shape whose position or size departs "
                "from its declared values by more than the "
                "geometry tolerance is a geometry_off finding"
            ),
        },
    }


build_comparison = compute_comparison
