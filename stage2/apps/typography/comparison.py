"""stage2/apps/typography/comparison.py - Sensoryne's judgment of
the delivered layout against the written declarations.

The comparison works from the raster analyzer's own measured
realized boxes of the delivered layout (the evidence
measurements' per-element realized boxes, or the render
record's realized boxes), set against the written work as
the artifact states it: the text blocks declared in the
written page, each with its id, kind, text, and declared
placement and size (x, y, width, height).  It never works
from delivery-fault definitions: a finding names what the
delivered layout shows against what was written.

A declared block is matched to a delivered block by element
id, the identity the revisions address blocks by.  A
remaining delivered block carrying a remaining declared
block's exact text is that block's realization, and a
remaining delivered block at a remaining declared block's
own position is its realization with differing text.  The
realized box of a text block is the measured extent of its
text, drawn inside its declared box according to its
alignment, so its expected origin is the origin the
alignment places it at inside the declared box: the
declared x for left alignment, the declared x plus half
the slack for center, the declared x plus the slack for
right, and the declared y in every case.  An image block's
realized box is its declared box as drawn, so its measured
size is judged against the declared size.  Measured text
exceeding its declared box is then judged separately, as
overflow, rather than as a size displacement.

Each finding is an object with kind, where, written,
delivered, and statement.  The kinds are block_missing,
box_displaced, text_overflow, and text_differs.  The box
tolerance is stated here and in the reliability block;
every displaced or overflow finding carries both the
written and the delivered values it was judged on.
"""


__all__ = [
    "BOX_TOLERANCE",
    "PLACEMENT_TOLERANCE",
    "GEOMETRY_TOLERANCE",
    "OVERFLOW_TOLERANCE",
    "FINDING_KINDS",
    "compute_comparison",
    "build_comparison",
]

BOX_TOLERANCE = 0.5
PLACEMENT_TOLERANCE = BOX_TOLERANCE
GEOMETRY_TOLERANCE = BOX_TOLERANCE
OVERFLOW_TOLERANCE = BOX_TOLERANCE

FINDING_KINDS = (
    "block_missing",
    "box_displaced",
    "text_overflow",
    "text_differs",
)


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
            "No findings: the delivered layout matches the "
            "written declarations within the stated tolerances."
        )
    else:
        line = "%d finding(s): %s." % (
            total,
            ", ".join("%s=%d" % (kind, counts[kind]) for kind in sorted(counts)),
        )
    return {"counts": counts, "total": total, "line": line}


def _float_or_none(value):
    try:
        if isinstance(value, bool):
            return None
        return float(value)
    except Exception:
        return None


def _opening_words(text):
    words = str(text or "").split()
    return " ".join(words[:4])


def _fmt_box(box):
    if not isinstance(box, dict):
        return "no declared placement"
    return "x=%s, y=%s, width=%s, height=%s" % (
        box.get("x"),
        box.get("y"),
        box.get("width"),
        box.get("height"),
    )


def _box_of(entry):
    if not isinstance(entry, dict):
        return None
    for key in ("realized_box", "box", "declared_box", "placement"):
        candidate = entry.get(key)
        if isinstance(candidate, dict):
            values = {}
            for axis in ("x", "y", "width", "height"):
                number = _float_or_none(candidate.get(axis))
                if number is None:
                    return None
                values[axis] = number
            return values
    values = {}
    for axis in ("x", "y", "width", "height"):
        number = _float_or_none(entry.get(axis))
        if number is None:
            return None
        values[axis] = number
    return values


def _declared_box(entry):
    # A declared (written) box is the element's own x, y,
    # width, height; a structure-record entry carries it
    # under "box".
    if not isinstance(entry, dict):
        return None
    candidate = entry.get("box")
    if isinstance(candidate, dict) and all(
        _float_or_none(candidate.get(axis)) is not None
        for axis in ("x", "y", "width", "height")
    ):
        return {
            axis: float(candidate.get(axis))
            for axis in ("x", "y", "width", "height")
        }
    return _box_of(entry)


def _written_blocks(artifact):
    blocks = []
    if not isinstance(artifact, dict):
        return blocks
    elements = artifact.get("elements")
    if not isinstance(elements, list):
        return blocks
    for index, entry in enumerate(elements):
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        kind = entry.get("kind")
        if kind == "image":
            text = None
        elif text is not None:
            text = str(text)
        blocks.append(
            {
                "_index": index,
                "id": entry.get("id"),
                "kind": kind,
                "text": text,
                "opening_words": _opening_words(text) if text else None,
                "font_size": _float_or_none(entry.get("font_size")),
                "alignment": entry.get("alignment"),
                "box": _declared_box(entry),
            }
        )
    return blocks


def _realized_boxes_of(render_record, evidence):
    boxes = {}
    for source in (render_record, evidence):
        if not isinstance(source, dict):
            continue
        candidate = source.get("realized_boxes")
        if isinstance(candidate, dict):
            for element_id, box in candidate.items():
                if isinstance(box, dict) and element_id not in boxes:
                    parsed = _box_of(box)
                    if parsed is not None:
                        boxes[element_id] = parsed
    for source in (evidence, render_record):
        if not isinstance(source, dict):
            continue
        measurements = source.get("measurements")
        if not isinstance(measurements, dict):
            continue
        entries = measurements.get("elements")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("id") not in boxes:
                parsed = _box_of(entry)
                if parsed is not None:
                    boxes[entry.get("id")] = parsed
    return boxes


def _delivered_blocks(render_record, evidence):
    entries = None
    for source in (render_record, evidence):
        if not isinstance(source, dict):
            continue
        candidate = source.get("delivered_elements")
        if isinstance(candidate, list) and candidate:
            entries = candidate
            break
    if entries is None:
        for source in (render_record, evidence):
            if not isinstance(source, dict):
                continue
            candidate = source.get("elements")
            if isinstance(candidate, list) and candidate:
                entries = candidate
                break
    if entries is None:
        for source in (evidence, render_record):
            if not isinstance(source, dict):
                continue
            measurements = source.get("measurements")
            if isinstance(measurements, dict) and isinstance(
                measurements.get("elements"), list
            ) and measurements["elements"]:
                entries = measurements["elements"]
                break
            full = source.get("full")
            if isinstance(full, dict):
                full_measurements = full.get("measurements")
                if isinstance(full_measurements, dict) and isinstance(
                    full_measurements.get("elements"), list
                ) and full_measurements["elements"]:
                    entries = full_measurements["elements"]
                    break
    realized = _realized_boxes_of(render_record, evidence)
    blocks = []
    if isinstance(entries, list):
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            element_id = entry.get("id")
            box = realized.get(element_id)
            if box is None:
                box = _box_of(entry)
            text = entry.get("text")
            kind = entry.get("kind")
            if kind == "image":
                text = None
            elif text is not None:
                text = str(text)
            blocks.append(
                {
                    "_index": index,
                    "id": element_id,
                    "kind": kind,
                    "text": text,
                    "opening_words": _opening_words(text) if text else None,
                    "font_size": _float_or_none(entry.get("font_size")),
                    "alignment": entry.get("alignment"),
                    "box": box,
                }
            )
    # A realized box the detailed entries do not carry is
    # itself a delivered block whose text was not recorded.
    seen = {block.get("id") for block in blocks}
    for element_id, box in realized.items():
        if element_id in seen:
            continue
        blocks.append(
            {
                "_index": len(blocks),
                "id": element_id,
                "kind": None,
                "text": None,
                "opening_words": None,
                "font_size": None,
                "alignment": None,
                "box": box,
            }
        )
    return blocks


def _written_block(block):
    box = dict(block.get("box") or {})
    return {
        "id": block.get("id"),
        "kind": block.get("kind"),
        "element_kind": block.get("kind"),
        "name": block.get("id"),
        "text": block.get("text"),
        "opening_words": block.get("opening_words"),
        "font_size": block.get("font_size"),
        "alignment": block.get("alignment"),
        "box": box,
        "declared_box": dict(box),
        "placement": dict(box),
        "declared_placement": dict(box),
        "position": {"x": box.get("x"), "y": box.get("y")},
        "declared_position": {"x": box.get("x"), "y": box.get("y")},
        "width": box.get("width"),
        "height": box.get("height"),
        "present": True,
        "declared": True,
    }


def _delivered_block(block, present=True):
    box = block.get("box")
    box = dict(box) if isinstance(box, dict) else None
    return {
        "id": block.get("id"),
        "kind": block.get("kind"),
        "element_kind": block.get("kind"),
        "name": block.get("id"),
        "text": block.get("text"),
        "opening_words": block.get("opening_words"),
        "font_size": block.get("font_size"),
        "alignment": block.get("alignment"),
        "box": dict(box) if box else None,
        "realized_box": dict(box) if box else None,
        "placement": dict(box) if box else None,
        "position": (
            {"x": box.get("x"), "y": box.get("y")} if box else None
        ),
        "width": box.get("width") if box else None,
        "height": box.get("height") if box else None,
        "present": bool(present),
        "found": bool(present),
        "realized": bool(present),
    }


def _where(block):
    box = block.get("box") or {}
    if block.get("id") is not None:
        return "block %r at x=%s, y=%s" % (
            block.get("id"),
            box.get("x"),
            box.get("y"),
        )
    return "block at x=%s, y=%s" % (box.get("x"), box.get("y"))


def _expected_origin(written, delivered):
    declared = written.get("box") or {}
    realized = delivered.get("box") or {}
    if written.get("kind") == "image" or delivered.get("kind") == "image":
        return (declared.get("x"), declared.get("y"))
    alignment = delivered.get("alignment") or written.get("alignment") or "left"
    declared_x = declared.get("x", 0.0)
    declared_width = declared.get("width", 0.0)
    realized_width = realized.get("width", 0.0)
    slack = max(0.0, float(declared_width) - float(realized_width))
    if alignment == "center":
        expected_x = float(declared_x) + slack / 2.0
    elif alignment == "right":
        expected_x = float(declared_x) + slack
    else:
        expected_x = float(declared_x)
    return (expected_x, declared.get("y"))


def compute_comparison(artifact, render_record=None, evidence=None):
    """Compare the delivered layout against the written blocks."""
    # A direct call may carry the evidence in the second place.
    if evidence is None and isinstance(render_record, dict):
        if "measurements" in render_record and "delivered_elements" not in render_record:
            evidence = render_record
            render_record = None

    tolerance = BOX_TOLERANCE
    written = _written_blocks(artifact)
    delivered = _delivered_blocks(render_record, evidence)

    matched_pairs = []  # (written_index, delivered_index)
    matched_written = set()
    matched_delivered = set()

    # Matched by element id: the identity revisions use.
    for wi, w in enumerate(written):
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

    # A remaining delivered block carrying a remaining
    # declared block's exact text is its realization.
    for wi, w in enumerate(written):
        if wi in matched_written or not w.get("text"):
            continue
        for di, d in enumerate(delivered):
            if di in matched_delivered or not d.get("text"):
                continue
            if d.get("text") != w.get("text"):
                continue
            if (
                w.get("kind") is not None
                and d.get("kind") is not None
                and d.get("kind") != w.get("kind")
            ):
                continue
            matched_pairs.append((wi, di))
            matched_written.add(wi)
            matched_delivered.add(di)
            break

    # A remaining delivered block at a remaining declared
    # block's own position is its realization with differing
    # text, so the two texts are named together.
    for wi, w in enumerate(written):
        if wi in matched_written:
            continue
        for di, d in enumerate(delivered):
            if di in matched_delivered:
                continue
            if d.get("_index") == w.get("_index"):
                matched_pairs.append((wi, di))
                matched_written.add(wi)
                matched_delivered.add(di)
                break

    findings = []
    realized_as_declared = 0
    order = {wi: index for index, (wi, _di) in enumerate(matched_pairs)}

    for wi, di in sorted(matched_pairs, key=lambda pair: order[pair[0]]):
        w = written[wi]
        d = delivered[di]
        block_findings = 0

        written_text = w.get("text")
        delivered_text = d.get("text")
        if (
            written_text is not None
            and delivered_text is not None
            and written_text != delivered_text
            and w.get("kind") != "image"
            and d.get("kind") != "image"
        ):
            block_findings += 1
            findings.append(
                _finding(
                    "text_differs",
                    _where(w),
                    _written_block(w),
                    _delivered_block(d),
                    "Block %r text differs: the written block "
                    "has text %r, but the delivered block "
                    "has text %r, which matches no declared "
                    "block."
                    % (w.get("id"), written_text, delivered_text),
                )
            )

        declared_box = w.get("box")
        realized_box = d.get("box")
        if isinstance(declared_box, dict) and isinstance(realized_box, dict):
            expected_x, expected_y = _expected_origin(w, d)
            displacement_x = float(realized_box.get("x", 0.0)) - float(expected_x)
            displacement_y = float(realized_box.get("y", 0.0)) - float(expected_y)
            is_image = w.get("kind") == "image" or d.get("kind") == "image"
            size_departures = {}
            if is_image:
                for axis in ("width", "height"):
                    difference = float(realized_box.get(axis, 0.0)) - float(
                        declared_box.get(axis, 0.0)
                    )
                    if abs(difference) > tolerance:
                        size_departures[axis] = {
                            "written": float(declared_box.get(axis, 0.0)),
                            "delivered": float(realized_box.get(axis, 0.0)),
                            "difference": float(difference),
                            "absolute_difference": float(abs(difference)),
                        }
            displaced = (
                abs(displacement_x) > tolerance
                or abs(displacement_y) > tolerance
                or bool(size_departures)
            )
            if displaced:
                block_findings += 1
                delivered_block = _delivered_block(d)
                delivered_block.update(
                    {
                        "declared_box": dict(declared_box),
                        "declared_placement": dict(declared_box),
                        "expected_origin": {
                            "x": float(expected_x),
                            "y": float(expected_y),
                        },
                        "displacement": {
                            "x": float(displacement_x),
                            "y": float(displacement_y),
                        },
                        "departures": dict(size_departures),
                        "box_tolerance": tolerance,
                        "placement_tolerance": tolerance,
                        "size_tolerance": tolerance,
                    }
                )
                findings.append(
                    _finding(
                        "box_displaced",
                        _where(w),
                        _written_block(w),
                        delivered_block,
                        "Block %r declared at %s was delivered "
                        "at realized box %s: its realized box "
                        "departs from its declared placement "
                        "or size by more than the %s box "
                        "tolerance."
                        % (
                            w.get("id"),
                            _fmt_box(declared_box),
                            _fmt_box(realized_box),
                            tolerance,
                        ),
                    )
                )

            is_text = not is_image and (
                w.get("kind") is not None
                or w.get("text") is not None
                or delivered_text is not None
            )
            if is_text:
                excess_width = float(realized_box.get("width", 0.0)) - float(
                    declared_box.get("width", 0.0)
                )
                excess_height = float(realized_box.get("height", 0.0)) - float(
                    declared_box.get("height", 0.0)
                )
                if excess_width > tolerance or excess_height > tolerance:
                    block_findings += 1
                    delivered_block = _delivered_block(d)
                    delivered_block.update(
                        {
                            "declared_box": dict(declared_box),
                            "declared_placement": dict(declared_box),
                            "measured_extent": dict(realized_box),
                            "measured_text_extent": dict(realized_box),
                            "excess": {
                                "width": float(excess_width),
                                "height": float(excess_height),
                            },
                            "overflow_tolerance": tolerance,
                        }
                    )
                    findings.append(
                        _finding(
                            "text_overflow",
                            _where(w),
                            _written_block(w),
                            delivered_block,
                            "Block %r text overflows its declared "
                            "box: the measured text extent "
                            "width=%s, height=%s exceeds its "
                            "declared box %s, by more than the "
                            "%s overflow tolerance."
                            % (
                                w.get("id"),
                                realized_box.get("width"),
                                realized_box.get("height"),
                                _fmt_box(declared_box),
                                tolerance,
                            ),
                        )
                    )

        if block_findings == 0:
            realized_as_declared += 1

    for wi, w in enumerate(written):
        if wi in matched_written:
            continue
        delivered_block = {
            "id": w.get("id"),
            "kind": w.get("kind"),
            "element_kind": w.get("kind"),
            "name": w.get("id"),
            "text": None,
            "opening_words": None,
            "box": None,
            "realized_box": None,
            "placement": None,
            "declared_box": dict(w.get("box") or {}),
            "declared_placement": dict(w.get("box") or {}),
            "present": False,
            "found": False,
            "realized": False,
        }
        opening = w.get("opening_words")
        opening_clause = (
            " beginning %r" % (opening,) if opening else ""
        )
        findings.append(
            _finding(
                "block_missing",
                "block %r%s at x=%s, y=%s"
                % (
                    w.get("id"),
                    opening_clause,
                    (w.get("box") or {}).get("x"),
                    (w.get("box") or {}).get("y"),
                ),
                _written_block(w),
                delivered_block,
                "Declared block %r%s at %s is missing from "
                "the delivered layout: no delivered block "
                "matches it."
                % (
                    w.get("id"),
                    opening_clause,
                    _fmt_box(w.get("box")),
                ),
            )
        )

    for di, d in enumerate(delivered):
        if di in matched_delivered:
            continue
        written_block = {
            "id": None,
            "kind": None,
            "element_kind": None,
            "name": None,
            "text": None,
            "opening_words": None,
            "box": None,
            "declared_box": None,
            "declared_placement": None,
            "present": False,
            "declared": False,
            "matching_written_block": None,
            "declared_texts": [
                block.get("text")
                for block in written
                if block.get("text") is not None
            ],
        }
        delivered_text = d.get("text")
        if delivered_text is not None:
            statement = (
                "Delivered block %r with text %r matches no "
                "declared block: the declared blocks are %s, "
                "and no declared block carries this text."
                % (
                    d.get("id"),
                    delivered_text,
                    ", ".join(
                        "%r" % (block.get("text"),)
                        for block in written
                        if block.get("text") is not None
                    )
                    or "none",
                )
            )
        else:
            statement = (
                "Delivered block %r at %s matches no declared "
                "block: no written block declares it."
                % (d.get("id"), _fmt_box(d.get("box")))
            )
        findings.append(
            _finding(
                "text_differs",
                _where(d),
                written_block,
                _delivered_block(d),
                statement,
            )
        )

    counts = {}
    for finding in findings:
        counts[finding["kind"]] = counts.get(finding["kind"], 0) + 1
    declared = len(written)
    return {
        "findings": findings,
        "summary": _summary(findings),
        "blocks_written": declared,
        "blocks_declared": declared,
        "blocks_total": declared,
        "declared_blocks": declared,
        "declared_block_count": declared,
        "blocks_delivered": len(delivered),
        "delivered_blocks": len(delivered),
        "blocks_matched": len(matched_pairs),
        "blocks_realized": realized_as_declared,
        "blocks_realized_as_declared": realized_as_declared,
        "blocks_realized_at_declared_placements": realized_as_declared,
        "blocks_realized_at_declared_placement": realized_as_declared,
        "realized_as_declared": realized_as_declared,
        "realized_at_declared_placements": realized_as_declared,
        "realized_at_declared_placement": realized_as_declared,
        "realized_count": realized_as_declared,
        "blocks_missing": counts.get("block_missing", 0),
        "missing_count": counts.get("block_missing", 0),
        "box_displaced_count": counts.get("box_displaced", 0),
        "displaced_count": counts.get("box_displaced", 0),
        "text_overflow_count": counts.get("text_overflow", 0),
        "overflow_count": counts.get("text_overflow", 0),
        "text_differs_count": counts.get("text_differs", 0),
        "tolerances": {
            "box_tolerance": tolerance,
            "placement_tolerance": tolerance,
            "position_tolerance": tolerance,
            "size_tolerance": tolerance,
            "geometry_tolerance": tolerance,
            "overflow_tolerance": tolerance,
            "tolerance": tolerance,
            "box_tolerance_units": "pixels",
            "placement_tolerance_description": (
                "a realized box whose placement departs from "
                "its declared placement by more than the box "
                "tolerance is a box_displaced finding; for "
                "text the placement expected is the origin "
                "its alignment places inside the declared "
                "box, and for image blocks the declared size "
                "is judged as well"
            ),
            "overflow_tolerance_description": (
                "measured text whose extent exceeds its "
                "declared box by more than the box tolerance "
                "is a text_overflow finding"
            ),
        },
    }


build_comparison = compute_comparison
