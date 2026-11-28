"""drawing_renderer.py - composing-side drawing renderer.

renderer(artifact, out_path, faults): renders the SVG to a PNG
at the viewBox size using cairosvg, with delivery faults applied
in the rendering only.  If no SVG renderer is available in the
environment, say so in those words instead of substituting
anything.  The record carries the fault-applied delivered SVG
itself, as the code and music renderers carry their
delivered structures. When cairosvg is unavailable,
renderer() raises before any record exists, so in that case
no delivered document is read.
"""

import copy
import os
import re
import xml.etree.ElementTree as ET

try:
    import cairosvg
except Exception:  # pragma: no cover
    cairosvg = None


def _local_name(tag):
    return str(tag).rsplit("}", 1)[-1]


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


def _number(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _apply_faults(svg, faults):
    root = ET.fromstring(svg)
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_element":
            element = _find_by_id(root, fault.get("id"))
            parent = _find_parent(root, element) if element is not None else None
            if parent is not None:
                parent.remove(element)
        elif kind == "shift_element":
            element = _find_by_id(root, fault.get("id"))
            if element is not None:
                dx = _number(fault.get("dx"))
                dy = _number(fault.get("dy"))
                for key in ("x", "x1", "x2", "cx"):
                    if element.get(key) is not None:
                        element.set(key, str(_number(element.get(key)) + dx))
                for key in ("y", "y1", "y2", "cy"):
                    if element.get(key) is not None:
                        element.set(key, str(_number(element.get(key)) + dy))
        elif kind == "recolor":
            element = _find_by_id(root, fault.get("id"))
            if element is not None:
                color = fault.get("color")
                if not isinstance(color, str) or not color:
                    raise ValueError("recolor requires a color")
                element.set("fill", color)
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return ET.tostring(root, encoding="unicode")


def _viewbox_size(root):
    viewbox = root.get("viewBox") or ""
    numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", viewbox)]
    if len(numbers) >= 4:
        return int(round(numbers[2])), int(round(numbers[3]))
    width = root.get("width")
    height = root.get("height")
    return int(round(_number(width, 100))), int(round(_number(height, 100)))


def renderer(artifact, out_path, faults):
    if cairosvg is None:
        raise RuntimeError("no SVG renderer is available in the environment")
    svg = _apply_faults(artifact.get("svg", ""), faults)
    root = ET.fromstring(svg)
    width, height = _viewbox_size(root)
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=out_path,
        output_width=width,
        output_height=height,
    )
    return {
        "path": str(out_path),
        "kind": "image",
        "width": width,
        "height": height,
        "delivered_svg": svg,
        "faults_applied": copy.deepcopy(list(faults or [])),
    }
