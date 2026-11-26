"""stage2/apps/typography/invariants.py - declared invariants, reliability, coverage.

Box invariants are evaluated on realized boxes from the
render record when the renderer supplies them, and fall back
to declared boxes only when no realized record exists.
"""
from collections import Counter

def _c(i, d, held, value, violations=None):
    return {"id": i, "description": d, "held": bool(held), "value": value, "violations": list(violations or [])}

def _elements(a):
    return a.get("elements", []) if isinstance(a, dict) else []

def _page(a):
    return a.get("page", {}) if isinstance(a, dict) and isinstance(a.get("page"), dict) else {}

def _boxes(artifact, render_record):
    realized = {}
    declared = {}
    for el in _elements(artifact):
        if isinstance(el, dict) and el.get("id"):
            declared[el["id"]] = {"x": float(el.get("x", 0)), "y": float(el.get("y", 0)), "width": float(el.get("width", 0)), "height": float(el.get("height", 0))}
    if isinstance(render_record, dict):
        rb = render_record.get("realized_boxes")
        if isinstance(rb, dict):
            for eid, box in rb.items():
                if isinstance(box, dict):
                    try:
                        realized[eid] = {"x": float(box.get("x", 0)), "y": float(box.get("y", 0)), "width": float(box.get("width", 0)), "height": float(box.get("height", 0))}
                    except Exception:
                        pass
    boxes = dict(declared)
    boxes.update(realized)
    return boxes, bool(realized)

def _median(vals):
    vals = sorted(vals)
    return float(vals[len(vals)//2]) if vals else None

def evaluate_invariants(artifact, render_record, evidence=None):
    page = _page(artifact)
    if isinstance(render_record, dict) and isinstance(render_record.get("page"), dict) and render_record["page"]:
        page = render_record["page"]
    pw = float(page.get("width") or 0); ph = float(page.get("height") or 0)
    elements = _elements(artifact)
    boxes, realized = _boxes(artifact, render_record)
    source = "realized" if realized else "declared"
    inside_bad = []
    for eid, box in boxes.items():
        x = box["x"]; y = box["y"]; w = box["width"]; h = box["height"]
        if pw and ph and not (x >= 0 and y >= 0 and x + w <= pw and y + h <= ph):
            inside_bad.append({"id": eid, "box": [x, y, w, h], "source": source})
    checked = [_c("elements_inside_page", "Every element %s box lies inside the page bounds." % source, not inside_bad, {"page_width": pw, "page_height": ph, "source": source}, inside_bad)]
    heading = _median([float(e.get("font_size") or 0) for e in elements if isinstance(e, dict) and e.get("kind") == "heading"])
    body = _median([float(e.get("font_size") or 0) for e in elements if isinstance(e, dict) and e.get("kind") == "body"])
    caption = _median([float(e.get("font_size") or 0) for e in elements if isinstance(e, dict) and e.get("kind") == "caption"])
    if heading is not None and body is not None:
        checked.append(_c("heading_larger_than_body", "The heading font size is larger than the body font size.", heading > body, {"heading": heading, "body": body}, [] if heading > body else [{"heading": heading, "body": body}]))
    if body is not None and caption is not None:
        checked.append(_c("body_larger_than_caption", "The body font size is larger than the caption font size.", body > caption, {"body": body, "caption": caption}, [] if body > caption else [{"body": body, "caption": caption}]))
    lefts = [box["x"] for box in boxes.values()]
    shared = max(Counter(lefts).values()) if lefts else 0
    checked.append(_c("shared_alignment_edge", "At least two elements share a %s left edge when at least two elements are present." % source, len(lefts) < 2 or shared >= 2, shared, [] if len(lefts) < 2 or shared >= 2 else [{"left_edges": lefts}]))
    return {"checked": checked, "summary": {"total_checked": len(checked), "held": sum(1 for c in checked if c["held"]), "broken": sum(1 for c in checked if not c["held"])}}

def reliability_block():
    return {"instrument": "raster analyzer", "limits": ["Rasterization resolution is the rendered image dimensions in pixels.", "Realized text boxes are recorded from the drawing calls; font metrics depend on the available font.", "Contrast uses sRGB relative luminance against the most common image color."]}

def objective_coverage(artifact, render_record, evidence=None):
    out = []
    page = _page(artifact)
    if page.get("width") and page.get("height"):
        out.append({"item": "page size %s by %s" % (page.get("width"), page.get("height")), "present": True})
    elements = _elements(artifact)
    kinds = {e.get("kind") for e in elements if isinstance(e, dict)}
    for kind in ("heading", "body", "caption", "image"):
        if kind in kinds:
            out.append({"item": "element kind %s" % kind, "present": True})
    out.append({"item": "element count %d" % len(elements), "present": len(elements) > 0})
    return out
