"""stage2/apps/typography/instrument.py - raster analyzer for a drawn page.

Measures realized boxes, margins, alignment consistency,
contrast, type hierarchy ratios, and whitespace share from the
rendered image and the delivered element record the renderer
returns. Reports measurements only.
"""
from collections import Counter
import numpy as np
from PIL import Image

METHOD = "image opened with Pillow; background is the most common color; realized boxes are the delivered boxes as drawn, recorded by the renderer from the drawing calls; margins are distances from each realized box to the page edges; contrast is the sRGB relative luminance ratio between the element color and the background color; whitespace share is the background pixel share"

def _lum(rgb):
    vals = []
    for c in rgb[:3]:
        v = float(c) / 255.0
        vals.append(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
    return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]

def _parse_color(value, default=(0, 0, 0)):
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return tuple(int(max(0, min(255, x))) for x in value[:3])
    if isinstance(value, str) and value.startswith("#") and len(value) == 7:
        try:
            return tuple(int(value[i:i+2], 16) for i in (1, 3, 5))
        except ValueError:
            return default
    return default

def analyze_page(path, page=None, elements=None, realized_boxes=None):
    page = dict(page or {})
    elements = list(elements or [])
    realized_boxes = dict(realized_boxes or {})
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixels = np.asarray(rgb)
    page_width = int(page.get("width") or width)
    page_height = int(page.get("height") or height)
    colors, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
    background = [int(x) for x in colors[int(np.argmax(counts))]]
    bg_count = int(np.max(counts))
    total = int(width * height) if width and height else 0
    whitespace = bg_count / total if total else 0.0
    bg_lum = _lum(background)
    measured = []
    left_edges = Counter()
    kind_sizes = {}
    for el in elements:
        eid = el.get("id")
        box = realized_boxes.get(eid) or {"x": el.get("x"), "y": el.get("y"), "width": el.get("width"), "height": el.get("height")}
        bx = float(box.get("x", 0)); by = float(box.get("y", 0)); bw = float(box.get("width", 0)); bh = float(box.get("height", 0))
        left_edges[bx] += 1
        kind_sizes.setdefault(el.get("kind"), []).append(float(el.get("font_size") or 0))
        color = _parse_color(el.get("color"))
        fg_lum = _lum(color)
        hi, lo = max(fg_lum, bg_lum), min(fg_lum, bg_lum)
        contrast = (hi + 0.05) / (lo + 0.05) if (lo + 0.05) else 0.0
        measured.append({"id": eid, "kind": el.get("kind"), "realized_box": {"x": bx, "y": by, "width": bw, "height": bh}, "margins": {"left": bx, "top": by, "right": float(page_width - (bx + bw)), "bottom": float(page_height - (by + bh))}, "contrast_ratio": float(contrast), "font_size": float(el.get("font_size") or 0), "alignment": el.get("alignment")})
    def median(kind):
        vals = sorted(kind_sizes.get(kind) or [])
        return float(vals[len(vals)//2]) if vals else None
    heading = median("heading"); body = median("body"); caption = median("caption")
    hierarchy = {"heading_median": heading, "body_median": body, "caption_median": caption, "heading_body_ratio": (heading/body if heading and body else None), "body_caption_ratio": (body/caption if body and caption else None), "heading_caption_ratio": (heading/caption if heading and caption else None)}
    shared = int(max(left_edges.values())) if left_edges else 0
    summary = "Page %d by %d pixels, %d elements measured. Background share %.4f. Shared left edge count %d. Heading median %s, body median %s, caption median %s." % (width, height, len(measured), whitespace, shared, heading, body, caption)
    measurements = {"instrument": "raster analyzer", "method": METHOD, "width": int(width), "height": int(height), "dimensions": [int(width), int(height)], "page_width": page_width, "page_height": page_height, "background_color": background, "elements": measured, "alignment_consistency": {"left_edges": {str(k): int(v) for k, v in left_edges.items()}, "shared_left_edge_count": shared}, "contrast": [{"id": e["id"], "contrast_ratio": e["contrast_ratio"]} for e in measured if e["kind"] in ("heading", "body", "caption")], "type_hierarchy": hierarchy, "whitespace_share": float(whitespace)}
    full = {"instrument": "raster analyzer", "method": METHOD, "render_path": str(path), "measurements": measurements, "interpretation": None}
    return {"summary": summary, "measurements": measurements, "full": full, "render_path": str(path), "instrument": "raster analyzer", "interpretation": None}

def perceive_page(path, page=None, elements=None, realized_boxes=None):
    return analyze_page(path, page, elements, realized_boxes)
