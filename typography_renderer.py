"""typography_renderer.py - composing-side typography renderer.

renderer(artifact, out_path, faults): draws the page to an
image with Pillow, faults applied in the drawing only.
"""
import copy, os
from PIL import Image, ImageDraw, ImageFont

def _font(size):
    size = int(max(6, round(float(size or 12))))
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def _apply_faults(elements, faults):
    elements = copy.deepcopy(elements)
    by_id = {e.get("id"): e for e in elements if isinstance(e, dict)}
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "shift_element":
            el = by_id.get(fault.get("id"))
            if el is not None:
                el["x"] = float(el.get("x", 0)) + float(fault.get("dx", 0)); el["y"] = float(el.get("y", 0)) + float(fault.get("dy", 0))
        elif kind in ("shrink_heading", "shrink_element"):
            el = by_id.get(fault.get("id"))
            if el is not None:
                el["font_size"] = float(fault.get("font_size"))
        elif kind == "move_into_margin":
            el = by_id.get(fault.get("id"))
            if el is not None:
                el["x"] = float(fault.get("x", 2)); el["y"] = float(fault.get("y", 2))
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return elements

def renderer(artifact, out_path, faults):
    page = dict(artifact.get("page") or {})
    width = int(page.get("width") or 800); height = int(page.get("height") or 1000)
    background = page.get("background") or "#ffffff"
    delivered = _apply_faults(artifact.get("elements", []), faults)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    realized = {}
    for el in delivered:
        x = float(el.get("x", 0)); y = float(el.get("y", 0)); w = float(el.get("width", 0)); h = float(el.get("height", 0))
        if el.get("kind") == "image":
            draw.rectangle([x, y, x + w, y + h], outline=(0, 0, 0), fill=(200, 200, 200))
            realized[el.get("id")] = {"x": x, "y": y, "width": w, "height": h}
            continue
        font = _font(el.get("font_size"))
        text = str(el.get("text", ""))
        align = el.get("alignment") or "left"
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = float(bbox[2] - bbox[0]); th = float(bbox[3] - bbox[1])
        except Exception:
            tw = min(w, len(text) * float(el.get("font_size") or 12) * 0.6); th = float(el.get("font_size") or 12)
        if align == "center":
            tx = x + max(0.0, (w - tw) / 2.0)
        elif align == "right":
            tx = x + max(0.0, w - tw)
        else:
            tx = x
        fill = (0, 0, 0)
        color = el.get("color")
        if isinstance(color, str) and color.startswith("#") and len(color) == 7:
            try:
                fill = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            except ValueError:
                pass
        draw.text((tx, y), text, fill=fill, font=font)
        realized[el.get("id")] = {"x": float(tx), "y": float(y), "width": float(tw), "height": float(th)}
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    # The session path passes an extensionless render path
    # (render_<n>); Pillow needs a format, so default to PNG
    # when no extension is present, as the animation renderer
    # does for its frames.
    save_path = str(out_path)
    if not os.path.splitext(save_path)[1]:
        save_path = save_path + ".png"
    image.save(save_path)
    out_path = save_path
    return {"path": str(out_path), "kind": "image", "width": width, "height": height, "page": page, "delivered_elements": delivered, "realized_boxes": realized, "faults_applied": copy.deepcopy(list(faults or []))}
