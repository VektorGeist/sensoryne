"""animation_renderer.py - composing-side animation renderer.

renderer(artifact, out_path, faults): produces the frames as
images with Pillow, faults applied in the rendering only.
Renderer output contract: out_path is one file path. A
renderer that produces several files derives their paths from
out_path by inserting a frame index before the file extension:
<stem>_frame_NNN<ext>. Frame times span the declared duration
inclusive, from 0 to duration, so the final frame is the final
keyframe state.
"""
import copy, os
from PIL import Image, ImageDraw

def _ease(name, t):
    t = max(0.0, min(1.0, float(t)))
    if name == "ease-in":
        return t * t
    if name == "ease-out":
        return 1.0 - (1.0 - t) ** 2
    if name == "ease-in-out":
        return 2.0 * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0
    return t

def _lerp(a, b, t):
    return float(a) + (float(b) - float(a)) * float(t)

def _track(el):
    return sorted(el.get("keyframes") or el.get("track") or [], key=lambda k: float(k.get("time", 0)))

def _frame_paths(out_path, frame_count):
    base = str(out_path)
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".png"
    return ["%s_frame_%03d%s" % (stem, i, ext) for i in range(frame_count)]

def _state_at(track, time):
    if not track:
        return {"x": 0, "y": 0, "width": 0, "height": 0, "opacity": 1.0}
    if time <= float(track[0].get("time", 0)):
        return dict(track[0])
    if time >= float(track[-1].get("time", 0)):
        return dict(track[-1])
    for a, b in zip(track, track[1:]):
        ta = float(a.get("time", 0)); tb = float(b.get("time", 0))
        if ta <= time <= tb:
            span = tb - ta
            local = 0.0 if span <= 0 else (time - ta) / span
            eased = _ease(a.get("easing") or "linear", local)
            return {"x": _lerp(a.get("x", 0), b.get("x", 0), eased), "y": _lerp(a.get("y", 0), b.get("y", 0), eased), "width": _lerp(a.get("width", 0), b.get("width", 0), eased), "height": _lerp(a.get("height", 0), b.get("height", 0), eased), "opacity": _lerp(a.get("opacity", 1), b.get("opacity", 1), eased)}
    return dict(track[-1])

def _apply_faults(elements, faults):
    elements = copy.deepcopy(elements)
    by_id = {e.get("id"): e for e in elements if isinstance(e, dict)}
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_keyframe":
            el = by_id.get(fault.get("id"))
            if el is not None:
                key = "keyframes" if isinstance(el.get("keyframes"), list) else "track"
                el[key] = [k for k in (el.get(key) or []) if not (isinstance(k, dict) and float(k.get("time", -1)) == float(fault.get("time")))]
        elif kind == "shift_keyframe_time":
            el = by_id.get(fault.get("id"))
            if el is not None:
                key = "keyframes" if isinstance(el.get("keyframes"), list) else "track"
                for k in el.get(key) or []:
                    if isinstance(k, dict) and float(k.get("time", -1)) == float(fault.get("time")):
                        k["time"] = float(fault.get("new_time"))
                el[key] = sorted(el.get(key) or [], key=lambda k: float(k.get("time", 0)))
        elif kind == "swap_easings":
            first = by_id.get(fault.get("first")); second = by_id.get(fault.get("second"))
            if first is not None and second is not None:
                def easings(el):
                    key = "keyframes" if isinstance(el.get("keyframes"), list) else "track"
                    return key, [k.get("easing") for k in (el.get(key) or []) if isinstance(k, dict)]
                k1, e1 = easings(first); k2, e2 = easings(second)
                for k, e in zip([k for k in first.get(k1) or [] if isinstance(k, dict)], e2):
                    k["easing"] = e
                for k, e in zip([k for k in second.get(k2) or [] if isinstance(k, dict)], e1):
                    k["easing"] = e
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return elements

def renderer(artifact, out_path, faults):
    duration = float(artifact.get("duration") or 0)
    frame_rate = float(artifact.get("frame_rate") or 0)
    canvas = dict(artifact.get("canvas") or {})
    width = int(canvas.get("width") or 320); height = int(canvas.get("height") or 240)
    frame_count = int(round(duration * frame_rate)) if duration > 0 and frame_rate > 0 else 0
    delivered = _apply_faults(artifact.get("elements", []), faults)
    frame_paths = _frame_paths(out_path, frame_count)
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    realized_tracks = {el.get("id"): [] for el in delivered if isinstance(el, dict)}
    for index, path in enumerate(frame_paths):
        if frame_count <= 1:
            time = 0.0
        else:
            time = duration * index / float(frame_count - 1)
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        for el in delivered:
            if not isinstance(el, dict):
                continue
            state = _state_at(_track(el), time)
            x = float(state.get("x", 0)); y = float(state.get("y", 0)); w = float(state.get("width", 0)); h = float(state.get("height", 0)); opacity = float(state.get("opacity", 1))
            shade = int(255 * (1.0 - opacity))
            draw.rectangle([x, y, x + w, y + h], fill=(shade, shade, shade), outline=(0, 0, 0))
            realized_tracks.setdefault(el.get("id"), []).append({"time": float(time), "x": x, "y": y, "width": w, "height": h, "opacity": opacity})
        image.save(path)
    return {"path": str(out_path), "kind": "frames", "frame_paths": frame_paths, "frame_count": len(frame_paths), "duration": duration, "frame_rate": frame_rate, "scene": {"duration": duration, "frame_rate": frame_rate, "canvas": canvas, "loop": bool(artifact.get("loop"))}, "delivered_elements": delivered, "realized_tracks": realized_tracks, "loop": bool(artifact.get("loop")), "faults_applied": copy.deepcopy(list(faults or []))}
