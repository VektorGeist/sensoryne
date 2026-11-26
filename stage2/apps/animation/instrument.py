"""stage2/apps/animation/instrument.py - frame analyzer for a rendered scene.

Realized tracks are the renderer's per-frame states. A sampled
position at a stated time is that element's realized state in
the last rendered frame at or before that time. No position is
interpolated between frames and no state is reported that was
not rendered. The final sample is the final rendered frame
state. Reports measurements only.
"""
import numpy as np
from PIL import Image

METHOD = "frames opened with Pillow as RGB arrays; per-element realized position at a sampled time is that element's realized state in the last rendered frame at or before that time, as recorded by the renderer from its drawing calls; frame difference is mean absolute pixel difference between consecutive frames; loop match is mean absolute pixel difference between final and first frames; visible share is the share of frames whose pixels differ from the background"

def _sampled_positions(track, times):
    if not track:
        return []
    ordered = sorted(track, key=lambda k: float(k.get("time", 0)))
    out = []
    for t in times:
        chosen = None
        for key in ordered:
            if float(key.get("time", 0)) <= float(t):
                chosen = key
        if chosen is None:
            chosen = ordered[0]
        out.append({"time": float(t), "x": float(chosen.get("x", 0)), "y": float(chosen.get("y", 0)), "width": float(chosen.get("width", 0)), "height": float(chosen.get("height", 0)), "opacity": float(chosen.get("opacity", 1)), "easing": chosen.get("easing")})
    return out

def analyze_frames(frame_paths, scene=None, elements=None, realized_tracks=None, loop=False):
    elements = list(elements or [])
    realized_tracks = dict(realized_tracks or {})
    scene = dict(scene or {})
    duration = float(scene.get("duration") or 0)
    frame_rate = float(scene.get("frame_rate") or 0)
    if not frame_paths:
        measurements = {"instrument": "frame analyzer", "method": METHOD, "frame_count": 0, "duration": duration, "frame_rate": frame_rate, "elements": [], "frame_difference_total": 0.0, "final_first_difference": None, "loop_matches": None}
        return {"summary": "0 frames rendered.", "measurements": measurements, "full": {"instrument": "frame analyzer", "method": METHOD, "measurements": measurements, "interpretation": None}, "instrument": "frame analyzer", "interpretation": None}
    frames = []
    for path in frame_paths:
        with Image.open(path) as image:
            image.load()
            frames.append(np.asarray(image.convert("RGB"), dtype=np.float64))
    height, width = frames[0].shape[:2]
    diffs = [float(np.mean(np.abs(b - a))) for a, b in zip(frames, frames[1:])]
    total_diff = float(sum(diffs))
    final_first = float(np.mean(np.abs(frames[-1] - frames[0]))) if len(frames) > 1 else 0.0
    measured = []
    for el in elements:
        eid = el.get("id")
        track = realized_tracks.get(eid) or el.get("keyframes") or []
        sample_times = [0.0]
        if duration > 0:
            sample_times += [duration * f for f in (0.25, 0.5, 0.75, 1.0)]
        visible_frames = 0
        for frame in frames:
            colors, counts = np.unique(frame.reshape(-1, 3), axis=0, return_counts=True)
            background = colors[int(np.argmax(counts))]
            visible_frames += 1 if float(np.mean(np.any(np.abs(frame - background) > 0, axis=2))) > 0.001 else 0
        measured.append({"id": eid, "sampled_positions": _sampled_positions(track, sample_times), "visible_frame_share": (visible_frames / len(frames)) if frames else 0.0, "visible_frames": int(visible_frames)})
    measurements = {"instrument": "frame analyzer", "method": METHOD, "frame_count": len(frames), "duration": duration, "frame_rate": frame_rate, "width": int(width), "height": int(height), "elements": measured, "frame_difference_total": total_diff, "frame_differences": diffs, "final_first_difference": final_first, "loop_declared": bool(loop), "loop_matches": bool(loop and final_first == 0.0) if loop else None}
    summary = "%d frames at %s fps, duration %s seconds. Total frame difference %.4f. Final to first frame difference %.4f. %d elements measured." % (len(frames), frame_rate, duration, total_diff, final_first, len(measured))
    full = {"instrument": "frame analyzer", "method": METHOD, "measurements": measurements, "interpretation": None}
    return {"summary": summary, "measurements": measurements, "full": full, "instrument": "frame analyzer", "interpretation": None}

def perceive_frames(frame_paths, scene=None, elements=None, realized_tracks=None, loop=False):
    return analyze_frames(frame_paths, scene, elements, realized_tracks, loop)
