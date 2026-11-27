"""stage2/apps/animation/invariants.py - declared invariants, reliability, coverage."""
def _c(i, d, held, value, violations=None):
    return {"id": i, "description": d, "held": bool(held), "value": value, "violations": list(violations or [])}
def _elements(a):
    return a.get("elements", []) if isinstance(a, dict) else []
def _tracks(el):
    return el.get("keyframes") or el.get("track") or []
def _comparison_of(evidence, render_record):
    for source in (evidence, render_record):
        if isinstance(source, dict):
            comp = source.get("comparison")
            if isinstance(comp, dict):
                return comp
            full = source.get("full")
            if isinstance(full, dict) and isinstance(full.get("comparison"), dict):
                return full["comparison"]
    return None
def _comparison(artifact, render_record, evidence):
    comp = _comparison_of(evidence, render_record)
    if isinstance(comp, dict):
        return comp
    try:
        from .comparison import compute_comparison
        return compute_comparison(artifact, render_record, evidence)
    except Exception:
        return None
def evaluate_invariants(artifact, render_record, evidence=None):
    scene = artifact if isinstance(artifact, dict) else {}
    elements = _elements(artifact)
    checked = []
    expected_frames = None
    if scene.get("duration") is not None and scene.get("frame_rate"):
        expected_frames = int(round(float(scene["duration"]) * float(scene["frame_rate"])))
    if isinstance(render_record, dict) and render_record.get("frame_count") is not None:
        measured = int(render_record["frame_count"])
        frame_held = expected_frames is None or measured == expected_frames
        checked.append(_c("frame_count_matches", "Rendered frame count matches duration times frame rate.", frame_held, measured, [] if frame_held else [{"expected": expected_frames, "measured": measured}]))
    bad = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        times = [float(k.get("time", 0)) for k in _tracks(el) if isinstance(k, dict)]
        if not times:
            bad.append({"id": el.get("id"), "reason": "no keyframes"})
        elif scene.get("duration") is not None and any(t < 0 or t > float(scene["duration"]) for t in times):
            bad.append({"id": el.get("id"), "reason": "keyframe time outside duration"})
    checked.append(_c("keyframes_inside_duration", "Every keyframe time lies inside the scene duration.", not bad, [el.get("id") for el in elements if isinstance(el, dict)], bad))
    order_bad = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        times = [float(k.get("time", 0)) for k in _tracks(el) if isinstance(k, dict)]
        if times != sorted(times):
            order_bad.append({"id": el.get("id"), "times": times})
    checked.append(_c("keyframe_time_order", "Keyframes are in time order.", not order_bad, [el.get("id") for el in elements if isinstance(el, dict)], order_bad))
    easings = {"linear", "ease-in", "ease-out", "ease-in-out"}
    bad_easing = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        for k in _tracks(el):
            if isinstance(k, dict) and k.get("easing") not in easings:
                bad_easing.append({"id": el.get("id"), "easing": k.get("easing")})
    checked.append(_c("easings_named", "Every keyframe segment easing is one of the named easings.", not bad_easing, sorted(easings), bad_easing))
    if scene.get("loop"):
        final_first = None
        if isinstance(evidence, dict) and isinstance(evidence.get("measurements"), dict):
            final_first = evidence["measurements"].get("final_first_difference")
        checked.append(_c("loop_final_matches_first", "For a work declared a loop, the final frame matches the first frame.", final_first == 0.0, final_first, [] if final_first == 0.0 else [{"final_first_difference": final_first}]))
    return {"checked": checked, "summary": {"total_checked": len(checked), "held": sum(1 for c in checked if c["held"]), "broken": sum(1 for c in checked if not c["held"])}}
def reliability_block():
    try:
        from .comparison import KEYFRAME_TIME_TOLERANCE_SECONDS
        tolerance = KEYFRAME_TIME_TOLERANCE_SECONDS
    except Exception:
        tolerance = 0.1
    return {"instrument": "frame analyzer", "limits": ["Rasterization resolution is the rendered frame dimensions in pixels.", "Positions are sampled at the scene start, quartiles, and end.", "Frame difference is mean absolute pixel difference between consecutive frames.", "Comparison keyframe time tolerance is one frame interval (1 / frame_rate seconds, %s seconds when no frame rate is known): a written keyframe is realized when the delivered frames hold its element within one frame interval of its written time, and the delivered duration (delivered frame count at the delivered frame rate) is judged against the written duration within the same frame interval." % tolerance]}
def objective_coverage(artifact, render_record, evidence=None):
    comp = _comparison(artifact, render_record, evidence)
    if isinstance(comp, dict) and comp.get("keyframes_written") is not None and comp.get("keyframes_realized") is not None:
        written = comp.get("keyframes_written")
        realized = comp.get("keyframes_realized")
    else:
        written = sum(len(_tracks(el)) for el in _elements(artifact) if isinstance(el, dict))
        realized = 0
    frame_count_written = comp.get("frame_count_written") if isinstance(comp, dict) else None
    frame_count_delivered = comp.get("frame_count_delivered") if isinstance(comp, dict) else None
    if frame_count_written is None or frame_count_delivered is None:
        scene = artifact if isinstance(artifact, dict) else {}
        if scene.get("duration") is not None and scene.get("frame_rate"):
            frame_count_written = int(round(float(scene["duration"]) * float(scene["frame_rate"])))
        if isinstance(render_record, dict) and render_record.get("frame_count") is not None:
            frame_count_delivered = int(render_record["frame_count"])
    return [{"item": "written keyframes realized, %s of %s" % (realized, written), "covered": bool(written) and realized == written}, {"item": "delivered frames matching the written count, %s of %s" % (frame_count_delivered, frame_count_written), "covered": frame_count_written is not None and frame_count_delivered == frame_count_written}]
