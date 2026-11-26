"""stage2/apps/animation/invariants.py - declared invariants, reliability, coverage."""
def _c(i, d, held, value, violations=None):
    return {"id": i, "description": d, "held": bool(held), "value": value, "violations": list(violations or [])}
def _elements(a):
    return a.get("elements", []) if isinstance(a, dict) else []
def _tracks(el):
    return el.get("keyframes") or el.get("track") or []
def evaluate_invariants(artifact, render_record, evidence=None):
    scene = artifact if isinstance(artifact, dict) else {}
    elements = _elements(artifact)
    checked = []
    expected_frames = None
    if scene.get("duration") is not None and scene.get("frame_rate"):
        expected_frames = int(round(float(scene["duration"]) * float(scene["frame_rate"])))
    if isinstance(render_record, dict) and render_record.get("frame_count") is not None:
        checked.append(_c("frame_count_matches", "Rendered frame count matches duration times frame rate.", expected_frames is None or int(render_record["frame_count"]) == expected_frames, render_record.get("frame_count"), [] if expected_frames is None or int(render_record["frame_count"]) == expected_frames else [{"expected": expected_frames, "measured": render_record.get("frame_count")}]))
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
    return {"instrument": "frame analyzer", "limits": ["Rasterization resolution is the rendered frame dimensions in pixels.", "Positions are sampled at the scene start, quartiles, and end.", "Frame difference is mean absolute pixel difference between consecutive frames."]}
def objective_coverage(artifact, render_record, evidence=None):
    out = []
    scene = artifact if isinstance(artifact, dict) else {}
    if scene.get("duration") is not None:
        out.append({"item": "duration %s" % scene.get("duration"), "present": True})
    if scene.get("frame_rate"):
        out.append({"item": "frame rate %s" % scene.get("frame_rate"), "present": True})
    if scene.get("canvas"):
        out.append({"item": "canvas %s" % scene.get("canvas"), "present": True})
    for el in _elements(artifact):
        if isinstance(el, dict) and el.get("id"):
            out.append({"item": "element %s" % el["id"], "present": bool(_tracks(el))})
    return out
