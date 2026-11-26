"""stage2/apps/sound_design/invariants.py - declared invariants, reliability, coverage."""
def _c(i, d, held, value, violations=None):
    return {"id": i, "description": d, "held": bool(held), "value": value, "violations": list(violations or [])}

def _events(a):
    return a.get("events", []) if isinstance(a, dict) else []

def _delivered(render_record):
    if isinstance(render_record, dict):
        return render_record.get("delivered_events") or render_record.get("events") or []
    return []

def _measured(evidence):
    if isinstance(evidence, dict) and isinstance(evidence.get("measurements"), dict):
        events = evidence["measurements"].get("events")
        return events if isinstance(events, list) else []
    return []

def evaluate_invariants(artifact, render_record, evidence=None):
    events = _events(artifact)
    delivered = _delivered(render_record)
    measured = _measured(evidence)
    duration = float(artifact.get("duration") or 0) if isinstance(artifact, dict) else 0.0
    delivered_ids = [e.get("id") for e in delivered if isinstance(e, dict)]
    artifact_ids = [e.get("id") for e in events if isinstance(e, dict)]
    missing = [x for x in artifact_ids if x not in delivered_ids]
    checked = []
    checked.append(_c("delivered_event_count_matches", "Delivered event count matches the artifact event count.", len(delivered) == len(events), {"artifact": len(events), "delivered": len(delivered)}, [] if len(delivered) == len(events) else [{"artifact": len(events), "delivered": len(delivered)}]))
    checked.append(_c("event_ids_present", "Every artifact event id is present among the delivered events.", not missing, delivered_ids, missing))
    outside = []
    for e in delivered:
        if not isinstance(e, dict):
            continue
        try:
            start = float(e.get("start", 0)); end = start + float(e.get("duration", 0))
        except Exception:
            outside.append({"id": e.get("id"), "reason": "non-numeric start or duration"})
            continue
        if start < 0 or end > duration:
            outside.append({"id": e.get("id"), "start": start, "end": end, "duration": duration})
    checked.append(_c("events_within_duration", "Every delivered event interval lies inside the stated duration.", not outside, duration, outside))
    onset_bad = []
    by_id = {e.get("id"): e for e in measured if isinstance(e, dict)}
    for e in delivered:
        if not isinstance(e, dict) or e.get("kind") == "silence":
            continue
        m = by_id.get(e.get("id"))
        if m is None or m.get("onset_time") is None:
            onset_bad.append({"id": e.get("id"), "reason": "no realized onset measured"})
        elif abs(float(m.get("onset_time")) - float(e.get("start", 0))) > 0.1:
            onset_bad.append({"id": e.get("id"), "declared_start": float(e.get("start", 0)), "realized_onset": m.get("onset_time")})
    checked.append(_c("realized_onsets_present", "Each non-silence delivered event has a realized onset within 0.1 seconds of its declared start.", not onset_bad, [e.get("id") for e in delivered if isinstance(e, dict)], onset_bad))
    duration_bad = []
    for e in delivered:
        if not isinstance(e, dict) or e.get("kind") == "silence":
            continue
        m = by_id.get(e.get("id"))
        if m is None or not float(m.get("realized_duration") or 0) > 0:
            duration_bad.append({"id": e.get("id"), "realized_duration": m.get("realized_duration") if m else None})
    checked.append(_c("realized_durations_positive", "Each non-silence delivered event has a positive realized duration.", not duration_bad, [e.get("id") for e in delivered if isinstance(e, dict)], duration_bad))
    silence_share = None
    if isinstance(evidence, dict) and isinstance(evidence.get("measurements"), dict):
        silence_share = evidence["measurements"].get("silence_share")
    checked.append(_c("silence_share_bounded", "Silence share is a measured fraction between 0 and 1.", silence_share is not None and 0.0 <= float(silence_share) <= 1.0, silence_share, [] if silence_share is not None and 0.0 <= float(silence_share) <= 1.0 else [{"silence_share": silence_share}]))
    return {"checked": checked, "summary": {"total_checked": len(checked), "held": sum(1 for c in checked if c["held"]), "broken": sum(1 for c in checked if not c["held"])}}

def reliability_block():
    return {"instrument": "non-pitch audio analyzer", "limits": ["Sample rate is that of the handed-in render; whole-file analysis uses a 4096-sample block with a 1024-sample hop.", "Realized onset uses a threshold of 5 percent of the event interval envelope maximum.", "Centroid and bandwidth are magnitude-weighted spectral statistics over the detected event span.", "Event windows come from the delivered event intervals in the render record."]}

def objective_coverage(artifact, render_record, evidence=None):
    out = []
    objective = None
    if isinstance(artifact, dict) and isinstance(artifact.get("objective"), dict):
        objective = artifact["objective"]
    elif isinstance(render_record, dict) and isinstance(render_record.get("objective"), dict):
        objective = render_record["objective"]
    if isinstance(objective, dict):
        if objective.get("duration") is not None:
            out.append({"item": "duration %s" % objective.get("duration"), "present": isinstance(artifact, dict) and artifact.get("duration") == objective.get("duration")})
        if objective.get("event_count") is not None:
            out.append({"item": "event count %s" % objective.get("event_count"), "present": len(_events(artifact)) == objective.get("event_count")})
        for kind in objective.get("kinds") or []:
            out.append({"item": "event kind %s" % kind, "present": any(e.get("kind") == kind for e in _events(artifact) if isinstance(e, dict))})
        for eid in objective.get("event_ids") or []:
            out.append({"item": "event id %s" % eid, "present": any(e.get("id") == eid for e in _events(artifact) if isinstance(e, dict))})
    return out
