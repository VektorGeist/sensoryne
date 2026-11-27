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
    # The judgment is about the written work as well as the delivered
    # copy: an artifact event that ends past the stated duration is the
    # judgment not holding, even when a delivery fault omits that event
    # from the render. Each id is checked once, so violations do not
    # duplicate an id that appears in both copies.
    seen_ids = set()
    checked_ids = []
    outside = []
    for e in [x for x in delivered if isinstance(x, dict)] + [x for x in events if isinstance(x, dict)]:
        eid = e.get("id")
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        checked_ids.append(eid)
        try:
            start = float(e.get("start", 0)); end = start + float(e.get("duration", 0))
        except Exception:
            outside.append({"id": eid, "reason": "non-numeric start or duration"})
            continue
        if start < 0 or end > duration:
            outside.append({"id": eid, "start": start, "end": end, "duration": duration})
    subject_ids = checked_ids
    checked.append(_c("events_within_duration", "Every delivered and written event interval lies inside the stated duration.", not outside, {"duration": duration, "checked_ids": sorted(x for x in checked_ids if x is not None)}, outside))
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
    silence_held = silence_share is not None and isinstance(silence_share, (int, float)) and not isinstance(silence_share, bool) and 0.0 <= float(silence_share) <= 1.0
    checked.append(_c("silence_share_bounded", "Silence share is a measured fraction between 0 and 1.", silence_held, silence_share, [] if silence_held else [{"silence_share": silence_share}]))
    return {"checked": checked, "summary": {"total_checked": len(checked), "held": sum(1 for c in checked if c["held"]), "broken": sum(1 for c in checked if not c["held"])}}

def reliability_block():
    return {"instrument": "non-pitch audio analyzer", "limits": ["Sample rate is that of the handed-in render; whole-file analysis uses a 4096-sample block with a 1024-sample hop.", "Realized onset uses a threshold of 5 percent of the event interval envelope maximum.", "Centroid and bandwidth are magnitude-weighted spectral statistics over the detected event span.", "Event windows come from the delivered event intervals in the render record.", "Comparison onset tolerance is 0.1 seconds: a realized onset within 0.1 seconds of a written onset realizes that written event at its written onset."]}

def objective_coverage(artifact, render_record, evidence=None):
    comp = _comparison(artifact, render_record, evidence)
    if isinstance(comp, dict) and comp.get("events_written") is not None and comp.get("events_realized_at_written_onsets") is not None:
        realized = comp.get("events_realized_at_written_onsets")
        written = comp.get("events_written")
    else:
        written = len([e for e in _events(artifact) if isinstance(e, dict) and e.get("kind") != "silence"])
        realized = 0
    return [{"item": "events realized at their written onsets, %s of %s" % (realized, written), "covered": bool(written) and realized == written}]
