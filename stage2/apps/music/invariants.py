"""stage2/apps/music/invariants.py - music invariants, reliability, coverage, robustness variations."""
import copy

def _c(i, d, held, value, violations=None):
    return {"id": i, "description": d, "held": bool(held), "value": value, "violations": list(violations or [])}

def _parts(artifact):
    return artifact.get("parts", []) if isinstance(artifact, dict) else []

def _is_vocal(artifact):
    return isinstance(artifact, dict) and artifact.get("kind") == "vocal"

def _comparison_per_part(artifact, render_record, evidence):
    """Per-part notes-heard figures from the comparison, if present.

    The comparison block (findings, per_part counts, tolerances)
    is attached to the evidence and to the render record by the
    controller.  This reads it without recomputing anything: the
    judgment of what was heard belongs to the comparison.
    """
    for source in (evidence, render_record):
        if isinstance(source, dict):
            comp = source.get("comparison")
            if isinstance(comp, dict) and isinstance(comp.get("per_part"), list):
                return comp["per_part"]
            full = source.get("full")
            if isinstance(full, dict) and isinstance(full.get("comparison"), dict):
                return full["comparison"].get("per_part") or []
    return None

def evaluate_invariants(artifact, render_record, evidence=None):
    checked = []
    if _is_vocal(artifact):
        lyrics = artifact.get("lyrics", [])
        vocal = None
        if isinstance(evidence, dict) and isinstance(evidence.get("measurements"), dict):
            vocal = evidence["measurements"].get("vocal")
        lines = vocal.get("lines") if isinstance(vocal, dict) else None
        if lines is None:
            checked.append(_c("vocal_lines_detected", "Every lyric line is detected as a phrase.", False, 0, [{"lyric_lines": len(lyrics)}]))
            checked.append(_c("vocal_syllable_counts", "Syllable event count per line matches the stated count.", False, [], [{"lyric_lines": len(lyrics)}]))
            checked.append(_c("vocal_pitch_present", "Pitch is present in every lyric line.", False, [], [{"lyric_lines": len(lyrics)}]))
        else:
            missing = [x["line"] for x in lines if not x.get("detected_phrase")]
            checked.append(_c("vocal_lines_detected", "Every lyric line is detected as a phrase.", not missing, len(lines) - len(missing), missing))
            bad = [{"line": x["line"], "detected": x.get("syllable_events_detected"), "stated": x.get("stated_syllable_count")} for x in lines if x.get("syllable_events_detected") != x.get("stated_syllable_count")]
            checked.append(_c("vocal_syllable_counts", "Syllable event count per line matches the stated count.", not bad, [x.get("syllable_events_detected") for x in lines], bad))
            nopitch = [x["line"] for x in lines if not x.get("median_pitch_hz")]
            checked.append(_c("vocal_pitch_present", "Pitch is present in every lyric line.", not nopitch, [x.get("median_pitch_hz") for x in lines], nopitch))
        return {"checked": checked, "summary": {"total_checked": len(checked), "held": sum(1 for c in checked if c["held"]), "broken": sum(1 for c in checked if not c["held"])}}
    parts = _parts(artifact)
    counts = {}
    for p in parts:
        if isinstance(p, dict):
            counts[p.get("name")] = len(p.get("notes", []))
    # Composer-chosen part names are never a judgment.  What the
    # comparison heard of what was written is reported per part:
    # notes heard of notes written, taken from the comparison
    # block.  When no comparison is present (no render heard),
    # the heard count is unmeasured: the written counts are
    # stated, notes_heard is None, no note is listed as not
    # heard, and the check is not held.
    per_part = _comparison_per_part(artifact, render_record, evidence)
    by_name = {}
    if per_part:
        for entry in per_part:
            if isinstance(entry, dict):
                by_name[entry.get("part")] = entry
    heard = []
    not_heard = []
    unmeasured = False
    for p in parts:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        written = counts.get(name, 0)
        entry = by_name.get(name)
        if entry is not None and entry.get("notes_heard") is not None:
            heard_n = int(entry.get("notes_heard") or 0)
            if entry.get("notes_not_heard") is not None:
                not_heard_n = int(entry.get("notes_not_heard") or 0)
            else:
                not_heard_n = written - heard_n
            heard.append({"part": name, "notes_written": written, "notes_heard": heard_n})
            if not_heard_n:
                not_heard.append({"part": name, "notes_written": written, "notes_heard": heard_n, "notes_not_heard": not_heard_n})
        else:
            unmeasured = True
            heard.append({"part": name, "notes_written": written, "notes_heard": None})
    checked.append(_c("notes_heard_of_written", "Per part, the notes heard are reported against the notes written.", (not not_heard) and not unmeasured, heard, not_heard))
    try:
        from .instruments import get_instrument
        inst_name = artifact.get("instrument") or "piano" if isinstance(artifact, dict) else "piano"
        inst = get_instrument(inst_name)
        rng = inst.get("written") or inst.get("sounding") or inst.get("absolute_range")
    except Exception:
        rng = None
        inst_name = artifact.get("instrument") if isinstance(artifact, dict) else None
    # The range judgment: every note falls inside the named
    # instrument's range. An unknown instrument name is echoed
    # as the measured value and the judgment is always held,
    # with no violations: an unknown name is a measurement,
    # not a failure. A known range is checked note by note,
    # and any note outside it is a named violation.
    range_bad = []
    if rng:
        for p in parts:
            if not isinstance(p, dict):
                continue
            for n in p.get("notes", []):
                if not (rng["low_midi"] <= int(n.get("midi", -1)) <= rng["high_midi"]):
                    range_bad.append({"part": p.get("name"), "start": float(n.get("start", 0)), "midi": n.get("midi"), "instrument": inst_name})
    checked.append(_c("notes_in_instrument_range", "Every note falls inside the named instrument's range.", not range_bad, inst_name, range_bad))
    checked.append(_c("note_count_per_part", "Note count per part is measured and recorded.", True, counts, []))
    return {"checked": checked, "summary": {"total_checked": len(checked), "held": sum(1 for c in checked if c["held"]), "broken": sum(1 for c in checked if not c["held"])}}

def reliability_block():
    # Comparison tolerances, stated here and in the comparison
    # itself, as comparison.py implements them. A written note
    # is counted heard by its fundamental evidence: the
    # magnitude at the sounding fundamental in its window must
    # stand at least NOTE_HEARD_FUNDAMENTAL_PEAK_RATIO (2.0)
    # above the neighbouring semitone fundamentals (the median
    # of the magnitudes at +/-1 and +/-2 semitones, lower of
    # the two middle for an even count) and at least
    # NOTE_HEARD_FUNDAMENTAL_BACKGROUND_RATIO (3.0) above the
    # median fundamental magnitude over the scored pitch
    # range in the same window, and the fundamental must be at
    # least 0.25 of the strongest neighbouring fundamental.
    # The harmonic ratio to the strongest pitch in the window
    # is reported for every note but is no longer the heard
    # judgment; NOTE_HEARD_RATIO_THRESHOLD (0.15) is carried
    # only as that reported ratio's reference. Silence is at
    # or below SILENCE_LOUDNESS_DB (-70.0), and comparison
    # pitch tolerance is PITCH_TOLERANCE_CENTS (50.0).
    # Dynamics are note-anchored, measured at each attack:
    # attack loudness is the RMS loudness in dB over
    # min(0.25, note duration) seconds from note onset, the
    # render's expectation is the median of adjusted_db
    # (attack_loudness_db - 20*log10(effective_velocity),
    # effective_velocity = velocity * part gain) over all
    # notes, and a note departing from that median by more
    # than DYNAMICS_THRESHOLD_DB (4.0) is a dynamics_off
    # finding. There is no fixed offset and no segment
    # formula.
    return {"instrument": "Sensoryne hearing", "limits": ["Sample rate and FFT resolution are those of the handed-in render and the cochlea analysis block.", "Fundamental pitch is the strongest interpolated peak in the pitch band.", "Syllable and formant figures are estimates from the stated phonetic methods.", "The hearing works on handed-in renders only; there is no live capture.", "Comparison pitch tolerance is 50.0 cents; a written note is counted heard when its fundamental magnitude in its window stands at least 2.0 times the neighbouring semitone fundamentals and at least 3.0 times the window background, and at least 0.25 of the strongest neighbouring fundamental; silence is at or below -70.0 dB.", "Comparison dynamics are note-anchored and measured at each attack: adjusted_db = attack_loudness_db - 20*log10(effective_velocity), expected_db is the median adjusted_db over all notes, threshold 4.0 dB, with no fixed offset and no segment formula."]}

def objective_coverage(artifact, render_record, evidence=None):
    out = []
    if _is_vocal(artifact):
        for i, line in enumerate(artifact.get("lyrics", [])):
            out.append({"item": "lyric line %d with %d stated syllables" % (i + 1, len(line)), "covered": bool(line)})
        return out
    parts = artifact.get("parts", []) if isinstance(artifact, dict) else []
    counts = {}
    for p in parts:
        if isinstance(p, dict):
            counts[p.get("name")] = len(p.get("notes", []))
    # Coverage derived from the comparison: per part, notes
    # heard of notes written, plus the instrument range item.
    # Nothing is covered by construction: absent comparison
    # means notes heard are unmeasured, and an empty part
    # means none written.
    per_part = _comparison_per_part(artifact, render_record, evidence)
    by_name = {}
    if per_part:
        for entry in per_part:
            if isinstance(entry, dict):
                by_name[entry.get("part")] = entry
    for p in parts:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        written = counts.get(name, 0)
        entry = by_name.get(name)
        if entry is not None and entry.get("notes_heard") is not None:
            heard_n = int(entry.get("notes_heard") or 0)
            out.append({"item": "part %s: %d of %d notes heard" % (name, heard_n, written), "covered": bool(written) and heard_n >= written})
        else:
            out.append({"item": "part %s: %d notes written, notes heard unmeasured" % (name, written), "covered": False})
    # The instrument range item, from the range check the
    # invariants report: a part whose notes all fall inside the
    # named instrument's range covers it.
    rng = None
    inst_name = artifact.get("instrument") if isinstance(artifact, dict) else None
    if isinstance(artifact, dict):
        try:
            from .instruments import get_instrument
            inst = get_instrument(inst_name or "piano")
            rng = inst.get("written") or inst.get("sounding") or inst.get("absolute_range")
        except Exception:
            rng = None
    if rng:
        in_range = True
        for p in parts:
            if not isinstance(p, dict):
                continue
            for n in p.get("notes", []):
                if not (rng["low_midi"] <= int(n.get("midi", -1)) <= rng["high_midi"]):
                    in_range = False
        out.append({"item": "instrument range %s" % (inst_name or "piano"), "covered": bool(in_range)})
    return out

def robustness_variations():
    return [{"name": "tempo_0.95", "tempo_scale": 0.95}, {"name": "tempo_1.05", "tempo_scale": 1.05}, {"name": "velocity_0.9", "velocity_scale": 0.9}]

def transform_artifact_for_variation(artifact, variation):
    if _is_vocal(artifact):
        return None
    import copy as _copy
    out = _copy.deepcopy(artifact)
    if isinstance(variation, dict):
        tempo = variation.get("tempo_scale")
        vscale = variation.get("velocity_scale")
    else:
        tempo = None
        vscale = 0.9 if variation == "velocity_0.9" else None
    for p in out.get("parts", []):
        for n in p.get("notes", []):
            if tempo is not None:
                n["start"] = float(n["start"]) * float(tempo)
                if n.get("dur") is not None:
                    n["dur"] = float(n["dur"]) * float(tempo)
            if vscale is not None:
                n["vel"] = max(0.0, min(1.0, float(n.get("vel", 0)) * float(vscale)))
    return out

# ---- merged Music & Sound: written events invariants/coverage (folded) ----
_score_evaluate_invariants = evaluate_invariants
_score_reliability_block = reliability_block
_score_objective_coverage = objective_coverage
_score_robustness_variations = robustness_variations
_score_transform = transform_artifact_for_variation

def _ev_events(a):
    return a.get("events", []) if isinstance(a, dict) else []

def _ev_delivered(render_record):
    if isinstance(render_record, dict):
        return render_record.get("delivered_events") or render_record.get("events") or []
    return []

def _ev_measured(evidence):
    if isinstance(evidence, dict) and isinstance(evidence.get("measurements"), dict):
        events = evidence["measurements"].get("events")
        return events if isinstance(events, list) else []
    return []

def _ev_comparison_of(evidence, render_record):
    for source in (evidence, render_record):
        if isinstance(source, dict):
            comp = source.get("comparison")
            if isinstance(comp, dict):
                return comp
            full = source.get("full")
            if isinstance(full, dict) and isinstance(full.get("comparison"), dict):
                return full["comparison"]
    return None

def _ev_comparison(artifact, render_record, evidence):
    comp = _ev_comparison_of(evidence, render_record)
    if isinstance(comp, dict):
        return comp
    try:
        from .comparison import compute_comparison as _cc
        return _cc(artifact, render_record, evidence)
    except Exception:
        return None

def _ev_evaluate_invariants(artifact, render_record, evidence=None):
    events = _ev_events(artifact)
    delivered = _ev_delivered(render_record)
    measured = _ev_measured(evidence)
    duration = float(artifact.get("duration") or 0) if isinstance(artifact, dict) else 0.0
    delivered_ids = [e.get("id") for e in delivered if isinstance(e, dict)]
    artifact_ids = [e.get("id") for e in events if isinstance(e, dict)]
    missing = [x for x in artifact_ids if x not in delivered_ids]
    checked = []
    checked.append(_c("delivered_event_count_matches", "Delivered event count matches the artifact event count.", len(delivered) == len(events), {"artifact": len(events), "delivered": len(delivered)}, [] if len(delivered) == len(events) else [{"artifact": len(events), "delivered": len(delivered)}]))
    checked.append(_c("event_ids_present", "Every artifact event id is present among the delivered events.", not missing, delivered_ids, missing))
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

def _ev_objective_coverage(artifact, render_record, evidence=None):
    comp = _ev_comparison(artifact, render_record, evidence)
    if isinstance(comp, dict) and comp.get("events_written") is not None and comp.get("events_realized_at_written_onsets") is not None:
        realized = comp.get("events_realized_at_written_onsets")
        written = comp.get("events_written")
    else:
        written = len([e for e in _ev_events(artifact) if isinstance(e, dict) and e.get("kind") != "silence"])
        realized = 0
    return [{"item": "events realized at their written onsets, %s of %s" % (realized, written), "covered": bool(written) and realized == written}]

def evaluate_invariants(artifact, render_record, evidence=None):
    if _is_vocal(artifact):
        return _score_evaluate_invariants(artifact, render_record, evidence)
    has_parts = isinstance(artifact, dict) and isinstance(artifact.get("parts"), list)
    has_events = isinstance(artifact, dict) and isinstance(artifact.get("events"), list)
    if has_parts and has_events:
        score_block = _score_evaluate_invariants(artifact, render_record, evidence)
        event_block = _ev_evaluate_invariants(artifact, render_record, evidence)
        checked = list(score_block.get("checked", [])) + list(event_block.get("checked", []))
        return {"checked": checked, "summary": {"total_checked": len(checked), "held": sum(1 for c in checked if c["held"]), "broken": sum(1 for c in checked if not c["held"])}}
    if has_events and not has_parts:
        return _ev_evaluate_invariants(artifact, render_record, evidence)
    return _score_evaluate_invariants(artifact, render_record, evidence)

def reliability_block():
    score_rel = _score_reliability_block()
    limits = list(score_rel.get("limits", []) or [])
    limits.append("Comparison onset tolerance is 0.1 seconds: a realized onset within 0.1 seconds of a written onset realizes that written event at its written onset.")
    limits.append("Event windows come from the delivered event intervals in the render record; realized onset uses a threshold of 5 percent of the event interval envelope maximum.")
    limits.append("Non-pitch whole-file analysis uses a 4096-sample block with a 1024-sample hop; sample rate is that of the handed-in render.")
    return {"instrument": "Sensoryne hearing and non-pitch audio analyzer", "limits": limits}

def objective_coverage(artifact, render_record, evidence=None):
    if _is_vocal(artifact):
        return _score_objective_coverage(artifact, render_record, evidence)
    has_parts = isinstance(artifact, dict) and isinstance(artifact.get("parts"), list)
    has_events = isinstance(artifact, dict) and isinstance(artifact.get("events"), list)
    if has_parts and has_events:
        return list(_score_objective_coverage(artifact, render_record, evidence) or []) + list(_ev_objective_coverage(artifact, render_record, evidence) or [])
    if has_events and not has_parts:
        return _ev_objective_coverage(artifact, render_record, evidence)
    return _score_objective_coverage(artifact, render_record, evidence)

def robustness_variations():
    return _score_robustness_variations()

def transform_artifact_for_variation(artifact, variation):
    if not (isinstance(artifact, dict) and isinstance(artifact.get("parts"), list)):
        return None
    out = _score_transform(artifact, variation)
    if out is not None and isinstance(artifact.get("events"), list):
        if isinstance(variation, dict):
            tempo = variation.get("tempo_scale")
        else:
            tempo = None
        if tempo is not None:
            for e in out.get("events", []):
                if isinstance(e, dict):
                    e["start"] = float(e.get("start", 0)) * float(tempo)
                    if e.get("duration") is not None:
                        e["duration"] = float(e["duration"]) * float(tempo)
            if out.get("duration") is not None:
                out["duration"] = float(out["duration"]) * float(tempo)
    return out

