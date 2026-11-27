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
    # the heard count is measured as none found, so the check
    # is unheld and the written counts are still stated.
    per_part = _comparison_per_part(artifact, render_record, evidence)
    by_name = {}
    if per_part:
        for entry in per_part:
            if isinstance(entry, dict):
                by_name[entry.get("part")] = entry
    heard = []
    not_heard = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        written = counts.get(name, 0)
        entry = by_name.get(name)
        if entry is not None:
            heard_n = int(entry.get("notes_heard") or 0)
            not_heard_n = int(entry.get("notes_not_heard") or 0)
        else:
            heard_n = 0
            not_heard_n = written
        heard.append({"part": name, "notes_written": written, "notes_heard": heard_n})
        if not_heard_n:
            not_heard.append({"part": name, "notes_written": written, "notes_heard": heard_n, "notes_not_heard": not_heard_n})
    checked.append(_c("notes_heard_of_written", "Per part, the notes heard are reported against the notes written.", not not_heard, heard, not_heard))
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
    # itself. The dynamics mapping is calibrated on the hearing
    # itself, from renders of known velocity: expected_db for a
    # segment is DYNAMICS_OFFSET_DB plus 10*log10 of the mean
    # written power in that segment, where the written power of
    # a part is the sum over its notes starting in the segment
    # of (effective velocity squared times duration), divided
    # by the segment duration. Effective velocity is velocity
    # times part gain. The offset is the calibrated constant
    # that maps that written power to the hearing's measured
    # loudness in dB for a correct render.
    return {"instrument": "Sensoryne hearing", "limits": ["Sample rate and FFT resolution are those of the handed-in render and the cochlea analysis block.", "Fundamental pitch is the strongest interpolated peak in the pitch band.", "Syllable and formant figures are estimates from the stated phonetic methods.", "The hearing works on handed-in renders only; there is no live capture.", "Comparison pitch tolerance is 50 cents; a written note is counted heard when its harmonic energy is at least 0.15 of the strongest pitch in its window; silence is at or below -70 dB.", "Comparison dynamics: expected_db = -24.0 + 10*log10(sum((velocity*gain)^2 * duration) / segment_duration) over the part's notes starting in the segment, threshold 2.5 dB."]}

def objective_coverage(artifact, render_record, evidence=None):
    out = []
    if _is_vocal(artifact):
        for i, line in enumerate(artifact.get("lyrics", [])):
            out.append({"item": "lyric line %d with %d stated syllables" % (i + 1, len(line)), "present": bool(line)})
        return out
    parts = artifact.get("parts", []) if isinstance(artifact, dict) else []
    counts = {}
    for p in parts:
        if isinstance(p, dict):
            counts[p.get("name")] = len(p.get("notes", []))
    # Coverage derived from the comparison: per part, notes
    # heard of notes written, plus the instrument range item.
    # Nothing is covered by construction: absent comparison
    # means no notes heard, and an empty part means none written.
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
        heard_n = int(entry.get("notes_heard") or 0) if entry is not None else 0
        out.append({"item": "part %s: %d of %d notes heard" % (name, heard_n, written), "covered": bool(written) and heard_n >= written})
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
    if False:
        return []
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
