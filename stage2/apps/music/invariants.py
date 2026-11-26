"""stage2/apps/music/invariants.py - music invariants, reliability, coverage, robustness variations."""
import copy

def _c(i, d, held, value, violations=None):
    return {"id": i, "description": d, "held": bool(held), "value": value, "violations": list(violations or [])}

def _parts(artifact):
    return artifact.get("parts", []) if isinstance(artifact, dict) else []

def _is_vocal(artifact):
    return isinstance(artifact, dict) and artifact.get("kind") == "vocal"

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
    total = 0.0
    for p in parts:
        for n in p.get("notes", []) if isinstance(p, dict) else []:
            dur = n.get("dur")
            total = max(total, float(n.get("start", 0)) + (float(dur) if dur is not None else 0.0))
    checked.append(_c("total_duration_15_20", "Total duration is between 15 and 20 seconds.", 15.0 <= total <= 20.0, total, [] if 15.0 <= total <= 20.0 else [{"total": total}]))
    harmony = next((p for p in parts if isinstance(p, dict) and p.get("name") == "harmony"), None)
    has_harmony = bool(harmony and any(float(n.get("start", 0)) >= 8.0 for n in harmony.get("notes", [])))
    checked.append(_c("harmony_at_or_after_8", "At least one harmony note starts at or after 8.0 seconds.", has_harmony, has_harmony, [] if has_harmony else [{"part": "harmony", "at_or_after": 8.0}]))
    violations = []
    counts = {}
    for p in parts:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        gain = float(p.get("gain", 1.0))
        cap = 0.41 if name == "bass" else 0.52
        counts[name] = len(p.get("notes", []))
        for n in p.get("notes", []):
            eff = float(n.get("vel", 0)) * gain
            if eff > cap:
                violations.append({"part": name, "start": float(n.get("start", 0)), "midi": n.get("midi"), "effective_velocity": eff, "cap": cap})
    checked.append(_c("effective_velocity_caps", "No note in any part exceeds that part's effective velocity cap (0.52, bass 0.41).", not violations, [{"part": k, "cap": 0.41 if k == "bass" else 0.52} for k in counts], violations))
    try:
        from .instruments import get_instrument
        inst_name = artifact.get("instrument") or "piano" if isinstance(artifact, dict) else "piano"
        inst = get_instrument(inst_name)
        rng = inst.get("written") or inst.get("sounding") or inst.get("absolute_range")
    except Exception:
        rng = None
        inst_name = artifact.get("instrument") if isinstance(artifact, dict) else None
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
    return {"instrument": "Sensoryne hearing", "limits": ["Sample rate and FFT resolution are those of the handed-in render and the cochlea analysis block.", "Fundamental pitch is the strongest interpolated peak in the pitch band.", "Syllable and formant figures are estimates from the stated phonetic methods.", "The hearing works on handed-in renders only; there is no live capture."]}

def objective_coverage(artifact, render_record, evidence=None):
    out = []
    if _is_vocal(artifact):
        for i, line in enumerate(artifact.get("lyrics", [])):
            out.append({"item": "lyric line %d with %d stated syllables" % (i + 1, len(line)), "present": bool(line)})
        return out
    parts = artifact.get("parts", []) if isinstance(artifact, dict) else []
    counts = {p.get("name"): len(p.get("notes", [])) for p in parts if isinstance(p, dict)}
    for name in ("melody", "harmony", "bass"):
        out.append({"item": "part %s" % name, "present": counts.get(name, 0) > 0})
    total_notes = sum(counts.values())
    out.append({"item": "note count %d" % total_notes, "present": total_notes > 0})
    if isinstance(artifact, dict) and artifact.get("instrument"):
        out.append({"item": "instrument %s" % artifact["instrument"], "present": True})
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
