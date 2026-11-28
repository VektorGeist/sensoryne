"""stage2/apps/music/perceive.py - Sensoryne hearing adapter and vocal hearing. Interpretation null."""
import numpy as np
from datetime import datetime, timezone
from .hearing.cochlea import Cochlea, load_wav
from .hearing.reading import aggregate_frames, read_frames
from .hearing.chladni import mode_weights, top_modes
from .hearing.token_pack import build_token_pack, model_token_pack
from .hearing.harmonic import harmonic_pitch
from .hearing.segmentation import segment_frames
from .hearing.phonetics import analyze_phonetics, track_formants
from .instruments import nearest_note

__all__ = ["perceive_wav", "perceive_render"]


def _j(v):
    if isinstance(v, dict):
        return {str(k): _j(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    if isinstance(v, np.ndarray):
        return _j(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _onsets(frames):
    if not frames:
        return 0
    flux = np.array([float(f.get("flux", 0)) for f in frames], dtype=np.float64)
    med = float(np.median(flux))
    mad = float(np.median(np.abs(flux - med)))
    thr = med + 3.0 * max(mad, 1e-12)
    count = 0
    above = False
    for x in flux:
        if x >= thr and not above:
            count += 1
            above = True
        elif x < thr:
            above = False
    return count


def _vibrato(pitches):
    if len(pitches) < 10:
        return {"rate_hz": None, "extent_cents": 0.0, "sustained_frames": len(pitches)}
    median = float(np.median(np.asarray(pitches, dtype=np.float64)))
    if median <= 0:
        return {"rate_hz": None, "extent_cents": 0.0, "sustained_frames": len(pitches)}
    x = np.log2(np.maximum(np.asarray(pitches, dtype=np.float64), 1e-9) / median)
    x = x - float(np.mean(x))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), d=1024 / 44100.0)
    mask = (freqs >= 3.0) & (freqs <= 8.0)
    rate = None
    if np.any(mask) and float(np.max(spec[mask])) > 0:
        rate = float(freqs[mask][int(np.argmax(spec[mask]))])
    return {"rate_hz": rate, "extent_cents": float(np.std(x) * 1200.0), "sustained_frames": len(pitches)}


def _vocal(path, lyrics):
    samples, sr = load_wav(path)
    frames = Cochlea(sr).analyze(samples)
    agg = aggregate_frames(frames, duration_seconds=len(samples) / sr if sr else None)
    descriptor = agg["descriptor"]
    weights = mode_weights(descriptor)
    reading = read_frames(frames, mode_weights=weights)
    segments = segment_frames(frames)
    phon = analyze_phonetics(frames, samples, sr)
    lines = []
    gaps = []
    for i in range(len(segments) - 1):
        gaps.append({"after_phrase": i + 1, "duration": float(segments[i + 1]["start_time"] - segments[i]["end_time"])})
    for idx, line in enumerate(lyrics):
        stated = len(line)
        phrase = segments[idx] if idx < len(segments) else None
        if phrase is None:
            lines.append({"line": idx + 1, "syllables": list(line), "stated_syllable_count": stated, "detected_phrase": False, "onset_time": None, "syllable_events_detected": 0, "median_pitch_hz": None, "note": None, "cents": None, "pitch_range_hz": None, "vibrato": {"rate_hz": None, "extent_cents": 0.0, "sustained_frames": 0}, "formants": {"f1": 0.0, "f2": 0.0, "f3": 0.0}})
            continue
        seg_frames = frames[phrase["start_frame"]:phrase["end_frame"] + 1]
        voiced = [float(f["f0"]) for f in seg_frames if f.get("voiced") or float(f.get("f0", 0)) > 0]
        median = float(np.median(voiced)) if voiced else 0.0
        nearest = nearest_note(median) if median > 0 else None
        sa = int(np.clip(phrase["start_time"] * sr, 0, len(samples)))
        sb = int(np.clip(phrase["end_time"] * sr, sa, len(samples)))
        track = track_formants(samples[sa:sb], sr)
        meanf = {"f1": float(np.mean([t["f1"] for t in track])) if track else 0.0, "f2": float(np.mean([t["f2"] for t in track])) if track else 0.0, "f3": float(np.mean([t["f3"] for t in track])) if track else 0.0}
        lines.append({"line": idx + 1, "syllables": list(line), "stated_syllable_count": stated, "detected_phrase": True, "onset_time": float(phrase["start_time"]), "syllable_events_detected": int(_onsets(seg_frames)), "median_pitch_hz": median if median > 0 else None, "note": nearest["name"] if nearest else None, "cents": float(nearest["cents"]) if nearest else None, "pitch_range_hz": [float(min(voiced)), float(max(voiced))] if voiced else None, "vibrato": _vibrato(voiced), "formants": meanf})
    vocal = {"lines": lines, "phrase_count": len(segments), "phrase_boundaries": [{"phrase": i + 1, "start_time": float(s["start_time"]), "end_time": float(s["end_time"]), "duration": float(s["duration"])} for i, s in enumerate(segments)], "breath_gaps": gaps}
    summary = "Vocal reading: %d lyric lines, %d phrases detected. " % (len(lyrics), len(segments)) + " ".join("Line %d onset %s syllable events %d of %d pitch %s." % (x["line"], x["onset_time"], x["syllable_events_detected"], x["stated_syllable_count"], x["note"]) for x in lines)
    pack = build_token_pack(path, 20)
    harm = harmonic_pitch(samples, sr, cochlea_f0=reading.get("f0") or agg.get("median_pitch") or None)
    model_pack = model_token_pack(pack)
    measurements = _j({"heard": summary, "summary": summary, "vocal": vocal, "note": reading.get("note"), "pitch_hz": reading.get("pitch_hz"), "loudness": reading.get("loudness"), "brightness": reading.get("brightness"), "energy_center_hz": reading.get("energy_center_hz"), "region": reading.get("region"), "pitch_trend": reading.get("pitch_trend"), "loud_trend": reading.get("loud_trend"), "descriptor": descriptor, "mode_weights": weights, "chladni_modes": top_modes(weights, 5), "harmonic_pitch_hz": harm["harmonic_pitch_hz"], "harmonic_confidence": harm["confidence"], "harmonic_disagreement": harm["disagreement_flag"], "polyphony_candidates": harm["candidates"], "token_pack": model_pack, "phonetics": phon})
    full = _j({"render_path": str(path), "perceived_at": datetime.now(timezone.utc).isoformat(), "instrument": "Sensoryne hearing", "measurements": measurements, "token_pack": pack, "vocal": vocal, "interpretation": None})
    return {"summary": summary, "measurements": measurements, "token_pack": _j(model_pack), "full": full, "render_path": str(path), "instrument": "Sensoryne hearing", "interpretation": None}


def perceive_wav(path, lyrics=None, artifact=None):
    if lyrics is not None:
        return _vocal(path, lyrics)
    if isinstance(artifact, dict) and artifact.get("kind") == "vocal":
        return _vocal(path, artifact.get("lyrics", []))
    samples, sr = load_wav(path)
    frames = Cochlea(sr).analyze(samples)
    agg = aggregate_frames(frames, duration_seconds=len(samples) / sr if sr else None)
    descriptor = agg["descriptor"]
    weights = mode_weights(descriptor)
    reading = read_frames(frames, mode_weights=weights)
    harm = harmonic_pitch(samples, sr, cochlea_f0=reading.get("f0") or agg.get("median_pitch") or None)
    pack = build_token_pack(path, 20)
    match = {"name": None, "nearest": None, "distance": None, "known": False, "candidates": []}
    model_pack = model_token_pack(pack)
    measurements = _j({"heard": reading.get("sentence"), "note": reading.get("note"), "cents": reading.get("cents"), "pitch_hz": reading.get("pitch_hz"), "loudness": reading.get("loudness"), "brightness": reading.get("brightness"), "energy_center_hz": reading.get("energy_center_hz"), "region": reading.get("region"), "peak_band": agg.get("peak_band"), "pitch_trend": reading.get("pitch_trend"), "loud_trend": reading.get("loud_trend"), "onset_rate": reading.get("onset_rate"), "duration": reading.get("duration"), "silent": reading.get("silent"), "tonal": reading.get("tonal"), "pitched": reading.get("pitched"), "descriptor": descriptor, "mode_weights": weights, "chladni_modes": top_modes(weights, 5), "harmonic_pitch_hz": harm["harmonic_pitch_hz"], "harmonic_midi": harm["harmonic_midi"], "harmonic_confidence": harm["confidence"], "harmonic_disagreement": harm["disagreement_flag"], "polyphony_candidates": harm["candidates"], "gallery_match": match, "token_pack": model_pack, "image_reading": reading.get("image_reading")})
    full = _j({"render_path": str(path), "perceived_at": datetime.now(timezone.utc).isoformat(), "instrument": "Sensoryne hearing", "measurements": measurements, "token_pack": pack, "reading": reading, "harmonic": harm, "interpretation": None})
    return {"summary": reading.get("sentence"), "measurements": measurements, "token_pack": _j(model_pack), "full": full, "render_path": str(path), "instrument": "Sensoryne hearing", "interpretation": None}


perceive_render = perceive_wav
