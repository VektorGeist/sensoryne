"""music_renderer.py - Music & Sound renderer entry for the merged application.

renderer(artifact, out_path, faults) renders to one audio file:

* a score-only work through the existing music rendering
  route (the sibling embers-piano "music_renderer" module
  when present, falling back to the same call signature);
* an event-only work through the event synthesis formerly
  in the former event renderer (folded here as
  _event_renderer);
* a work with both to one audio file containing both: the
  score render and the event render are mixed sample by
  sample into the single output file.

No separate event renderer registration remains; this
module is the one renderer entry for the merged application.
"""
import copy
import hashlib
import os
import sys
import wave

import numpy as np

SAMPLE_RATE = 44100


def _param(event, key, default=None):
    value = event.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    params = event.get("params")
    if isinstance(params, dict):
        value = params.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return default


def _seed_for(event, index):
    text = "%s:%d" % (event.get("id"), index)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _filtered_noise(n, sample_rate, cutoff, bandwidth, rng):
    x = rng.standard_normal(n)
    if n == 0:
        return x
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    if bandwidth is not None and bandwidth > 0:
        low = max(0.0, cutoff - bandwidth / 2.0); high = cutoff + bandwidth / 2.0
        mask = (freqs >= low) & (freqs <= high)
    else:
        mask = freqs <= cutoff
    y = np.fft.irfft(spectrum * mask, n=n)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return y / peak if peak > 1e-12 else y


def _event_wave(event, n, sample_rate, index):
    kind = event.get("kind")
    if n <= 0 or kind == "silence":
        return np.zeros(max(0, n), dtype=np.float64)
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    rng = np.random.default_rng(_seed_for(event, index))
    cutoff = _param(event, "filter_cutoff", 440.0)
    bandwidth = _param(event, "bandwidth", 0.0)
    if kind == "noise burst":
        y = rng.standard_normal(n)
        fade = max(1, int(0.002 * sample_rate))
        if n > fade * 2:
            y[:fade] *= np.linspace(0, 1, fade); y[-fade:] *= np.linspace(1, 0, fade)
        return y
    if kind == "filtered noise":
        return _filtered_noise(n, sample_rate, float(cutoff or 4000.0), float(bandwidth or 0.0), rng)
    if kind == "sine cluster":
        center = float(cutoff or 440.0)
        y = np.zeros(n, dtype=np.float64)
        for factor in (0.94, 1.0, 1.06):
            y += np.sin(2.0 * np.pi * center * factor * t)
        return y / 3.0
    if kind == "percussive hit":
        freq = float(cutoff or 180.0)
        tau = max(0.02, float(event.get("duration") or 0.1) / 4.0)
        envelope = np.exp(-t / tau)
        return np.sin(2.0 * np.pi * freq * t) * envelope + 0.35 * rng.standard_normal(n) * envelope
    raise ValueError("unknown event kind: %r" % (kind,))


def _apply_event_faults(events, faults):
    events = copy.deepcopy(events)
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_event":
            events = [e for e in events if not (isinstance(e, dict) and e.get("id") == fault.get("id"))]
        elif kind == "shift_event_start":
            for e in events:
                if isinstance(e, dict) and e.get("id") == fault.get("id"):
                    if fault.get("new_start") is not None:
                        e["start"] = float(fault["new_start"])
                    elif fault.get("delta") is not None:
                        e["start"] = float(e.get("start", 0)) + float(fault["delta"])
        elif kind == "raise_event_gain":
            for e in events:
                if isinstance(e, dict) and e.get("id") == fault.get("id"):
                    e["gain"] = float(e.get("gain", 0)) * float(fault.get("factor", 1.0))
        elif kind in ("part_gain", "swell", "drop_note"):
            continue
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return events


def _apply_score_faults(score, faults):
    score = copy.deepcopy(score)
    for fault in faults or []:
        kind = fault.get("kind")
        if kind in ("drop_event", "shift_event_start", "raise_event_gain"):
            continue
        elif kind == "part_gain":
            for part in score.get("parts", []):
                if isinstance(part, dict) and part.get("name") == fault.get("part"):
                    part["gain"] = float(part.get("gain", 1.0)) * float(fault.get("factor", 1.0))
        elif kind == "swell":
            for part in score.get("parts", []):
                if not isinstance(part, dict):
                    continue
                for note in part.get("notes", []) or []:
                    if isinstance(note, dict) and float(fault.get("from", 0)) <= float(note.get("start", 0)) < float(fault.get("to", 0)):
                        note["vel"] = max(0.0, min(1.0, float(note.get("vel", 0)) * float(fault.get("factor", 1.0))))
        elif kind == "drop_note":
            for part in score.get("parts", []):
                if not isinstance(part, dict) or part.get("name") != fault.get("part"):
                    continue
                kept = []
                dropped = False
                for note in part.get("notes", []) or []:
                    if isinstance(note, dict) and not dropped and float(note.get("start", -1)) == float(fault.get("start", -2)) and int(note.get("midi", -1)) == int(fault.get("midi", -2)):
                        dropped = True
                        continue
                    kept.append(note)
                part["notes"] = kept
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return score


def _music_renderer_callable():
    # Resolve the sibling embers-piano renderer. This module is
    # itself named music_renderer, so an unguarded import would
    # import itself; the sibling is loaded only through a
    # distinct module name and only when its folder exists.
    candidate = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embers-piano"))
    if not os.path.isdir(candidate):
        return None
    path = os.path.join(candidate, "music_renderer.py")
    if not os.path.isfile(path):
        return None
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location("embers_piano_music_renderer", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        spec.loader.exec_module(module)
        return getattr(module, "renderer", None)
    except Exception:
        return None


def _render_score_to_samples(score_artifact, faults):
    from stage2.apps.music.app import score_total_length
    total = score_total_length(score_artifact)
    total = max(total, 0.0)
    count = int(round(total * SAMPLE_RATE))
    buffer = np.zeros(max(0, count) + SAMPLE_RATE // 2 if count else 0, dtype=np.float64)
    faulted = _apply_score_faults(score_artifact, faults)
    for part in faulted.get("parts", []):
        if not isinstance(part, dict):
            continue
        gain = float(part.get("gain", 1.0))
        for note in part.get("notes", []) or []:
            if not isinstance(note, dict):
                continue
            start = float(note.get("start", 0.0))
            dur = float(note["dur"]) if note.get("dur") is not None else 0.5
            amp = float(note.get("vel", 0.0)) * gain
            n = int(round(dur * SAMPLE_RATE))
            if n <= 0 or amp <= 0:
                continue
            t = np.arange(n, dtype=np.float64) / float(SAMPLE_RATE)
            freq = 440.0 * (2.0 ** ((int(note["midi"]) - 69) / 12.0))
            envelope = np.ones(n, dtype=np.float64)
            fade = max(1, int(0.01 * SAMPLE_RATE))
            if n > fade * 2:
                envelope[:fade] = np.linspace(0.0, 1.0, fade)
                envelope[-fade:] = np.linspace(1.0, 0.0, fade)
            sample = amp * envelope * np.sin(2.0 * np.pi * freq * t)
            begin = int(round(start * SAMPLE_RATE))
            end = min(len(buffer), begin + n)
            if begin < 0:
                sample = sample[-begin:]
                begin = 0
            if end > begin:
                buffer[begin:end] += sample[: end - begin]
    return buffer, float(total)


def _event_mix(events, total_seconds):
    total = int(round(float(total_seconds or 0) * SAMPLE_RATE))
    mix = np.zeros(max(0, total), dtype=np.float64)
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        start = int(round(float(event.get("start", 0)) * SAMPLE_RATE))
        count = int(round(float(event.get("duration", 0)) * SAMPLE_RATE))
        if count <= 0 or start >= total:
            continue
        count = min(count, total - start)
        rendered = _event_wave(event, count, SAMPLE_RATE, index) * float(event.get("gain", 0))
        mix[start:start + count] += rendered
    return mix


def _write_wav(path, mix):
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.89:
        mix = mix * (0.89 / peak)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    pcm = np.clip(mix, -1.0, 1.0)
    pcm16 = np.round(pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm16.tobytes())


def _event_renderer(artifact, out_path, faults):
    duration = float(artifact.get("duration") or 0)
    delivered = _apply_event_faults(artifact.get("events", []), faults)
    mix = _event_mix(delivered, duration)
    _write_wav(out_path, mix)
    return {"path": str(out_path), "kind": "audio", "duration": float(duration), "sample_rate": SAMPLE_RATE, "delivered_events": delivered, "event_count": len(delivered), "faults_applied": copy.deepcopy(list(faults or []))}


def renderer(artifact, out_path, faults=None):
    faults = list(faults or [])
    if isinstance(artifact, dict) and artifact.get("kind") == "vocal":
        raise ValueError("vocal works use the vocal render route")
    has_parts = isinstance(artifact, dict) and isinstance(artifact.get("parts"), list)
    has_events = isinstance(artifact, dict) and isinstance(artifact.get("events"), list)
    if has_events and not has_parts:
        return _event_renderer(artifact, out_path, faults)
    if has_parts and not has_events:
        music_route = _music_renderer_callable()
        if music_route is not None:
            return music_route(copy.deepcopy(artifact), str(out_path), copy.deepcopy([f for f in faults if f.get("kind") in ("part_gain", "swell", "drop_note")]))
        samples, total = _render_score_to_samples(artifact, faults)
        _write_wav(out_path, samples)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        return {"path": str(out_path), "kind": "audio", "duration": float(total), "sample_rate": SAMPLE_RATE, "peak": peak, "faults_applied": copy.deepcopy(faults)}
    if has_parts and has_events:
        music_route = _music_renderer_callable()
        delivered_events = _apply_event_faults(artifact.get("events", []), faults)
        if music_route is not None:
            score_record = music_route(copy.deepcopy(artifact), str(out_path), copy.deepcopy([f for f in faults if f.get("kind") in ("part_gain", "swell", "drop_note")]))
            score_path = score_record.get("path") if isinstance(score_record, dict) else str(out_path)
        else:
            score_path = None
            score_record = None
        score_mix = None
        score_total = 0.0
        if score_path and os.path.isfile(str(score_path)):
            from scipy.io import wavfile
            sr, data = wavfile.read(str(score_path))
            if data.ndim > 1:
                data = data.mean(axis=1)
            if np.issubdtype(data.dtype, np.integer):
                info = np.iinfo(data.dtype)
                score_mix = data.astype(np.float64) / float(max(abs(info.min), info.max))
            else:
                score_mix = data.astype(np.float64)
            score_total = float(len(score_mix) / float(sr)) if sr else 0.0
            if int(sr) != SAMPLE_RATE:
                score_mix = None
        if score_mix is None:
            score_mix, score_total = _render_score_to_samples(artifact, faults)
        event_seconds = float(artifact.get("duration") or 0)
        total_seconds = max(score_total, event_seconds)
        total = int(round(total_seconds * SAMPLE_RATE))
        mix = np.zeros(max(0, total), dtype=np.float64)
        n = min(len(score_mix), len(mix))
        if n:
            mix[:n] += score_mix[:n]
        event_mix = _event_mix(delivered_events, event_seconds)
        n = min(len(event_mix), len(mix))
        if n:
            mix[:n] += event_mix[:n]
        _write_wav(out_path, mix)
        result = dict(score_record) if isinstance(score_record, dict) else {}
        result.update({"path": str(out_path), "kind": "audio", "duration": float(total_seconds), "sample_rate": SAMPLE_RATE, "delivered_events": delivered_events, "event_count": len(delivered_events), "faults_applied": copy.deepcopy(faults)})
        return result
    raise ValueError("artifact must carry a score, an event list, or both")
