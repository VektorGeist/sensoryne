"""sound_design_renderer.py - composing-side sound design renderer.

renderer(artifact, out_path, faults): synthesizes the events
to audio with numpy, faults applied in the rendering only.
out_path is a single audio file path for every renderer that
produces one file. A renderer that produces several files
(this one produces one) uses out_path as its sole output file.
"""
import copy, hashlib, os, wave
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

def _apply_faults(events, faults):
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
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return events

def renderer(artifact, out_path, faults):
    duration = float(artifact.get("duration") or 0)
    total = int(round(duration * SAMPLE_RATE))
    mix = np.zeros(max(0, total), dtype=np.float64)
    delivered = _apply_faults(artifact.get("events", []), faults)
    for index, event in enumerate(delivered):
        if not isinstance(event, dict):
            continue
        start = int(round(float(event.get("start", 0)) * SAMPLE_RATE))
        count = int(round(float(event.get("duration", 0)) * SAMPLE_RATE))
        if count <= 0 or start >= total:
            continue
        count = min(count, total - start)
        rendered = _event_wave(event, count, SAMPLE_RATE, index) * float(event.get("gain", 0))
        mix[start:start + count] += rendered
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.89:
        mix *= 0.89 / peak
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    pcm = np.clip(mix, -1.0, 1.0)
    pcm16 = np.round(pcm * 32767.0).astype(np.int16)
    with wave.open(out_path, "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm16.tobytes())
    return {"path": str(out_path), "kind": "audio", "duration": float(duration), "sample_rate": SAMPLE_RATE, "delivered_events": delivered, "event_count": len(delivered), "faults_applied": copy.deepcopy(list(faults or []))}
