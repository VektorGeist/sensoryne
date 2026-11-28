"""stage2/apps/music/instrument.py - non-pitch analyzer for sound events.

Measures delivered audio only: per-event realized onset and
duration, spectral centroid and bandwidth per event, the
loudness contour of the whole, silence share, and event
overlap. Reports measurements only. This is the onset
measurement the merged Music & Sound application uses for
written events.
"""
import numpy as np
from scipy.io import wavfile

METHOD = "audio loaded as mono float samples; whole-file frames use a 4096-sample block with a 1024-sample hop, Hanning window, magnitude spectrum, RMS loudness in dB, magnitude-weighted spectral centroid and bandwidth; a frame is silent when RMS is at or below 0.0001; per-event realized onset and duration are the first and last samples of the event interval whose smoothed envelope exceeds 5 percent of that interval maximum; overlap is the time covered by at least two delivered event intervals"

_EPS = 1e-12
_SILENT_RMS = 1e-4
_FRAME_SIZE = 4096
_HOP = 1024


def _load(path):
    sample_rate, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        samples = data.astype(np.float64) / float(max(abs(info.min), info.max))
    else:
        samples = data.astype(np.float64)
    return samples, int(sample_rate)


def _spectral_stats(segment, sample_rate):
    n = len(segment)
    if n == 0:
        return 0.0, 0.0
    windowed = segment * np.hanning(n)
    spectrum = np.fft.rfft(windowed)
    magnitude = np.abs(spectrum)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    total = float(np.sum(magnitude))
    if total <= _EPS:
        return 0.0, 0.0
    centroid = float(np.sum(freqs * magnitude) / total)
    bandwidth = float(np.sqrt(np.sum(magnitude * (freqs - centroid) ** 2) / total))
    return centroid, bandwidth


def _frame_analysis(samples, sample_rate):
    if len(samples) == 0:
        return [], 0.0, 0.0, 0.0, 0.0
    count = 1 + max(0, (len(samples) - _FRAME_SIZE) // _HOP) if len(samples) >= _FRAME_SIZE else 1
    padded = np.zeros(count * _HOP + _FRAME_SIZE, dtype=np.float64)
    padded[:len(samples)] = samples
    window = np.hanning(_FRAME_SIZE)
    frames = []
    centroids = []
    bandwidths = []
    silent = 0
    for i in range(count):
        block = padded[i * _HOP:i * _HOP + _FRAME_SIZE]
        rms = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
        loudness = float(20.0 * np.log10(rms + _EPS))
        centroid, bandwidth = _spectral_stats(block * window, sample_rate)
        is_silent = rms <= _SILENT_RMS
        silent += 1 if is_silent else 0
        if not is_silent:
            centroids.append(centroid)
            bandwidths.append(bandwidth)
        frames.append({"time": float(i * _HOP / float(sample_rate)), "rms": rms, "loudness_db": loudness, "centroid_hz": centroid, "bandwidth_hz": bandwidth, "silent": bool(is_silent)})
    silence_share = silent / float(count) if count else 0.0
    median_centroid = float(np.median(centroids)) if centroids else 0.0
    median_bandwidth = float(np.median(bandwidths)) if bandwidths else 0.0
    return frames, median_centroid, median_bandwidth, float(silence_share), float(np.max([f["rms"] for f in frames])) if frames else 0.0


def _overlap_seconds(events):
    points = []
    for e in events:
        try:
            start = float(e.get("start", 0)); end = start + float(e.get("duration", 0))
        except Exception:
            continue
        if end > start:
            points.append((start, 1)); points.append((end, -1))
    points.sort(key=lambda x: (x[0], -x[1]))
    total = 0.0
    active = 0
    last = None
    for time, delta in points:
        if last is not None and active >= 2:
            total += max(0.0, time - last)
        active += delta
        last = time
    pairs = []
    for i, a in enumerate(events):
        for b in events[i + 1:]:
            try:
                overlap = min(float(a.get("start", 0)) + float(a.get("duration", 0)), float(b.get("start", 0)) + float(b.get("duration", 0))) - max(float(a.get("start", 0)), float(b.get("start", 0)))
            except Exception:
                overlap = 0.0
            if overlap > 0:
                pairs.append({"first": a.get("id"), "second": b.get("id"), "overlap_seconds": float(overlap)})
    return float(total), pairs


def _event_measurement(event, samples, sample_rate):
    start = float(event.get("start", 0))
    declared = float(event.get("duration", 0))
    a = int(np.clip(round(start * sample_rate), 0, len(samples)))
    b = int(np.clip(round((start + declared) * sample_rate), a, len(samples)))
    segment = samples[a:b]
    if segment.size:
        envelope = np.convolve(np.abs(segment), np.ones(256) / 256.0, mode="same")
        peak_env = float(np.max(envelope))
        threshold = max(1e-5, 0.05 * peak_env) if peak_env > 0 else None
        active = np.where(envelope > threshold)[0] if threshold is not None else np.array([], dtype=int)
    else:
        active = np.array([], dtype=int)
    if active.size:
        first = int(active[0]); last = int(active[-1])
        onset = float(start + first / float(sample_rate))
        realized_duration = float((last - first + 1) / float(sample_rate))
        span = segment[first:last + 1]
    else:
        onset = None
        realized_duration = 0.0
        span = segment
    centroid, bandwidth = _spectral_stats(span, sample_rate)
    return {
        "id": event.get("id"),
        "kind": event.get("kind"),
        "declared_start": start,
        "declared_duration": declared,
        "onset_time": onset,
        "realized_onset": onset,
        "realized_duration": realized_duration,
        "peak_amplitude": float(np.max(np.abs(segment))) if segment.size else 0.0,
        "rms": float(np.sqrt(np.mean(segment ** 2))) if segment.size else 0.0,
        "spectral_centroid_hz": float(centroid),
        "spectral_bandwidth_hz": float(bandwidth),
        "gain": float(event.get("gain", 0)),
    }


def analyze_audio(path, delivered_events=None, duration=None):
    samples, sample_rate = _load(path)
    delivered_events = list(delivered_events or [])
    file_duration = float(len(samples) / float(sample_rate)) if sample_rate else 0.0
    frames, median_centroid, median_bandwidth, silence_share, _peak = _frame_analysis(samples, sample_rate)
    measured_events = [_event_measurement(e, samples, sample_rate) for e in delivered_events if isinstance(e, dict)]
    overlap_seconds, overlap_pairs = _overlap_seconds(delivered_events)
    summary = "Rendered audio %.2f seconds at %d Hz, %d delivered events measured. Median spectral centroid %.1f Hz, median spectral bandwidth %.1f Hz, silence share %.4f, event overlap %.4f seconds." % (file_duration, sample_rate, len(measured_events), median_centroid, median_bandwidth, silence_share, overlap_seconds)
    measurements = {
        "instrument": "non-pitch audio analyzer",
        "method": METHOD,
        "sample_rate": int(sample_rate),
        "duration_seconds": float(duration if duration is not None else file_duration),
        "file_duration_seconds": float(file_duration),
        "event_count": len(measured_events),
        "events": measured_events,
        "loudness_contour": frames,
        "median_spectral_centroid_hz": float(median_centroid),
        "median_spectral_bandwidth_hz": float(median_bandwidth),
        "silence_share": float(silence_share),
        "event_overlap_seconds": float(overlap_seconds),
        "event_overlap_pairs": overlap_pairs,
    }
    full = {"instrument": "non-pitch audio analyzer", "method": METHOD, "render_path": str(path), "delivered_events": delivered_events, "measurements": measurements, "interpretation": None}
    return {"summary": summary, "measurements": measurements, "full": full, "render_path": str(path), "instrument": "non-pitch audio analyzer", "interpretation": None}


def perceive_audio(path, delivered_events=None, duration=None):
    return analyze_audio(path, delivered_events, duration)
