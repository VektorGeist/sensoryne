"""stage2/apps/music/hearing/reading.py - gate, aggregation, and
the plain language reading.

Frames from the cochlea are classified and aggregated into one
reading of the whole sound.  The reading composes a plain
language sentence and its structured fields.  A second,
independent reading renders the place-by-time field as an
image and reads that image back into numbers.

This module measures and describes only.  It judges nothing.
Chladni mode weights are accepted from the plate model when
the caller has them; they are reported, never invented here.
"""

import math

import numpy as np
from PIL import Image

from stage2.apps.music.instruments import nearest_note

from .cochlea import PLACE_FREQS

__all__ = [
    "SILENT_ENERGY",
    "SILENT_LOUD_DB",
    "TONAL_FLATNESS_MAX",
    "VOICED_FRACTION_MIN",
    "classify_frames",
    "aggregate_frames",
    "image_reading",
    "read_frames",
]

SILENT_ENERGY = 1e-7
SILENT_LOUD_DB = -70.0
TONAL_FLATNESS_MAX = 0.35
VOICED_FRACTION_MIN = 0.30

_EPS = 1e-12


def _is_silent(frame):
    return float(frame.get("energy", 0.0)) < SILENT_ENERGY or float(
        frame.get("loud", frame.get("loudness", -120.0))
    ) <= SILENT_LOUD_DB


def _median(values, default=0.0):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return float(default)
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _thirds(values):
    if len(values) < 3:
        return values, [], values
    cut_a = len(values) // 3
    cut_b = len(values) - cut_a
    return values[:cut_a], values[cut_a:cut_b], values[cut_b:]


def _trend_from_thirds(first, last, rising_ratio=None, rising_db=None):
    if not first or not last:
        return "steady"
    first_value = _median(first)
    last_value = _median(last)
    if rising_ratio is not None:
        if first_value <= 0 or last_value <= 0:
            return "steady"
        ratio = last_value / first_value
        if ratio > rising_ratio:
            return "rising"
        if ratio < 1.0 / rising_ratio:
            return "falling"
        return "steady"
    difference = last_value - first_value
    if difference > rising_db:
        return "rising"
    if difference < -rising_db:
        return "falling"
    return "steady"


def classify_frames(frames):
    """Add a classification to each frame and return the list.

    Silent is detected explicitly.  A frame is voiced when it
    is not silent and carries a fundamental pitch.
    """
    classified = []
    for frame in frames:
        entry = dict(frame)
        silent = _is_silent(entry)
        voiced = (not silent) and float(entry.get("f0", entry.get("pitch", 0.0))) > 0.0
        entry["silent"] = bool(silent)
        entry["voiced"] = bool(voiced)
        classified.append(entry)
    return classified


def aggregate_frames(frames, duration_seconds=None):
    """Aggregate classified or raw frames into a whole reading."""
    frames = classify_frames(frames)
    if not frames:
        return {
            "duration": 0.0,
            "frame_count": 0,
            "silent": True,
            "tonal": False,
            "pitched": False,
            "voiced_fraction": 0.0,
            "median_flatness": 0.0,
            "descriptor": [0.0] * len(PLACE_FREQS),
            "median_place": [0.0] * len(PLACE_FREQS),
            "loud": -120.0,
            "loudness": -120.0,
            "onset_count": 0,
            "onset_rate": 0.0,
            "pitch_trend": "steady",
            "loud_trend": "steady",
            "peak_band": None,
            "median_pitch": 0.0,
            "brightness": 0.0,
            "energy_center_hz": 0.0,
        }

    duration = float(duration_seconds) if duration_seconds else float(frames[-1]["time"])
    if duration <= 0:
        duration = float(len(frames))

    sounding = [f for f in frames if not f["silent"]]
    voiced = [f for f in frames if f["voiced"]]
    voiced_fraction = len(voiced) / float(len(frames)) if frames else 0.0
    median_flatness = _median([f["flatness"] for f in sounding], default=0.0) if sounding else 0.0
    tonal = bool(sounding) and median_flatness <= TONAL_FLATNESS_MAX
    pitched = bool(sounding) and voiced_fraction >= VOICED_FRACTION_MIN

    place_matrix = np.asarray([f["place"] for f in frames], dtype=np.float64)
    descriptor = np.median(place_matrix, axis=0) if place_matrix.size else np.zeros(len(PLACE_FREQS))
    descriptor_total = float(np.sum(descriptor))
    if descriptor_total > _EPS:
        descriptor = descriptor / descriptor_total

    loudness = _median([f["loud"] for f in sounding], default=-120.0) if sounding else -120.0
    median_pitch = _median([f["f0"] for f in voiced], default=0.0) if voiced else 0.0
    brightness = _median([f["centroid"] for f in sounding], default=0.0) if sounding else 0.0

    fluxes = np.asarray([f["flux"] for f in frames], dtype=np.float64)
    if fluxes.size:
        flux_median = float(np.median(fluxes))
        flux_mad = float(np.median(np.abs(fluxes - flux_median)))
        flux_threshold = flux_median + 3.0 * max(flux_mad, _EPS)
        onset_count = 0
        above = False
        for flux in fluxes:
            if flux >= flux_threshold and not above:
                onset_count += 1
                above = True
            elif flux < flux_threshold:
                above = False
    else:
        flux_threshold = 0.0
        onset_count = 0
    onset_rate = onset_count / duration if duration > 0 else 0.0

    pitch_first, _pitch_middle, pitch_last = _thirds([f["f0"] for f in voiced])
    loud_first, _loud_middle, loud_last = _thirds([f["loud"] for f in sounding])
    pitch_trend = _trend_from_thirds(pitch_first, pitch_last, rising_ratio=1.03)
    loud_trend = _trend_from_thirds(loud_first, loud_last, rising_db=3.0)

    band_totals = {
        band: float(np.sum([f[band] for f in frames]))
        for band in ("bass", "mid", "high")
    }
    peak_band = max(band_totals, key=lambda band: band_totals[band]) if frames else None

    energy_center = float(np.sum(PLACE_FREQS * descriptor)) if descriptor.size else 0.0

    return {
        "duration": float(duration),
        "frame_count": len(frames),
        "silent": not bool(sounding),
        "tonal": bool(tonal),
        "pitched": bool(pitched),
        "voiced_fraction": float(voiced_fraction),
        "median_flatness": float(median_flatness),
        "descriptor": [float(v) for v in descriptor],
        "median_place": [float(v) for v in descriptor],
        "loud": float(loudness),
        "loudness": float(loudness),
        "onset_count": int(onset_count),
        "onset_rate": float(onset_rate),
        "flux_threshold": float(flux_threshold),
        "pitch_trend": pitch_trend,
        "loud_trend": loud_trend,
        "peak_band": peak_band,
        "band_totals": band_totals,
        "median_pitch": float(median_pitch),
        "brightness": float(brightness),
        "energy_center_hz": float(energy_center),
    }


def _region_name(freq_hz):
    if freq_hz <= 0:
        return "silent"
    if freq_hz < 60.0:
        return "sub-bass"
    if freq_hz < 250.0:
        return "bass"
    if freq_hz < 500.0:
        return "low mid"
    if freq_hz < 2000.0:
        return "mid"
    if freq_hz < 6000.0:
        return "high mid"
    if freq_hz < 12000.0:
        return "presence"
    return "brilliance"


def _strongest_modes(mode_weights, count=3):
    if not mode_weights:
        return []
    values = [float(v) for v in mode_weights]
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return [
        {"mode": int(index), "weight": float(values[index])}
        for index in order[:count]
    ]


def image_reading(frames):
    """Render the place-by-time field as an image and read it back.

    Returns peak band, motion, and brightness, with the image
    array itself.  The reading is taken from the image pixels,
    not from the frames directly.
    """
    if not frames:
        return {
            "peak_band": None,
            "peak_place_hz": 0.0,
            "motion": 0.0,
            "brightness": 0.0,
            "image": np.zeros((len(PLACE_FREQS), 1), dtype=np.float64),
        }
    field = np.asarray([f["place"] for f in frames], dtype=np.float64).T
    peak = float(np.max(field)) if field.size else 0.0
    if peak > _EPS:
        field = field / peak
    image_array = np.clip(field * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_array, mode="L")
    read_back = np.asarray(image, dtype=np.float64) / 255.0

    row_means = np.mean(read_back, axis=1) if read_back.size else np.zeros(len(PLACE_FREQS))
    peak_row = int(np.argmax(row_means)) if row_means.size else 0
    if read_back.shape[1] > 1:
        motion = float(np.mean(np.abs(np.diff(read_back, axis=1))))
    else:
        motion = 0.0
    overall = float(np.mean(read_back)) if read_back.size else 0.0
    upper = float(np.mean(read_back[int(len(row_means) * 2 / 3):])) if read_back.size else 0.0
    brightness = upper / overall if overall > _EPS else 0.0

    return {
        "peak_band": "place %d" % peak_row,
        "peak_place": int(peak_row),
        "peak_place_hz": float(PLACE_FREQS[peak_row]),
        "motion": float(motion),
        "brightness": float(brightness),
        "image": read_back,
    }


def read_frames(frames, mode_weights=None, duration_seconds=None):
    """Produce the composed plain language reading of frames."""
    aggregate = aggregate_frames(frames, duration_seconds=duration_seconds)
    image = image_reading(frames)
    strongest = _strongest_modes(mode_weights)

    if aggregate["silent"]:
        sentence = (
            "Silence: no sound above the detection threshold was "
            "heard in %.2f seconds." % aggregate["duration"]
        )
        return {
            "sentence": sentence,
            "summary": sentence,
            "silent": True,
            "tonal": False,
            "pitched": False,
            "note": None,
            "cents": None,
            "pitch_hz": 0.0,
            "f0": 0.0,
            "loudness": aggregate["loudness"],
            "loud": aggregate["loudness"],
            "brightness": 0.0,
            "energy_center_hz": 0.0,
            "region": "silent",
            "onset_rate": 0.0,
            "pitch_trend": "steady",
            "loud_trend": "steady",
            "strongest_modes": strongest,
            "descriptor": aggregate["descriptor"],
            "duration": aggregate["duration"],
            "aggregate": aggregate,
            "image_reading": {
                "peak_band": image["peak_band"],
                "peak_place_hz": image["peak_place_hz"],
                "motion": image["motion"],
                "brightness": image["brightness"],
            },
        }

    note_name = None
    cents = None
    if aggregate["pitched"] and aggregate["median_pitch"] > 0:
        nearest = nearest_note(aggregate["median_pitch"])
        note_name = nearest["name"]
        cents = float(nearest["cents"])

    region = _region_name(aggregate["energy_center_hz"])
    if note_name is not None:
        pitch_text = "pitched at %s, %.1f Hz, %+0.1f cents from exact" % (
            note_name,
            aggregate["median_pitch"],
            cents,
        )
    elif aggregate["median_pitch"] > 0:
        pitch_text = "unpitched, median pitch %.1f Hz" % aggregate["median_pitch"]
    else:
        pitch_text = "unpitched"

    sentence = (
        "Heard a %s sound lasting %.2f seconds: %s; loudness "
        "%.1f dB; brightness %.0f Hz; energy centered at %.0f Hz "
        "in the %s region; onset rate %.2f per second; pitch "
        "trend %s; loudness trend %s."
        % (
            "tonal" if aggregate["tonal"] else "noisy",
            aggregate["duration"],
            pitch_text,
            aggregate["loudness"],
            aggregate["brightness"],
            aggregate["energy_center_hz"],
            region,
            aggregate["onset_rate"],
            aggregate["pitch_trend"],
            aggregate["loud_trend"],
        )
    )
    if strongest:
        sentence += " Strongest plate modes: %s." % ", ".join(
            "mode %d (weight %.3f)" % (m["mode"], m["weight"]) for m in strongest
        )

    return {
        "sentence": sentence,
        "summary": sentence,
        "silent": False,
        "tonal": aggregate["tonal"],
        "pitched": aggregate["pitched"],
        "note": note_name,
        "cents": cents,
        "pitch_hz": float(aggregate["median_pitch"]),
        "f0": float(aggregate["median_pitch"]),
        "loudness": float(aggregate["loudness"]),
        "loud": float(aggregate["loudness"]),
        "brightness": float(aggregate["brightness"]),
        "energy_center_hz": float(aggregate["energy_center_hz"]),
        "region": region,
        "onset_rate": float(aggregate["onset_rate"]),
        "pitch_trend": aggregate["pitch_trend"],
        "loud_trend": aggregate["loud_trend"],
        "strongest_modes": strongest,
        "descriptor": aggregate["descriptor"],
        "duration": float(aggregate["duration"]),
        "aggregate": aggregate,
        "image_reading": {
            "peak_band": image["peak_band"],
            "peak_place_hz": image["peak_place_hz"],
            "motion": image["motion"],
            "brightness": image["brightness"],
        },
    }
