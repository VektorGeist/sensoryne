"""stage2/apps/music/comparison.py - Sensoryne's judgment of the render
against the written work.

The comparison works from the hearing's own analysis of the render:
the same Cochlea frame analysis that perceive performs, plus the
harmonic scoring the hearing uses, set against
structure_record(artifact).  It never works from the score alone
and never from file metadata.  Dynamics are compared per note,
anchored at each note's attack, using the attack loudness the
hearing measured.

Each finding is an object with kind, where, written, heard, and
statement.  Tolerances are stated here and in the reliability
block; every finding carries the figures it was judged on.
"""
import bisect
import copy
import math
import os

import numpy as np

from .app import structure_record
from .hearing.cochlea import Cochlea, load_wav
from .hearing.harmonic import score_midi
from .instruments import nearest_note

__all__ = [
    "PITCH_TOLERANCE_CENTS",
    "NOTE_HEARD_RATIO_THRESHOLD",
    "NOTE_HEARD_FUNDAMENTAL_PEAK_RATIO",
    "NOTE_HEARD_FUNDAMENTAL_BACKGROUND_RATIO",
    "SILENCE_LOUDNESS_DB",
    "DYNAMICS_THRESHOLD_DB",
    "DYNAMICS_CLEAN_SPREAD_DB",
    "DYNAMICS_OFFSET_DB",
    "DYNAMICS_MAPPING",
    "compute_comparison",
    "build_comparison",
]

PITCH_TOLERANCE_CENTS = 50.0
NOTE_HEARD_RATIO_THRESHOLD = 0.15
# A written note is counted heard when a pitched component is
# present at its own fundamental in its window: the magnitude
# at the sounding fundamental must stand at least this far
# above the neighbouring semitone fundamentals (the median of
# the magnitudes at +/-1 and +/-2 semitones) and at least the
# background ratio above the median fundamental magnitude
# over the scored pitch range in the same window.  The
# harmonic ratio to the strongest pitch in the window is
# reported for every note but is no longer the heard
# judgment: in a dense polyphonic window the strongest
# harmonic score belongs to a low note whose harmonic stack
# spans the window, so a genuinely sounding upper note can
# sit far below 0.15 of that stack while its own fundamental
# stands out cleanly from its neighbours.
NOTE_HEARD_FUNDAMENTAL_PEAK_RATIO = 2.0
NOTE_HEARD_FUNDAMENTAL_BACKGROUND_RATIO = 3.0
SILENCE_LOUDNESS_DB = -70.0
DYNAMICS_THRESHOLD_DB = 4.0
DYNAMICS_CLEAN_SPREAD_DB = 0.0
DYNAMICS_ATTACK_WINDOW_SECONDS = 0.25
# Retained for compatibility; the note-anchored dynamics no
# longer uses a fixed offset (see DYNAMICS_MAPPING): the
# render's own median adjusted level is the expectation.
DYNAMICS_OFFSET_DB = -10.0
DYNAMICS_MAPPING = {
    "formula": "adjusted_db = attack_loudness_db - 20*log10(effective_velocity); expected_db = median adjusted_db over all notes; attack_loudness_db = RMS loudness in dB over min(0.25, note duration) seconds from note onset",
    "offset_db": DYNAMICS_OFFSET_DB,
    "effective_velocity": "velocity * part gain",
    "threshold_db": DYNAMICS_THRESHOLD_DB,
    "clean_spread_db": DYNAMICS_CLEAN_SPREAD_DB,
    "attack_window_seconds": DYNAMICS_ATTACK_WINDOW_SECONDS,
    "notes": "every written note in every part; attack loudness from the hearing's loudness machinery per note over the window starting at the note onset",
    "calibration": "median normalization: on a faithful render the adjusted level clusters tightly because every note is measured at the same phase of its envelope; clean spread measured on faithful renders of the suite's clean cases",
}

_EPS = 1e-12


def _j(v):
    if isinstance(v, dict):
        return {str(k): _j(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, (str, int, float)):
        return v
    try:
        import numpy as np

        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
    except Exception:
        pass
    return str(v)


def _median(values):
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _is_vocal(artifact):
    return isinstance(artifact, dict) and artifact.get("kind") == "vocal"


def _finding(kind, where, written, heard, statement):
    return {
        "kind": kind,
        "where": where,
        "written": _j(written),
        "heard": _j(heard),
        "statement": statement,
    }


def _summary(findings):
    counts = {}
    for f in findings:
        counts[f["kind"]] = counts.get(f["kind"], 0) + 1
    total = len(findings)
    if total == 0:
        line = "No findings: the render matches the written work within the stated tolerances."
    else:
        line = "%d finding(s): %s." % (
            total,
            ", ".join("%s=%d" % (k, counts[k]) for k in sorted(counts)),
        )
    return {"counts": counts, "total": total, "line": line}


def _vocal_comparison(artifact, render_record, evidence):
    findings = []
    lyrics = artifact.get("lyrics", []) if isinstance(artifact, dict) else []
    vocal = None
    if isinstance(evidence, dict) and isinstance(evidence.get("measurements"), dict):
        vocal = evidence["measurements"].get("vocal")
    lines = vocal.get("lines") if isinstance(vocal, dict) else None
    if lines is None:
        for idx, line in enumerate(lyrics):
            findings.append(
                _finding(
                    "line_not_detected",
                    "line %d" % (idx + 1),
                    {"line": idx + 1, "syllables": list(line), "stated_syllable_count": len(line)},
                    {"detected_phrase": False, "syllable_events_detected": 0, "median_pitch_hz": None},
                    "Lyric line %d was not detected as a phrase in the render." % (idx + 1),
                )
            )
    else:
        for idx, line in enumerate(lyrics):
            info = lines[idx] if idx < len(lines) else None
            if info is None or not info.get("detected_phrase"):
                findings.append(
                    _finding(
                        "line_not_detected",
                        "line %d" % (idx + 1),
                        {
                            "line": idx + 1,
                            "syllables": list(line),
                            "stated_syllable_count": len(line),
                        },
                        {"detected_phrase": False, "syllable_events_detected": 0, "median_pitch_hz": None},
                        "Lyric line %d was not detected as a phrase in the render." % (idx + 1),
                    )
                )
                continue
            detected = info.get("syllable_events_detected")
            stated = info.get("stated_syllable_count", len(line))
            if detected != stated:
                findings.append(
                    _finding(
                        "syllable_count_off",
                        "line %d" % (idx + 1),
                        {"line": idx + 1, "stated_syllable_count": stated, "syllables": list(line)},
                        {"syllable_events_detected": detected},
                        "Lyric line %d has %s syllable events heard against %s stated syllables."
                        % (idx + 1, detected, stated),
                    )
                )
            if not info.get("median_pitch_hz"):
                findings.append(
                    _finding(
                        "pitch_absent",
                        "line %d" % (idx + 1),
                        {"line": idx + 1, "syllables": list(line), "stated_syllable_count": stated},
                        {"median_pitch_hz": None, "note": info.get("note")},
                        "Lyric line %d has no pitch heard in the render." % (idx + 1),
                    )
                )
    return {
        "findings": findings,
        "summary": _summary(findings),
        "per_part": [],
        "tolerances": {
            "pitch_tolerance_cents": PITCH_TOLERANCE_CENTS,
            "note_heard_harmonic_ratio_threshold": NOTE_HEARD_RATIO_THRESHOLD,
            "note_heard_fundamental_peak_ratio": NOTE_HEARD_FUNDAMENTAL_PEAK_RATIO,
            "note_heard_fundamental_background_ratio": NOTE_HEARD_FUNDAMENTAL_BACKGROUND_RATIO,
            "silence_loudness_db": SILENCE_LOUDNESS_DB,
            "dynamics_threshold_db": DYNAMICS_THRESHOLD_DB,
            "dynamics_clean_spread_db": DYNAMICS_CLEAN_SPREAD_DB,
            "dynamics_mapping": copy.deepcopy(DYNAMICS_MAPPING),
        },
    }


def _seg_loudness(seg):
    """Segment loudness from the model pack (reading) or stored pack."""
    if not isinstance(seg, dict):
        return None
    if seg.get("loudness") is not None:
        return seg.get("loudness")
    reading = seg.get("reading")
    if isinstance(reading, dict):
        return reading.get("loudness")
    return None


def _seg_pitch_hz(seg):
    """Segment pitch from the model pack (reading) or stored pack."""
    if not isinstance(seg, dict):
        return None
    if seg.get("pitch_hz") is not None:
        return seg.get("pitch_hz")
    reading = seg.get("reading")
    if isinstance(reading, dict):
        return reading.get("pitch_hz")
    return None


def _token_segments(evidence):
    if not isinstance(evidence, dict):
        return []
    pack = None
    meas = evidence.get("measurements")
    if isinstance(meas, dict) and isinstance(meas.get("token_pack"), dict):
        pack = meas["token_pack"]
    if pack is None and isinstance(evidence.get("token_pack"), dict):
        pack = evidence["token_pack"]
    if pack is None:
        full = evidence.get("full")
        if isinstance(full, dict):
            if isinstance(full.get("token_pack"), dict):
                pack = full["token_pack"]
            elif isinstance(full.get("measurements"), dict) and isinstance(
                full["measurements"].get("token_pack"), dict
            ):
                pack = full["measurements"]["token_pack"]
    if not isinstance(pack, dict):
        return []
    segs = pack.get("segments_data")
    return segs if isinstance(segs, list) else []


def _midi_freq(midi):
    return 440.0 * (2.0 ** ((int(midi) - 69) / 12.0))


def _fundamental_magnitudes(samples, sr, start, end):
    """Fundamental magnitude at each scored midi in a window.

    The magnitude spectrum is computed the same way the
    harmonic scoring computes its own (Hann window, zero
    padding to four times the segment length), and each
    midi's fundamental is read at its exact frequency.
    Returns ((midi, magnitude) pairs, background median) or
    (None, None) when the window holds no samples.
    """
    if samples is None or sr is None:
        return None, None
    sa = int(max(0, min(len(samples), round(float(start) * sr))))
    sb = int(max(sa, min(len(samples), round(float(end) * sr))))
    if sb <= sa:
        return {}, None
    try:
        x = np.asarray(samples[sa:sb], dtype=np.float64)
    except Exception:
        return None, None
    if x.ndim > 1:
        x = x[:, 0]
    if x.size and np.max(np.abs(x)) > 1.5:
        x = x / 32768.0
    n = 1
    while n < len(x) * 4:
        n *= 2
    win = np.hanning(len(x)) if len(x) > 1 else np.ones_like(x)
    try:
        spec = np.abs(np.fft.rfft(x * win, n=n))
    except Exception:
        return None, None
    bin_hz = float(sr) / float(n)

    def at(freq):
        if freq <= 0:
            return 0.0
        i = int(round(freq / bin_hz))
        return float(spec[i]) if 0 <= i < len(spec) else 0.0

    magnitudes = {}
    for midi in range(21, 109):
        freq = _midi_freq(midi)
        if freq >= float(sr) / 2.0:
            continue
        magnitudes[midi] = at(freq)
    background = _median(list(magnitudes.values())) if magnitudes else None
    return magnitudes, background


def compute_comparison(artifact, render_record, evidence=None):
    if _is_vocal(artifact):
        return _vocal_comparison(artifact, render_record, evidence)
    structure = structure_record(artifact)
    parts_structure = structure.get("parts", []) if isinstance(structure, dict) else []
    # Original parts for velocity/gain, by name.
    orig_parts = {}
    orig_order = []
    if isinstance(artifact, dict):
        for p in artifact.get("parts", []) or []:
            if isinstance(p, dict):
                orig_parts[p.get("name")] = p
                orig_order.append(p.get("name"))

    render_path = None
    if isinstance(render_record, dict):
        render_path = render_record.get("path")
    frames = []
    frame_times = []
    samples = None
    sr = None
    if render_path and isinstance(render_path, str) and os.path.isfile(render_path):
        try:
            samples, sr = load_wav(render_path)
            frames = Cochlea(sr).analyze(samples)
            frame_times = [float(f.get("time", 0.0)) for f in frames]
        except Exception:
            frames = []
            frame_times = []
            samples = None
            sr = None

    # All written notes flattened for active-count and overlap checks.
    all_notes = []  # (start, end, midi)
    for part in parts_structure:
        if not isinstance(part, dict):
            continue
        for n in part.get("notes", []) or []:
            if not isinstance(n, dict):
                continue
            st = float(n.get("start", 0.0))
            dur = n.get("dur")
            en = st + (float(dur) if dur is not None else 0.5)
            all_notes.append((st, en, n.get("sounding_midi")))

    def active_count(start, end):
        return sum(1 for (s, e, _m) in all_notes if s < end and e > start)

    def frames_in(start, end):
        if not frame_times:
            return []
        lo = bisect.bisect_left(frame_times, start)
        hi = bisect.bisect_left(frame_times, end)
        return frames[lo:hi]

    score_cache = {}

    def harmonic_scores(start, end):
        key = (round(float(start), 4), round(float(end), 4))
        if key in score_cache:
            return score_cache[key]
        if samples is None or sr is None:
            score_cache[key] = None
            return None
        sa = int(max(0, min(len(samples), round(float(start) * sr))))
        sb = int(max(sa, min(len(samples), round(float(end) * sr))))
        if sb <= sa:
            score_cache[key] = {}
            return {}
        try:
            sc = score_midi(samples[sa:sb], sr)
        except Exception:
            sc = {}
        score_cache[key] = sc
        return sc

    fund_cache = {}

    def fundamental_info(start, end, sounding_midi):
        """Per-note fundamental evidence in [start, end).

        The magnitude at the sounding fundamental, its ratio
        to the median neighbouring-semitone fundamental
        magnitude, its ratio to the window's median
        fundamental magnitude, and whether the note is heard
        by those two ratios.  Returns None when there are no
        samples to judge on.
        """
        key = (round(float(start), 4), round(float(end), 4))
        if key not in fund_cache:
            fund_cache[key] = _fundamental_magnitudes(samples, sr, start, end)
        magnitudes, background = fund_cache[key]
        if magnitudes is None:
            return None
        if sounding_midi is None:
            return {
                "fundamental": None,
                "fundamental_peak_ratio": None,
                "fundamental_background_ratio": None,
                "background": background,
                "heard": False,
            }
        try:
            target_midi = int(sounding_midi)
        except Exception:
            return None
        fundamental = float(magnitudes.get(target_midi, 0.0))
        neighbours = [
            float(magnitudes.get(target_midi + d, 0.0))
            for d in (-2, -1, 1, 2)
            if (target_midi + d) in magnitudes
        ]
        # For the neighbour level use the lower of the two
        # middle neighbour magnitudes (for an even count),
        # not their mean: a single neighbouring pitch that
        # was actually played (a semitone substitution
        # next door) must not lift the neighbour level to
        # that pitch's magnitude and mask a missing
        # fundamental, and a quiet fundamental beside a
        # loud substituted neighbour is still judged
        # against the quieter neighbours.  With one or
        # three neighbours this is the ordinary median.
        if neighbours:
            _srt = sorted(neighbours)
            neighbour_median = _srt[(len(_srt) - 1) // 2]
        else:
            neighbour_median = None
        if fundamental > 0 and neighbour_median is not None and neighbour_median > 0:
            peak_ratio = fundamental / neighbour_median
        elif fundamental > 0:
            peak_ratio = float("inf")
        else:
            peak_ratio = 0.0
        if fundamental > 0 and background is not None and background > 0:
            background_ratio = fundamental / background
        elif fundamental > 0:
            background_ratio = float("inf")
        else:
            background_ratio = 0.0
        # A substituted neighbour (a pitch actually
        # played a semitone or two away) is itself a
        # strong fundamental candidate in this window:
        # when the strongest neighbour fundamental is
        # far stronger than the written fundamental,
        # the written pitch was not played there even
        # if its own small residual stands above the
        # quiet neighbours.  The written note is heard
        # only when its fundamental is also comparable
        # to that strongest neighbour pitch.
        strongest_neighbour = max(neighbours) if neighbours else 0.0
        neighbour_dominance = (
            (fundamental / strongest_neighbour)
            if (strongest_neighbour and strongest_neighbour > 0 and fundamental > 0)
            else (0.0 if fundamental == 0 else float("inf"))
        )
        # The dominance line is deliberately loose
        # (one quarter of the strongest neighbour):
        # its only job is to catch residuals that are
        # orders of magnitude below a substituted
        # neighbour actually played in the window,
        # while a genuinely sounding note that happens
        # to sit a semitone from a louder sounding
        # neighbour (as in a close chord) still counts
        # as heard on its own peak and background
        # ratios.
        heard = bool(
            fundamental > 0
            and peak_ratio >= NOTE_HEARD_FUNDAMENTAL_PEAK_RATIO
            and background_ratio >= NOTE_HEARD_FUNDAMENTAL_BACKGROUND_RATIO
            and neighbour_dominance >= 0.25
        )
        return {
            "fundamental": float(fundamental),
            "fundamental_peak_ratio": float(peak_ratio),
            "fundamental_background_ratio": float(background_ratio),
            "background": background,
            "neighbour_dominance": float(neighbour_dominance),
            "heard": heard,
        }

    findings = []
    per_part = []
    # For section_silent: per part list of (start, end, silent, note_info)
    silent_by_part = {}

    for part in parts_structure:
        if not isinstance(part, dict):
            continue
        part_name = part.get("part")
        notes = part.get("notes", []) or []
        heard_count = 0
        silent_entries = []
        for n in notes:
            if not isinstance(n, dict):
                continue
            start = float(n.get("start", 0.0))
            dur = n.get("dur")
            end = start + (float(dur) if dur is not None else 0.5)
            sounding_midi = n.get("sounding_midi")
            sounding_name = n.get("sounding_name")
            sounding_freq = n.get("sounding_freq_hz")
            wf = frames_in(start, end)
            loud_vals = [float(f.get("loudness", f.get("loud", -120.0))) for f in wf]
            med_loud = _median(loud_vals) if loud_vals else None
            wf_first = frames_in(start, start + min((float(dur) if dur is not None else 0.5), 0.5))
            f0_vals = [float(f.get("f0", 0.0)) for f in wf_first if float(f.get("f0", 0.0)) > 0]
            med_f0 = _median(f0_vals) if f0_vals else None
            harm_end = start + min((float(dur) if dur is not None else 0.5), 0.5)
            scores = harmonic_scores(start, harm_end)
            best = max(scores.values()) if scores else 0.0
            target = float(scores.get(sounding_midi, 0.0)) if scores and sounding_midi is not None else 0.0
            ratio = (target / best) if best and best > 0 else 0.0
            # The pitch dominating this note's window, by the
            # harmonic scoring over its first 0.5 seconds: the
            # midi with the strongest harmonic energy there.
            dominant_midi = None
            dominant_note = None
            dominant_freq = None
            if scores and best and best > 0:
                dominant_midi = max(scores.items(), key=lambda kv: kv[1])[0]
                try:
                    dominant_freq = 440.0 * (2.0 ** ((int(dominant_midi) - 69) / 12.0))
                    dominant_note = nearest_note(float(dominant_freq)).get("name")
                except Exception:
                    dominant_note = None
                    dominant_freq = None
            if dominant_midi is not None:
                dominant_clause = (
                    " The strongest pitch heard in its window was midi %s (%s, %.1f Hz)."
                    % (dominant_midi, dominant_note, dominant_freq)
                )
            else:
                dominant_clause = ""
            fund = fundamental_info(start, harm_end, sounding_midi)
            fund_heard = bool(fund and fund.get("heard"))
            silent = True
            if med_loud is not None and med_loud > SILENCE_LOUDNESS_DB and best > 0:
                silent = False
            if med_loud is not None and med_loud > SILENCE_LOUDNESS_DB and scores is None:
                # No harmonic scores available (no samples): fall back to loudness.
                silent = False
            # If there are no frames at all, silent stays True.
            silent_entries.append((start, end, silent, n))
            where_str = "part %s start %s note %s midi %s" % (
                part_name,
                start,
                sounding_name,
                sounding_midi,
            )
            written = {
                "part": part_name,
                "start": start,
                "end": end,
                "midi": n.get("midi"),
                "written_name": n.get("written_name"),
                "sounding_midi": sounding_midi,
                "sounding_name": sounding_name,
                "sounding_freq_hz": sounding_freq,
            }
            cents_diff = None
            heard_note = None
            heard_midi = None
            if med_f0 and sounding_freq and float(sounding_freq) > 0:
                cents_diff = 1200.0 * math.log2(float(med_f0) / float(sounding_freq))
                try:
                    nn = nearest_note(float(med_f0))
                    heard_note = nn.get("name")
                    heard_midi = nn.get("midi")
                except Exception:
                    heard_note = None
                    heard_midi = None
            heard = {
                "median_loudness_db": med_loud,
                "median_pitch_hz": med_f0,
                "heard_note": heard_note,
                "heard_midi": heard_midi,
                "cents_difference": cents_diff,
                "harmonic_ratio": ratio,
                "harmonic_best": float(best) if best else 0.0,
                "harmonic_target": float(target),
                "harmonic_best_midi": dominant_midi,
                "dominant_midi": dominant_midi,
                "dominant_note": dominant_note,
                "dominant_freq_hz": dominant_freq,
                "fundamental": fund.get("fundamental") if fund else None,
                "fundamental_peak_ratio": fund.get("fundamental_peak_ratio") if fund else None,
                "fundamental_background_ratio": fund.get("fundamental_background_ratio") if fund else None,
                "fundamental_background": fund.get("background") if fund else None,
                "fundamental_peak_threshold": NOTE_HEARD_FUNDAMENTAL_PEAK_RATIO,
                "fundamental_background_threshold": NOTE_HEARD_FUNDAMENTAL_BACKGROUND_RATIO,
                "pitch_tolerance_cents": PITCH_TOLERANCE_CENTS,
                "note_heard_ratio_threshold": NOTE_HEARD_RATIO_THRESHOLD,
            }
            if silent:
                findings.append(
                    _finding(
                        "note_not_heard",
                        where_str,
                        written,
                        heard,
                        "Written note %s midi %s in part %s at %s seconds was not heard: "
                        "no pitched energy at its sounding pitch was heard in its window.%s"
                        % (sounding_name, sounding_midi, part_name, start, dominant_clause),
                    )
                )
                continue
            # The strongest harmonic pitch in the window, when
            # it is not the written pitch and no component is
            # heard at the written fundamental, is the pitch
            # that was actually played there: a substituted
            # pitch is named with the pitch heard in its place.
            substituted_note = None
            substituted_midi = None
            if (
                not fund_heard
                and scores is not None
                and best > 0
                and dominant_midi is not None
                and sounding_midi is not None
                and int(dominant_midi) != int(sounding_midi)
            ):
                try:
                    sub_freq = float(dominant_freq)
                    nn_sub = nearest_note(sub_freq)
                    substituted_note = nn_sub.get("name")
                    substituted_midi = nn_sub.get("midi")
                except Exception:
                    substituted_note = dominant_note
                    substituted_midi = dominant_midi
            # Pitch check only where a single written note is active in
            # the window, so the dominant pitch can be attributed.
            act = active_count(start, end)
            if act == 1 and cents_diff is not None and abs(cents_diff) > PITCH_TOLERANCE_CENTS:
                findings.append(
                    _finding(
                        "pitch_off",
                        where_str,
                        written,
                        dict(
                            heard,
                            heard_note=heard_note,
                            heard_midi=heard_midi,
                            difference_cents=float(cents_diff),
                        ),
                        "Written note %s midi %s in part %s at %s seconds was heard as %s midi %s, "
                        "off by %s cents, more than the %s cent tolerance."
                        % (
                            sounding_name,
                            sounding_midi,
                            part_name,
                            start,
                            heard_note,
                            heard_midi,
                            round(float(cents_diff), 1),
                            PITCH_TOLERANCE_CENTS,
                        ),
                    )
                )
                heard_count += 1
                continue
            if scores is not None and best > 0 and not fund_heard:
                sub_clause = ""
                if substituted_midi is not None:
                    sub_clause = (
                        " Pitch %s midi %s was heard in its window instead."
                        % (substituted_note, substituted_midi)
                    )
                findings.append(
                    _finding(
                        "note_not_heard",
                        where_str,
                        written,
                        dict(
                            heard,
                            substituted_note=substituted_note,
                            substituted_midi=substituted_midi,
                        ),
                        "Written note %s midi %s in part %s at %s seconds was not heard: "
                        "no pitched component at its sounding fundamental was heard in its "
                        "window: fundamental magnitude %s, %.3f of the neighbouring "
                        "semitones (threshold %.2f), %.3f of the window background "
                        "(threshold %.2f); harmonic energy at its sounding pitch is %.3f "
                        "of the strongest pitch in the window.%s%s"
                        % (
                            sounding_name,
                            sounding_midi,
                            part_name,
                            start,
                            ("%.3g" % fund.get("fundamental")) if fund and fund.get("fundamental") is not None else None,
                            float(fund.get("fundamental_peak_ratio") or 0.0) if fund else 0.0,
                            NOTE_HEARD_FUNDAMENTAL_PEAK_RATIO,
                            float(fund.get("fundamental_background_ratio") or 0.0) if fund else 0.0,
                            NOTE_HEARD_FUNDAMENTAL_BACKGROUND_RATIO,
                            ratio,
                            dominant_clause,
                            sub_clause,
                        ),
                    )
                )
                continue
            if scores is not None and best == 0:
                findings.append(
                    _finding(
                        "note_not_heard",
                        where_str,
                        written,
                        heard,
                        "Written note %s midi %s in part %s at %s seconds was not heard: "
                        "no pitched energy was heard in its window.%s"
                        % (sounding_name, sounding_midi, part_name, start, dominant_clause),
                    )
                )
                continue
            if fund is not None and not fund_heard and scores is None:
                # Samples existed for the fundamental but the
                # harmonic scoring did not run; the fundamental
                # evidence alone does not count the note heard.
                findings.append(
                    _finding(
                        "note_not_heard",
                        where_str,
                        written,
                        heard,
                        "Written note %s midi %s in part %s at %s seconds was not heard: "
                        "no pitched component at its sounding fundamental was heard in its "
                        "window.%s"
                        % (sounding_name, sounding_midi, part_name, start, dominant_clause),
                    )
                )
                continue
            heard_count += 1
        silent_by_part[part_name] = silent_entries
        per_part.append(
            {
                "part": part_name,
                "notes_written": len(notes),
                "notes_heard": heard_count,
                "notes_not_heard": len(notes) - heard_count,
            }
        )

    # section_silent: maximal runs of at least two consecutive silent
    # notes in a part.
    for part in parts_structure:
        if not isinstance(part, dict):
            continue
        part_name = part.get("part")
        entries = silent_by_part.get(part_name, [])
        # entries are already in written order; sort by start to be safe.
        entries_sorted = sorted(entries, key=lambda e: e[0])
        run = []
        runs = []
        for entry in entries_sorted:
            if entry[2]:
                run.append(entry)
            else:
                if run:
                    runs.append(run)
                    run = []
        if run:
            runs.append(run)
        for run in runs:
            if len(run) < 2:
                continue
            run_start = run[0][0]
            run_end = run[-1][1]
            findings.append(
                _finding(
                    "section_silent",
                    "part %s %s-%s seconds" % (part_name, run_start, run_end),
                    {
                        "part": part_name,
                        "start": run_start,
                        "end": run_end,
                        "note_count": len(run),
                        "notes": [
                            {
                                "start": e[3].get("start"),
                                "sounding_midi": e[3].get("sounding_midi"),
                                "sounding_name": e[3].get("sounding_name"),
                            }
                            for e in run
                        ],
                    },
                    {"median_loudness_db": None, "heard": "silence"},
                    "Part %s from %s to %s seconds (%d written notes) was not sounded: "
                    "no heard sound was found for that written span."
                    % (part_name, run_start, run_end, len(run)),
                )
            )

    segments = _token_segments(evidence)
    # dynamics_off: note-anchored. For every written note the
    # heard loudness is measured at its attack as the onset
    # increment: the mean frame power over the attack window
    # starting at the onset minus the mean frame power over
    # the equal window ending at the onset (the sounding-
    # through background), floored at a small positive
    # fraction of the after-window power. At the start of
    # the render, where no before-window exists, the attack
    # power is the after-window power itself. Routing: a
    # note whose onset is shared with another part's attack
    # AND whose before window's power is within 3 dB of its
    # after window's power is instead measured by the
    # per-pitch harmonic score at its own midi over the
    # attack window, and those notes form their own pool
    # judged against its own median. Where exactly one note
    # attacks at an onset the increment belongs to that note
    # and it enters the per-note findings; where several
    # notes attack at the same onset the increment is shared
    # for the per-part medians only, split by the attacking
    # notes' written effective-velocity-squared shares, and
    # those (non-routed) notes do not enter the per-note
    # findings. The note's adjusted level is that loudness
    # minus 20*log10(effective velocity), where effective
    # velocity is velocity times part gain. The render
    # level for a pool is the median adjusted level over
    # that pool's notes; a note, and a part (by its median
    # adjusted level over its notes in that pool), departing
    # from its pool median by more than
    # DYNAMICS_THRESHOLD_DB is a dynamics_off finding.
    dynamics_median = None
    dynamics_observed_spread = None
    dynamics_main_median = None
    dynamics_main_observed_spread = None
    dynamics_routed_median = None
    dynamics_routed_observed_spread = None
    if isinstance(artifact, dict):
        # Every written note flattened, for attack grouping.
        flat_notes = []
        for pname in orig_order:
            porig = orig_parts.get(pname) or {}
            try:
                pgain = float(porig.get("gain", 1.0)) if isinstance(porig, dict) else 1.0
            except Exception:
                pgain = 1.0
            for on in (porig.get("notes", []) if isinstance(porig, dict) else []) or []:
                if not isinstance(on, dict):
                    continue
                try:
                    st = float(on.get("start", 0.0))
                    vel = float(on.get("vel", 0.0))
                except Exception:
                    continue
                dur = on.get("dur")
                try:
                    dur_f = float(dur) if dur is not None else 0.5
                except Exception:
                    dur_f = 0.5
                flat_notes.append(
                    {
                        "part": pname,
                        "start": st,
                        "end": st + dur_f,
                        "vel": vel,
                        "gain": pgain,
                        "eff": vel * pgain,
                        "midi": on.get("midi"),
                    }
                )
        attacks_by_start = {}
        for other in flat_notes:
            if other.get("eff", 0.0) <= 0:
                continue
            attacks_by_start.setdefault(other["start"], []).append(other)
        records = []
        per_part_records = {}
        for part_name in orig_order:
            orig = orig_parts.get(part_name) or {}
            try:
                gain = float(orig.get("gain", 1.0)) if isinstance(orig, dict) else 1.0
            except Exception:
                gain = 1.0
            per_part_records[part_name] = []
            orig_notes = orig.get("notes", []) if isinstance(orig, dict) else []
            for on in orig_notes or []:
                if not isinstance(on, dict):
                    continue
                try:
                    start = float(on.get("start", 0.0))
                    vel = float(on.get("vel", 0.0))
                except Exception:
                    continue
                eff = vel * gain
                if eff <= 0:
                    continue
                dur = on.get("dur")
                try:
                    dur_f = float(dur) if dur is not None else 0.5
                except Exception:
                    dur_f = 0.5
                window = min(DYNAMICS_ATTACK_WINDOW_SECONDS, dur_f) if dur_f > 0 else DYNAMICS_ATTACK_WINDOW_SECONDS
                if window <= 0:
                    continue
                wf = frames_in(start, start + window)
                if not wf:
                    continue
                vals = [float(f.get("loudness", f.get("loud", -120.0))) for f in wf]
                if not vals:
                    continue
                mean_power = sum(10.0 ** (v / 10.0) for v in vals) / len(vals)
                if mean_power <= 0:
                    continue
                wf_before = frames_in(start - window, start)
                mean_before_power = None
                if wf_before:
                    vals_before = [
                        float(f.get("loudness", f.get("loud", -120.0)))
                        for f in wf_before
                    ]
                    if vals_before:
                        mean_before_power = (
                            sum(10.0 ** (v / 10.0) for v in vals_before)
                            / len(vals_before)
                        )
                    else:
                        mean_before_power = 0.0
                attackers = attacks_by_start.get(start, [])
                shared = len(attackers) > 1
                midi = on.get("midi")
                # Routing: shared onset and before power within
                # 3 dB of after power -> per-pitch harmonic score
                # at the note's own midi over the attack window.
                routed = False
                attack_db = None
                harmonic_score = None
                if (
                    shared
                    and mean_before_power is not None
                    and mean_before_power > 0
                    and mean_power > 0
                ):
                    before_ratio_db = 10.0 * math.log10(mean_before_power / mean_power)
                    if abs(before_ratio_db) <= 3.0:
                        scores_r = harmonic_scores(start, start + window)
                        if scores_r and midi is not None:
                            try:
                                score_val = float(scores_r.get(int(midi), 0.0) or 0.0)
                            except Exception:
                                score_val = 0.0
                            if score_val > 0:
                                harmonic_score = score_val
                                attack_db = 20.0 * math.log10(score_val)
                                routed = True
                if not routed:
                    if mean_before_power is None:
                        increment_power = mean_power
                    else:
                        increment_power = mean_power - mean_before_power
                        floor_power = 0.01 * mean_power
                        if increment_power < floor_power:
                            increment_power = floor_power
                    if increment_power <= 0:
                        continue
                    own_power = increment_power
                    if shared:
                        total_attacker_power = sum(
                            a["eff"] * a["eff"] for a in attackers
                        )
                        if total_attacker_power > 0:
                            own_power = (
                                increment_power
                                * (eff * eff)
                                / total_attacker_power
                            )
                    if own_power <= 0:
                        continue
                    attack_db = 10.0 * math.log10(own_power)
                if attack_db is None:
                    continue
                adjusted_db = attack_db - 20.0 * math.log10(eff)
                note_name = None
                try:
                    if midi is not None:
                        freq = 440.0 * (2.0 ** ((int(midi) - 69) / 12.0))
                        note_name = nearest_note(float(freq)).get("name")
                except Exception:
                    note_name = None
                record = {
                    "part": part_name,
                    "start": start,
                    "end": start + dur_f,
                    "midi": midi,
                    "note_name": note_name,
                    "velocity": vel,
                    "gain": gain,
                    "effective_velocity": eff,
                    "attack_loudness_db": float(attack_db),
                    "adjusted_level_db": float(adjusted_db),
                    "shared": bool(shared),
                    "routed": bool(routed),
                    "harmonic_score": harmonic_score,
                }
                records.append(record)
                per_part_records[part_name].append(record)
        if records:
            main_records = [r for r in records if not r.get("routed")]
            routed_records = [r for r in records if r.get("routed")]
            per_part_main = {
                name: [r for r in lst if not r.get("routed")]
                for name, lst in per_part_records.items()
            }
            per_part_routed = {
                name: [r for r in lst if r.get("routed")]
                for name, lst in per_part_records.items()
            }

            def judge_pool(pool_records, pool_per_part, include_shared):
                if not pool_records:
                    return None, None
                pool_median = _median([r["adjusted_level_db"] for r in pool_records])
                if pool_median is None:
                    return None, None
                departures_list = [
                    r["adjusted_level_db"] - pool_median for r in pool_records
                ]
                pool_spread = (
                    max(abs(d) for d in departures_list) if departures_list else None
                )
                for record, departure in zip(pool_records, departures_list):
                    if record.get("shared") and not include_shared:
                        continue
                    if abs(departure) <= DYNAMICS_THRESHOLD_DB:
                        continue
                    where_str = "part %s start %s note %s midi %s" % (
                        record["part"],
                        record["start"],
                        record["note_name"],
                        record["midi"],
                    )
                    written = {
                        "part": record["part"],
                        "scope": "note",
                        "start": record["start"],
                        "end": record["end"],
                        "midi": record["midi"],
                        "note": record["note_name"],
                        "written_name": record["note_name"],
                        "velocity": record["velocity"],
                        "gain": record["gain"],
                        "effective_velocity": record["effective_velocity"],
                        "expected_db": float(pool_median),
                        "adjusted_level_db": record["adjusted_level_db"],
                        "dynamics_mapping": copy.deepcopy(DYNAMICS_MAPPING),
                        "difference_db": float(departure),
                        "departure_db": float(departure),
                        "median_departure_db": float(departure),
                    }
                    heard = {
                        "part": record["part"],
                        "scope": "note",
                        "start": record["start"],
                        "end": record["end"],
                        "midi": record["midi"],
                        "note": record["note_name"],
                        "loudness_db": record["attack_loudness_db"],
                        "attack_loudness_db": record["attack_loudness_db"],
                        "adjusted_level_db": record["adjusted_level_db"],
                        "expected_db": float(pool_median),
                        "median_adjusted_level_db": float(pool_median),
                        "difference_db": float(departure),
                        "departure_db": float(departure),
                        "median_departure_db": float(departure),
                        "threshold_db": DYNAMICS_THRESHOLD_DB,
                    }
                    findings.append(
                        _finding(
                            "dynamics_off",
                            where_str,
                            written,
                            heard,
                            "Part %s note %s midi %s at %s seconds: heard attack "
                            "loudness %.1f dB, adjusted level %.1f dB, departs from "
                            "the render median adjusted level %.1f dB by %.1f dB, "
                            "more than the %.1f dB threshold."
                            % (
                                record["part"],
                                record["note_name"],
                                record["midi"],
                                record["start"],
                                record["attack_loudness_db"],
                                record["adjusted_level_db"],
                                pool_median,
                                departure,
                                DYNAMICS_THRESHOLD_DB,
                            ),
                        )
                    )
                for part_name in orig_order:
                    part_records = pool_per_part.get(part_name, [])
                    if not part_records:
                        continue
                    part_median = _median(
                        [r["adjusted_level_db"] for r in part_records]
                    )
                    if part_median is None:
                        continue
                    part_departure = float(part_median - pool_median)
                    if abs(part_departure) <= DYNAMICS_THRESHOLD_DB:
                        continue
                    where_str = "part %s" % (part_name,)
                    written = {
                        "part": part_name,
                        "scope": "part",
                        "note_count": len(part_records),
                        "median_adjusted_level_db": float(part_median),
                        "expected_db": float(pool_median),
                        "render_median_adjusted_level_db": float(pool_median),
                        "dynamics_mapping": copy.deepcopy(DYNAMICS_MAPPING),
                        "difference_db": float(part_departure),
                        "departure_db": float(part_departure),
                        "median_departure_db": float(part_departure),
                    }
                    heard = {
                        "part": part_name,
                        "scope": "part",
                        "note_count": len(part_records),
                        "loudness_db": float(part_median),
                        "median_adjusted_level_db": float(part_median),
                        "expected_db": float(pool_median),
                        "render_median_adjusted_level_db": float(pool_median),
                        "difference_db": float(part_departure),
                        "departure_db": float(part_departure),
                        "median_departure_db": float(part_departure),
                        "threshold_db": DYNAMICS_THRESHOLD_DB,
                    }
                    findings.append(
                        _finding(
                            "dynamics_off",
                            where_str,
                            written,
                            heard,
                            "Part %s: median adjusted level %.1f dB departs from "
                            "the render median adjusted level %.1f dB by %.1f dB, "
                            "more than the %.1f dB threshold."
                            % (
                                part_name,
                                part_median,
                                pool_median,
                                part_departure,
                                DYNAMICS_THRESHOLD_DB,
                            ),
                        )
                    )
                return pool_median, pool_spread

            dynamics_main_median, dynamics_main_observed_spread = judge_pool(
                main_records, per_part_main, False
            )
            dynamics_routed_median, dynamics_routed_observed_spread = judge_pool(
                routed_records, per_part_routed, True
            )
            dynamics_median = (
                dynamics_main_median
                if dynamics_main_median is not None
                else dynamics_routed_median
            )
            spreads = [
                s
                for s in (
                    dynamics_main_observed_spread,
                    dynamics_routed_observed_spread,
                )
                if s is not None
            ]
            dynamics_observed_spread = max(spreads) if spreads else None

    # sound_not_written: pitched sound in a span with no written note.
    if segments:
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            try:
                seg_start = float(seg.get("start_time"))
                seg_end = float(seg.get("end_time"))
            except Exception:
                continue
            # Frame-window leakage: a frame centred just
            # outside this segment still carries energy
            # from a note that starts or ends just outside
            # it (the analysis window straddles the
            # boundary). A written note that starts or
            # ends within one window of the segment is
            # therefore treated as covering it; the sound
            # will be judged in the neighbouring segment
            # if it truly belongs there.
            leak = 0.08
            overlapping = False
            for (s, e, _m) in all_notes:
                if s < seg_end + leak and e > seg_start - leak:
                    overlapping = True
                    break
            if overlapping:
                continue
            heard_db = _seg_loudness(seg)
            try:
                heard_db_f = float(heard_db) if heard_db is not None else None
            except Exception:
                heard_db_f = None
            if heard_db_f is None or heard_db_f <= SILENCE_LOUDNESS_DB:
                continue
            pitch_hz = _seg_pitch_hz(seg)
            try:
                pitch_f = float(pitch_hz) if pitch_hz is not None else 0.0
            except Exception:
                pitch_f = 0.0
            harm_midi = seg.get("harmonic_midi")
            if pitch_f <= 0 and harm_midi is None:
                continue
            seg_idx = seg.get("segment")
            where_str = "segment %s %s-%s seconds" % (seg_idx, seg_start, seg_end)
            heard_pitch_note = None
            if pitch_f > 0:
                try:
                    heard_pitch_note = nearest_note(pitch_f).get("name")
                except Exception:
                    heard_pitch_note = seg.get("note") or (seg.get("reading") or {}).get("note") if isinstance(seg.get("reading"), dict) else seg.get("note")
            findings.append(
                _finding(
                    "sound_not_written",
                    where_str,
                    {"segment": seg_idx, "start": seg_start, "end": seg_end, "notes_written": 0},
                    {
                        "segment": seg_idx,
                        "start": seg_start,
                        "end": seg_end,
                        "pitch_hz": float(pitch_f) if pitch_f else None,
                        "note": heard_pitch_note or seg.get("note"),
                        "harmonic_midi": harm_midi,
                        "loudness_db": float(heard_db_f),
                    },
                    "Pitched sound %s (%.1f Hz, midi %s) was heard in segment %s (%s-%s seconds) "
                    "where no written note sounds."
                    % (heard_pitch_note, pitch_f, harm_midi, seg_idx, seg_start, seg_end),
                )
            )

    return {
        "findings": findings,
        "summary": _summary(findings),
        "per_part": per_part,
        "tolerances": {
            "pitch_tolerance_cents": PITCH_TOLERANCE_CENTS,
            "note_heard_harmonic_ratio_threshold": NOTE_HEARD_RATIO_THRESHOLD,
            "note_heard_fundamental_peak_ratio": NOTE_HEARD_FUNDAMENTAL_PEAK_RATIO,
            "note_heard_fundamental_background_ratio": NOTE_HEARD_FUNDAMENTAL_BACKGROUND_RATIO,
            "silence_loudness_db": SILENCE_LOUDNESS_DB,
            "dynamics_threshold_db": DYNAMICS_THRESHOLD_DB,
            "dynamics_clean_spread_db": DYNAMICS_CLEAN_SPREAD_DB,
            "dynamics_observed_spread_db": dynamics_observed_spread,
            "dynamics_median_adjusted_level_db": dynamics_median,
            "dynamics_main_pool_median_adjusted_level_db": dynamics_main_median,
            "dynamics_main_pool_observed_spread_db": dynamics_main_observed_spread,
            "dynamics_routed_pool_median_adjusted_level_db": dynamics_routed_median,
            "dynamics_routed_pool_observed_spread_db": dynamics_routed_observed_spread,
            "dynamics_mapping": copy.deepcopy(DYNAMICS_MAPPING),
        },
    }


build_comparison = compute_comparison
