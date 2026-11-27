"""stage2/apps/animation/comparison.py - Sensoryne's judgment of the
delivered animation against the written timeline.

The comparison works from the frame analyzer's own measurements
of the delivered frames: the delivered frame count the analyzer
counted in the rendered frames, the delivered elements it
measured, and the realized per-frame states the renderer
recorded for those frames (the instrument's sampled positions
are taken from those realized tracks), set against the written
timeline the artifact declares: its duration and frame rate,
hence its written frame count, and every written keyframe of
every written element with its time.  It never works from
delivery-fault definitions: a finding names what the delivered
frames show against what was written.

A written keyframe is realized when the delivered frames hold
its element at that keyframe's time within one frame interval
(the instrument's 1 / frame_rate seconds, the spacing of the
delivered frames as the analyzer measures them, with a
0.1 second fallback when no frame rate is known).  The
delivered duration is the delivered frame count at the
delivered frame rate, judged against the written duration
within the same frame interval.

Each finding is an object with kind, where, written, delivered,
and statement.  The kinds are frame_count_off, duration_off,
keyframe_not_realized, and element_not_written.  The
tolerances are stated here and in the reliability block.
"""

import copy

__all__ = [
    "KEYFRAME_TIME_TOLERANCE_SECONDS",
    "FINDING_KINDS",
    "compute_comparison",
    "build_comparison",
]

# Fallback time tolerance, in seconds, used when the written
# scene declares no usable frame rate.  When a frame rate is
# known the tolerance is one frame interval, 1 / frame_rate.
KEYFRAME_TIME_TOLERANCE_SECONDS = 0.1

FINDING_KINDS = (
    "frame_count_off",
    "duration_off",
    "keyframe_not_realized",
    "element_not_written",
)

_EPS = 1e-9


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


def _finding(kind, where, written, delivered, statement):
    return {
        "kind": kind,
        "where": where,
        "written": _j(written),
        "delivered": _j(delivered),
        "statement": statement,
    }


def _summary(findings):
    counts = {}
    for finding in findings:
        counts[finding["kind"]] = counts.get(finding["kind"], 0) + 1
    total = len(findings)
    if total == 0:
        line = (
            "No findings: the delivered animation matches the "
            "written timeline within the stated tolerances."
        )
    else:
        line = "%d finding(s): %s." % (
            total,
            ", ".join(
                "%s=%d" % (kind, counts[kind]) for kind in sorted(counts)
            ),
        )
    return {"counts": counts, "total": total, "line": line}


def _float_or_none(value):
    try:
        if isinstance(value, bool):
            return None
        return float(value)
    except Exception:
        return None


def _written_keyframes(artifact):
    entries = []
    raw = artifact.get("elements", []) if isinstance(artifact, dict) else []
    for el in raw or []:
        if not isinstance(el, dict):
            continue
        eid = el.get("id")
        track = el.get("keyframes") or el.get("track") or []
        for k in track or []:
            if not isinstance(k, dict):
                continue
            time = _float_or_none(k.get("time"))
            if time is None:
                continue
            entries.append(
                {
                    "id": eid,
                    "element": eid,
                    "name": eid,
                    "time": time,
                    "written_time": time,
                    "keyframe_time": time,
                    "x": k.get("x"),
                    "y": k.get("y"),
                    "width": k.get("width"),
                    "height": k.get("height"),
                    "opacity": k.get("opacity"),
                    "easing": k.get("easing"),
                    "keyframe": copy.deepcopy(k),
                }
            )
    return entries


def _measurements(source):
    if isinstance(source, dict):
        measurements = source.get("measurements")
        if isinstance(measurements, dict):
            return measurements
        full = source.get("full")
        if isinstance(full, dict):
            full_measurements = full.get("measurements")
            if isinstance(full_measurements, dict):
                return full_measurements
    return None


def _delivered_frame_count(render_record, evidence):
    for source in (evidence, render_record):
        measurements = _measurements(source)
        if measurements is not None:
            count = measurements.get("frame_count")
            value = _float_or_none(count)
            if value is not None:
                return int(value)
    if isinstance(render_record, dict):
        value = _float_or_none(render_record.get("frame_count"))
        if value is not None:
            return int(value)
        frame_paths = render_record.get("frame_paths") or render_record.get("frames")
        if isinstance(frame_paths, list):
            return len(frame_paths)
    return None


def _delivered_frame_rate(artifact, render_record, evidence):
    for source in (evidence, render_record):
        measurements = _measurements(source)
        if measurements is not None:
            rate = _float_or_none(measurements.get("frame_rate"))
            if rate is not None and rate > 0:
                return rate
    if isinstance(render_record, dict):
        rate = _float_or_none(render_record.get("frame_rate"))
        if rate is not None and rate > 0:
            return rate
        scene = render_record.get("scene")
        if isinstance(scene, dict):
            rate = _float_or_none(scene.get("frame_rate"))
            if rate is not None and rate > 0:
                return rate
    if isinstance(artifact, dict):
        rate = _float_or_none(artifact.get("frame_rate"))
        if rate is not None and rate > 0:
            return rate
    return None


def _delivered_elements(render_record, evidence):
    for source in (evidence, render_record):
        measurements = _measurements(source)
        if measurements is not None:
            elements = measurements.get("elements")
            if isinstance(elements, list) and elements:
                return [e for e in elements if isinstance(e, dict)]
    if isinstance(render_record, dict):
        elements = render_record.get("delivered_elements")
        if isinstance(elements, list) and elements:
            return [e for e in elements if isinstance(e, dict)]
        elements = render_record.get("elements")
        if isinstance(elements, list) and elements:
            return [e for e in elements if isinstance(e, dict)]
    return []


def _delivered_frame_times(render_record, evidence):
    """Element id -> list of delivered frame times, or None.

    None means the element is present across the delivered
    frames as a whole (its delivered track is known, but no
    per-frame times were carried), so presence at a written
    keyframe time is not contradicted by a frame time.
    """
    if isinstance(render_record, dict):
        realized = render_record.get("realized_tracks")
        if isinstance(realized, dict) and realized:
            times_by_id = {}
            for eid, track in realized.items():
                times = []
                if isinstance(track, list):
                    for entry in track:
                        if not isinstance(entry, dict):
                            continue
                        time = _float_or_none(entry.get("time"))
                        if time is not None:
                            times.append(time)
                times_by_id[eid] = sorted(times)
            if times_by_id:
                return times_by_id
    for source in (evidence, render_record):
        measurements = _measurements(source)
        if measurements is None:
            continue
        elements = measurements.get("elements")
        if not isinstance(elements, list) or not elements:
            continue
        times_by_id = {}
        for entry in elements:
            if not isinstance(entry, dict):
                continue
            eid = entry.get("id")
            times = []
            sampled = entry.get("sampled_positions")
            if isinstance(sampled, list):
                for position in sampled:
                    if not isinstance(position, dict):
                        continue
                    time = _float_or_none(position.get("time"))
                    if time is not None:
                        times.append(time)
            times_by_id[eid] = sorted(times)
        if times_by_id:
            return times_by_id
    times_by_id = {}
    for entry in _delivered_elements(render_record, evidence):
        eid = entry.get("id")
        if eid is not None and eid not in times_by_id:
            times_by_id[eid] = None
    return times_by_id


def _delivered_element_ids(render_record, evidence):
    ids = []
    seen = set()
    for entry in _delivered_elements(render_record, evidence):
        eid = entry.get("id")
        if eid is not None and eid not in seen:
            seen.add(eid)
            ids.append(eid)
    if isinstance(render_record, dict):
        realized = render_record.get("realized_tracks")
        if isinstance(realized, dict):
            for eid in realized:
                if eid is not None and eid not in seen:
                    seen.add(eid)
                    ids.append(eid)
    return ids


def compute_comparison(artifact, render_record=None, evidence=None):
    """Compare the delivered frames against the written timeline."""
    if evidence is None and isinstance(render_record, dict):
        if "measurements" in render_record:
            evidence = render_record

    scene = artifact if isinstance(artifact, dict) else {}
    written_duration = _float_or_none(scene.get("duration"))
    written_rate = _float_or_none(scene.get("frame_rate"))
    if written_rate is not None and written_rate <= 0:
        written_rate = None
    written_frames = None
    if written_duration is not None and written_rate is not None:
        written_frames = int(round(written_duration * written_rate))

    delivered_frames = _delivered_frame_count(render_record, evidence)
    delivered_rate = _delivered_frame_rate(artifact, render_record, evidence)
    frame_interval = (
        1.0 / delivered_rate if delivered_rate else KEYFRAME_TIME_TOLERANCE_SECONDS
    )
    tolerance = frame_interval

    delivered_duration = None
    if delivered_frames is not None and delivered_rate:
        delivered_duration = delivered_frames / delivered_rate
    if delivered_duration is None:
        for source in (evidence, render_record):
            measurements = _measurements(source)
            if measurements is not None:
                delivered_duration = _float_or_none(measurements.get("duration"))
                if delivered_duration is not None:
                    break
    if delivered_duration is None and isinstance(render_record, dict):
        delivered_duration = _float_or_none(render_record.get("duration"))
        if delivered_duration is None:
            rec_scene = render_record.get("scene")
            if isinstance(rec_scene, dict):
                delivered_duration = _float_or_none(rec_scene.get("duration"))

    written_keyframes = _written_keyframes(artifact)
    written_ids = []
    written_id_set = set()
    raw_elements = scene.get("elements", []) if isinstance(scene, dict) else []
    for el in raw_elements or []:
        if isinstance(el, dict) and el.get("id") is not None:
            if el.get("id") not in written_id_set:
                written_id_set.add(el.get("id"))
                written_ids.append(el.get("id"))

    frame_times = _delivered_frame_times(render_record, evidence)
    delivered_ids = _delivered_element_ids(render_record, evidence)

    findings = []

    if written_frames is not None and delivered_frames is not None:
        if delivered_frames != written_frames:
            difference = delivered_frames - written_frames
            findings.append(
                _finding(
                    "frame_count_off",
                    "scene",
                    {
                        "frame_count": written_frames,
                        "written_frame_count": written_frames,
                        "duration": written_duration,
                        "written_duration": written_duration,
                        "frame_rate": written_rate,
                        "written_frame_rate": written_rate,
                    },
                    {
                        "frame_count": delivered_frames,
                        "delivered_frame_count": delivered_frames,
                        "duration": delivered_duration,
                        "delivered_duration": delivered_duration,
                        "frame_rate": delivered_rate,
                        "delivered_frame_rate": delivered_rate,
                        "difference_frames": difference,
                    },
                    "Written frame count is %s but the delivered frame "
                    "count is %s: the delivered animation has %s "
                    "frames, %s from the %s frames written."
                    % (
                        written_frames,
                        delivered_frames,
                        delivered_frames,
                        difference,
                        written_frames,
                    ),
                )
            )

    if (
        written_duration is not None
        and delivered_duration is not None
        and abs(delivered_duration - written_duration) > tolerance + _EPS
    ):
        difference = delivered_duration - written_duration
        findings.append(
            _finding(
                "duration_off",
                "scene",
                {
                    "duration": written_duration,
                    "written_duration": written_duration,
                    "frame_count": written_frames,
                    "written_frame_count": written_frames,
                    "frame_rate": written_rate,
                    "written_frame_rate": written_rate,
                },
                {
                    "duration": delivered_duration,
                    "delivered_duration": delivered_duration,
                    "frame_count": delivered_frames,
                    "delivered_frame_count": delivered_frames,
                    "frame_rate": delivered_rate,
                    "delivered_frame_rate": delivered_rate,
                    "difference_seconds": difference,
                    "duration_tolerance_seconds": tolerance,
                },
                "Written duration is %s seconds but the delivered "
                "duration is %s seconds: the delivered animation "
                "runs %s seconds, %s from the %s seconds written."
                % (
                    written_duration,
                    delivered_duration,
                    delivered_duration,
                    difference,
                    written_duration,
                ),
            )
        )

    keyframes_realized = 0
    for entry in written_keyframes:
        eid = entry.get("id")
        time = entry["time"]
        times = frame_times.get(eid) if isinstance(frame_times, dict) else None
        if eid not in frame_times:
            present = False
            nearest = None
        elif times is None:
            present = True
            nearest = time
        elif not times:
            present = False
            nearest = None
        else:
            nearest = min(times, key=lambda t: abs(t - time))
            present = abs(nearest - time) <= tolerance + _EPS
        if present:
            keyframes_realized += 1
            continue
        findings.append(
            _finding(
                "keyframe_not_realized",
                "element %r at %s seconds" % (eid, time),
                {
                    "id": eid,
                    "element": eid,
                    "name": eid,
                    "time": time,
                    "written_time": time,
                    "keyframe_time": time,
                    "keyframe": entry.get("keyframe"),
                    "x": entry.get("x"),
                    "y": entry.get("y"),
                    "width": entry.get("width"),
                    "height": entry.get("height"),
                    "opacity": entry.get("opacity"),
                    "easing": entry.get("easing"),
                },
                {
                    "id": eid,
                    "element": eid,
                    "name": eid,
                    "time": time,
                    "written_time": time,
                    "keyframe_time": time,
                    "present": False,
                    "found": False,
                    "realized": False,
                    "nearest_delivered_frame_time": nearest,
                    "keyframe_time_tolerance_seconds": tolerance,
                    "frame_interval_seconds": frame_interval,
                },
                "Written keyframe of element %r at %s seconds was not "
                "realized in the delivered frames: element %r does not "
                "appear in the delivered frames at %s seconds within "
                "the %s second frame interval tolerance."
                % (eid, time, eid, time, tolerance),
            )
        )

    for eid in delivered_ids:
        if eid in written_id_set:
            continue
        times = frame_times.get(eid) if isinstance(frame_times, dict) else None
        frame_count = len(times) if isinstance(times, list) else None
        findings.append(
            _finding(
                "element_not_written",
                "element %r" % (eid,),
                {
                    "id": None,
                    "element": None,
                    "name": None,
                    "written_element": None,
                    "matching_written_element": None,
                },
                {
                    "id": eid,
                    "element": eid,
                    "name": eid,
                    "present": True,
                    "declared": False,
                    "delivered_frame_count": frame_count,
                    "frame_count": frame_count,
                },
                "Element %r appears in the delivered frames but the "
                "written timeline never declares it: no written "
                "element has id %r." % (eid, eid),
            )
        )

    return {
        "findings": findings,
        "summary": _summary(findings),
        "keyframes_written": len(written_keyframes),
        "keyframes_realized": keyframes_realized,
        "keyframes_realized_at_written_times": keyframes_realized,
        "keyframes_realized_count": keyframes_realized,
        "frame_count_written": written_frames,
        "frames_written": written_frames,
        "frame_count_delivered": delivered_frames,
        "frames_delivered": delivered_frames,
        "written_frame_count": written_frames,
        "delivered_frame_count": delivered_frames,
        "duration_written": written_duration,
        "duration_delivered": delivered_duration,
        "written_duration": written_duration,
        "delivered_duration": delivered_duration,
        "elements_written": len(written_ids),
        "elements_delivered": len(delivered_ids),
        "tolerances": {
            "keyframe_time_tolerance_seconds": tolerance,
            "keyframe_tolerance_seconds": tolerance,
            "frame_interval_seconds": frame_interval,
            "duration_tolerance_seconds": tolerance,
            "keyframe_time_tolerance_description": (
                "a written keyframe is realized when the delivered "
                "frames hold its element within one frame interval "
                "(1 / frame_rate seconds) of its written time"
            ),
        },
    }


build_comparison = compute_comparison
