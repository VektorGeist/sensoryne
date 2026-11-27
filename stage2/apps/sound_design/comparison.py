"""stage2/apps/sound_design/comparison.py - Sensoryne's judgment of the
delivered audio against the written event list.

The comparison works from the non-pitch analyzer's own
measurements of the delivered audio: the realized onsets the
analyzer measured for the delivered events (the evidence
measurements' per-event onset_time / realized_onset), set
against the written events the artifact declares, each with
its written onset (start).  It never works from delivery-fault
definitions and never from the delivered event list alone: a
finding names what a realized onset, or the absence of one,
shows against what was written.

Each finding is an object with kind, where, written, delivered,
and statement.  The kinds are event_not_realized,
onset_displaced, and onset_not_written.  The onset tolerance
is the application's stated 0.1 seconds: a realized onset
within 0.1 seconds of a written onset realizes that event at
its written onset.
"""

import copy
import os

__all__ = [
    "ONSET_TOLERANCE_SECONDS",
    "FINDING_KINDS",
    "compute_comparison",
    "build_comparison",
]

ONSET_TOLERANCE_SECONDS = 0.1

FINDING_KINDS = (
    "event_not_realized",
    "onset_displaced",
    "onset_not_written",
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
            "No findings: the delivered audio matches the written "
            "events within the stated tolerances."
        )
    else:
        line = "%d finding(s): %s." % (
            total,
            ", ".join(
                "%s=%d" % (kind, counts[kind]) for kind in sorted(counts)
            ),
        )
    return {"counts": counts, "total": total, "line": line}


def _written_events(artifact):
    entries = []
    raw = artifact.get("events", []) if isinstance(artifact, dict) else []
    for event in raw or []:
        if not isinstance(event, dict):
            continue
        if event.get("kind") == "silence":
            continue
        try:
            start = float(event.get("start", 0.0))
        except Exception:
            continue
        try:
            duration = float(event.get("duration", 0.0))
        except Exception:
            duration = 0.0
        eid = event.get("id")
        entries.append(
            {
                "id": eid,
                "event": eid,
                "name": eid,
                "kind": event.get("kind"),
                "start": start,
                "written_onset": start,
                "written_start": start,
                "onset": start,
                "duration": duration,
                "end": start + duration,
                "gain": event.get("gain"),
            }
        )
    return entries


def _measured_events(render_record, evidence):
    for source in (evidence, render_record):
        if not isinstance(source, dict):
            continue
        measurements = source.get("measurements")
        if isinstance(measurements, dict) and isinstance(
            measurements.get("events"), list
        ):
            return list(measurements["events"])
        full = source.get("full")
        if isinstance(full, dict):
            full_measurements = full.get("measurements")
            if isinstance(full_measurements, dict) and isinstance(
                full_measurements.get("events"), list
            ):
                return list(full_measurements["events"])
    # No perceived measurements are attached (for example a
    # control-condition render, where nothing was perceived for
    # the model): fall back to the application's own onset
    # measurement of the delivered file, when the render
    # record carries a file and the delivered events.
    if isinstance(render_record, dict):
        path = render_record.get("path")
        delivered = render_record.get("delivered_events")
        if delivered is None:
            delivered = render_record.get("events")
        if (
            isinstance(path, str)
            and os.path.isfile(path)
            and isinstance(delivered, list)
            and delivered
        ):
            try:
                from .instrument import perceive_audio

                result = perceive_audio(
                    path, delivered, render_record.get("duration")
                )
                measurements = result.get("measurements") if isinstance(result, dict) else None
                if isinstance(measurements, dict) and isinstance(
                    measurements.get("events"), list
                ):
                    return list(measurements["events"])
            except Exception:
                pass
    return []


def _realized_onsets(render_record, evidence):
    realized = []
    for entry in _measured_events(render_record, evidence):
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") == "silence":
            continue
        onset = entry.get("onset_time")
        if onset is None:
            onset = entry.get("realized_onset")
        if onset is None:
            onset = entry.get("onset")
        if onset is None:
            continue
        try:
            onset_f = float(onset)
        except Exception:
            continue
        eid = entry.get("id")
        realized.append(
            {
                "id": eid,
                "event": eid,
                "name": eid,
                "kind": entry.get("kind"),
                "onset_time": onset_f,
                "realized_onset": onset_f,
                "onset": onset_f,
                "time": onset_f,
            }
        )
    realized.sort(key=lambda item: item["onset_time"])
    return realized


def compute_comparison(artifact, render_record=None, evidence=None):
    """Compare the realized onsets against the written events."""
    if evidence is None and isinstance(render_record, dict):
        if "measurements" in render_record:
            evidence = render_record

    written = _written_events(artifact)
    realized = _realized_onsets(render_record, evidence)
    tolerance = ONSET_TOLERANCE_SECONDS

    # Realized at the written onset: greedy matching of a
    # realized onset to a written onset within tolerance,
    # closest first, each used once.
    candidates = []
    for wi, w in enumerate(written):
        for ri, r in enumerate(realized):
            distance = abs(r["onset_time"] - w["start"])
            if distance <= tolerance + _EPS:
                candidates.append((distance, wi, ri))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_written = {}
    matched_realized = {}
    for _distance, wi, ri in candidates:
        if wi in matched_written or ri in matched_realized:
            continue
        matched_written[wi] = ri
        matched_realized[ri] = wi

    events_realized_at_written = len(matched_written)

    remaining_written = [
        wi for wi in range(len(written)) if wi not in matched_written
    ]
    remaining_realized = {
        ri for ri in range(len(realized)) if ri not in matched_realized
    }

    # Displaced: a remaining written event whose own realized
    # onset (the delivered event carrying its id) remains, the
    # nearest realized onset for it.  A written event with no
    # such realized onset went unrealized; a remaining
    # realized onset matching no written event within tolerance
    # was not written.
    displaced = {}
    for wi in list(remaining_written):
        w = written[wi]
        found = None
        for ri in sorted(remaining_realized):
            r = realized[ri]
            if r.get("id") is not None and r.get("id") == w.get("id"):
                found = ri
                break
        if found is None:
            # No id to carry the identity: fall back to a
            # mutual-nearest pairing with an id-less realized
            # onset, so a displaced onset stated without an id
            # is still displaced rather than unrealized plus
            # not written.
            idless = [
                ri
                for ri in remaining_realized
                if realized[ri].get("id") is None
            ]
            if idless and remaining_written:
                nearest_ri = min(
                    idless,
                    key=lambda ri: abs(
                        realized[ri]["onset_time"] - w["start"]
                    ),
                )
                nearest_wi = min(
                    remaining_written,
                    key=lambda idx: abs(
                        realized[nearest_ri]["onset_time"]
                        - written[idx]["start"]
                    ),
                )
                if nearest_wi == wi:
                    found = nearest_ri
        if found is not None:
            displaced[wi] = found
            remaining_realized.discard(found)

    findings = []
    for wi, w in enumerate(written):
        if wi in matched_written:
            continue
        if wi in displaced:
            r = realized[displaced[wi]]
            realized_onset = r["onset_time"]
            displacement = float(realized_onset - w["start"])
            where = "event %s at %s seconds" % (w.get("id"), w["start"])
            findings.append(
                _finding(
                    "onset_displaced",
                    where,
                    {
                        "id": w.get("id"),
                        "event": w.get("id"),
                        "name": w.get("id"),
                        "kind": w.get("kind"),
                        "start": w["start"],
                        "written_onset": w["start"],
                        "written_start": w["start"],
                        "onset": w["start"],
                        "duration": w.get("duration"),
                    },
                    {
                        "id": r.get("id"),
                        "event": r.get("id"),
                        "name": r.get("id"),
                        "kind": r.get("kind"),
                        "realized_onset": realized_onset,
                        "onset_time": realized_onset,
                        "onset": realized_onset,
                        "time": realized_onset,
                        "displacement_seconds": displacement,
                        "displacement": float(abs(displacement)),
                        "absolute_displacement_seconds": float(
                            abs(displacement)
                        ),
                        "difference_seconds": displacement,
                        "written_onset": w["start"],
                        "onset_tolerance_seconds": tolerance,
                    },
                    "Written event %r at %s seconds was realized at "
                    "%s seconds, displaced by %s seconds: the nearest "
                    "realized onset is %s seconds, %s seconds from its "
                    "written onset %s seconds."
                    % (
                        w.get("id"),
                        w["start"],
                        realized_onset,
                        displacement,
                        realized_onset,
                        displacement,
                        w["start"],
                    ),
                )
            )
            continue
        where = "event %s at %s seconds" % (w.get("id"), w["start"])
        findings.append(
            _finding(
                "event_not_realized",
                where,
                {
                    "id": w.get("id"),
                    "event": w.get("id"),
                    "name": w.get("id"),
                    "kind": w.get("kind"),
                    "start": w["start"],
                    "written_onset": w["start"],
                    "written_start": w["start"],
                    "onset": w["start"],
                    "duration": w.get("duration"),
                },
                {
                    "id": w.get("id"),
                    "event": w.get("id"),
                    "name": w.get("id"),
                    "realized_onset": None,
                    "onset_time": None,
                    "onset": None,
                    "present": False,
                    "realized": False,
                    "written_onset": w["start"],
                    "onset_tolerance_seconds": tolerance,
                },
                "Written event %r at %s seconds was not realized in the "
                "delivered audio: no realized onset was heard at its "
                "written onset %s seconds."
                % (w.get("id"), w["start"], w["start"]),
            )
        )

    for ri in sorted(remaining_realized, key=lambda i: realized[i]["onset_time"]):
        r = realized[ri]
        onset_time = r["onset_time"]
        if any(
            abs(onset_time - w["start"]) <= tolerance + _EPS
            for w in written
        ):
            continue
        findings.append(
            _finding(
                "onset_not_written",
                "realized onset at %s seconds" % (onset_time,),
                {
                    "id": None,
                    "event": None,
                    "name": None,
                    "matching_written_event": None,
                    "written_onset": None,
                    "matching_written_onset": None,
                    "onset_tolerance_seconds": tolerance,
                },
                {
                    "id": r.get("id"),
                    "event": r.get("id"),
                    "name": r.get("id"),
                    "kind": r.get("kind"),
                    "realized_onset": onset_time,
                    "onset_time": onset_time,
                    "onset": onset_time,
                    "time": onset_time,
                    "onset_tolerance_seconds": tolerance,
                },
                "Realized onset at %s seconds matches no written event "
                "within %s seconds: no written event has an onset within "
                "tolerance of %s seconds."
                % (onset_time, tolerance, onset_time),
            )
        )

    return {
        "findings": findings,
        "summary": _summary(findings),
        "events_written": len(written),
        "events_realized": events_realized_at_written,
        "events_realized_at_written_onsets": events_realized_at_written,
        "events_realized_at_written": events_realized_at_written,
        "events_not_realized": sum(
            1 for f in findings if f["kind"] == "event_not_realized"
        ),
        "onsets_realized": len(realized),
        "tolerances": {
            "onset_tolerance_seconds": tolerance,
            "onset_tolerance": tolerance,
            "tolerance_seconds": tolerance,
        },
    }


build_comparison = compute_comparison
