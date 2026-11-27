"""stage2/apps/code/comparison.py - Sensoryne's judgment of the
delivered program against the objective's stated checks.

The comparison works from the execution instrument's own
measurements of the delivered program (the evidence
measurements' per-check results: exit status, standard output,
standard error, run time, limit hit, and exact match against
the expected output), set against the written work as the
artifact states it: the tests the artifact states, each with
its name, input, and expected output.  It never works from
delivery-fault definitions: a finding names what the execution
measured against what was written.

Each finding is an object with kind, where, written, delivered,
and statement.  The kinds are check_failed, check_errored, and
check_timed_out.  The instrument's stated limits are carried
here unchanged: a 10 second limit per stated check and exact
character-for-character output comparison.
"""

import copy
import os

__all__ = [
    "FINDING_KINDS",
    "compute_comparison",
    "build_comparison",
]

FINDING_KINDS = (
    "check_failed",
    "check_errored",
    "check_timed_out",
)

_TIME_LIMIT_SECONDS = 10
try:
    from .instrument import TIME_LIMIT_SECONDS as _INSTRUMENT_LIMIT

    _TIME_LIMIT_SECONDS = _INSTRUMENT_LIMIT
except Exception:
    pass


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
            "No findings: the delivered program matches the "
            "objective's stated checks within the stated limits."
        )
    else:
        line = "%d finding(s): %s." % (
            total,
            ", ".join("%s=%d" % (kind, counts[kind]) for kind in sorted(counts)),
        )
    return {"counts": counts, "total": total, "line": line}


def _tests_of(source):
    if not isinstance(source, dict):
        return None
    for key in ("tests", "checks"):
        value = source.get(key)
        if isinstance(value, list) and value:
            return [entry for entry in value if isinstance(entry, dict)]
    objective = source.get("objective")
    if isinstance(objective, dict):
        for key in ("tests", "checks"):
            value = objective.get(key)
            if isinstance(value, list) and value:
                return [entry for entry in value if isinstance(entry, dict)]
    return None


def _written_checks(artifact, render_record, evidence):
    for source in (artifact, render_record, evidence):
        tests = _tests_of(source)
        if tests:
            return tests
        if isinstance(source, dict):
            structure = source.get("structure_record")
            if isinstance(structure, dict):
                tests = _tests_of(structure)
                if tests:
                    return tests
    if isinstance(render_record, dict):
        structure = render_record.get("structure_record")
        if isinstance(structure, dict):
            value = structure.get("tests")
            if isinstance(value, list) and value:
                return [entry for entry in value if isinstance(entry, dict)]
    return []


def _measurements_of(source):
    if not isinstance(source, dict):
        return None
    if isinstance(source.get("measurements"), dict):
        return source["measurements"]
    full = source.get("full")
    if isinstance(full, dict) and isinstance(full.get("measurements"), dict):
        return full["measurements"]
    if isinstance(source.get("results"), list):
        return source
    return None


def _results_of(render_record, evidence):
    for source in (evidence, render_record):
        measurements = _measurements_of(source)
        if isinstance(measurements, dict) and isinstance(
            measurements.get("results"), list
        ):
            return list(measurements["results"]), measurements
    return [], {}


def _time_limit_of(measurements):
    if isinstance(measurements, dict):
        for key in ("time_limit_seconds", "time_limit", "limit_seconds"):
            value = measurements.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
    return _TIME_LIMIT_SECONDS


def _fallback_results(render_record, evidence, written):
    """Execute the delivered program when no measurements are
    attached, using the instrument itself, as the other
    comparisons fall back to the delivered file."""
    path = None
    for source in (render_record, evidence):
        if isinstance(source, dict):
            candidate = source.get("path")
            if isinstance(candidate, str) and (
                os.path.isdir(candidate) or os.path.isfile(candidate)
            ):
                path = candidate
                break
    if path is None:
        return [], {}
    try:
        from .instrument import analyze_delivered

        result = analyze_delivered(path, list(written))
    except Exception:
        return [], {}
    measurements = result.get("measurements") if isinstance(result, dict) else None
    if isinstance(measurements, dict) and isinstance(
        measurements.get("results"), list
    ):
        return list(measurements["results"]), measurements
    return [], {}


def compute_comparison(artifact, render_record=None, evidence=None):
    """Compare the execution measurements against the stated checks."""
    written = _written_checks(artifact, render_record, evidence)
    results, measurements = _results_of(render_record, evidence)
    if not results and written:
        fallback, fallback_measurements = _fallback_results(
            render_record, evidence, written
        )
        if fallback:
            results = fallback
            measurements = fallback_measurements
    if not written and results:
        written = [
            {
                "name": entry.get("name", "") if isinstance(entry, dict) else "",
                "input": entry.get("input", "") if isinstance(entry, dict) else "",
                "expected": entry.get("expected") if isinstance(entry, dict) else None,
            }
            for entry in results
        ]
    time_limit = _time_limit_of(measurements)

    # Match each written check to its measured result, by name
    # in written order, each result used once.
    remaining = list(results)
    matched = []
    for check in written:
        name = check.get("name", check.get("check", "")) if isinstance(check, dict) else ""
        found = None
        for index, entry in enumerate(remaining):
            if not isinstance(entry, dict):
                continue
            if entry.get("name", entry.get("check", "")) == name:
                found = index
                break
        if found is not None:
            matched.append(remaining.pop(found))
        elif remaining:
            # No name match (for example unnamed checks): take
            # the next result in measured order.
            matched.append(remaining.pop(0))
        else:
            matched.append(None)

    findings = []
    passed = 0
    for check, result in zip(written, matched):
        if not isinstance(check, dict):
            continue
        name = check.get("name", check.get("check", ""))
        expected = check.get("expected")
        test_input = check.get("input", "")
        where = "check %s" % (name,)
        written_block = {
            "name": name,
            "check": name,
            "input": test_input,
            "expected": _j(expected),
            "expected_output": _j(expected),
            "expected_outcome": _j(expected),
        }
        if result is None:
            delivered_block = {
                "name": name,
                "check": name,
                "stdout": "",
                "actual": "",
                "actual_output": "",
                "output": "",
                "actual_outcome": "",
                "expected": _j(expected),
                "expected_output": _j(expected),
                "exit_status": None,
                "stderr": "no execution result was recorded for this check",
                "error": "no execution result was recorded for this check",
                "runtime_seconds": 0.0,
                "limit_hit": False,
                "matches_expected": False,
                "entry_file": None,
                "time_limit_seconds": time_limit,
            }
            findings.append(
                _finding(
                    "check_errored",
                    where,
                    written_block,
                    delivered_block,
                    "Check %r errored: no execution result was "
                    "recorded for this check; the error was %r."
                    % (name, delivered_block["error"]),
                )
            )
            continue
        stdout = result.get("stdout", "")
        if stdout is None:
            stdout = result.get("actual", result.get("actual_output", ""))
        stderr = result.get("stderr", "") or ""
        exit_status = result.get("exit_status")
        limit_hit = bool(result.get("limit_hit"))
        delivered_block = {
            "name": name,
            "check": name,
            "input": result.get("input", test_input),
            "stdout": _j(stdout),
            "actual": _j(stdout),
            "actual_output": _j(stdout),
            "output": _j(stdout),
            "actual_outcome": _j(stdout),
            "expected": _j(expected),
            "expected_output": _j(expected),
            "expected_outcome": _j(expected),
            "exit_status": exit_status,
            "stderr": _j(stderr),
            "error": _j(stderr),
            "runtime_seconds": result.get("runtime_seconds"),
            "limit_hit": limit_hit,
            "matches_expected": result.get("matches_expected"),
            "entry_file": result.get("entry_file"),
            "time_limit_seconds": time_limit,
        }
        if limit_hit:
            findings.append(
                _finding(
                    "check_timed_out",
                    where,
                    written_block,
                    delivered_block,
                    "Check %r timed out: the delivered program "
                    "exceeded the %s second time limit for this "
                    "check, producing stdout %r."
                    % (name, time_limit, stdout),
                )
            )
            continue
        errored = False
        if exit_status is not None and exit_status != 0:
            errored = True
        elif exit_status is None and stderr:
            errored = True
        elif result.get("entry_file") is None and stderr:
            errored = True
        if errored:
            findings.append(
                _finding(
                    "check_errored",
                    where,
                    written_block,
                    delivered_block,
                    "Check %r errored: the delivered program "
                    "exited with status %s and the error was %s"
                    % (name, exit_status, stderr),
                )
            )
            continue
        matches = result.get("matches_expected")
        if matches is None:
            matches = expected is not None and stdout == str(expected)
        if matches:
            passed += 1
            continue
        findings.append(
            _finding(
                "check_failed",
                where,
                written_block,
                delivered_block,
                "Check %r failed: expected output %r as written, "
                "but the delivered program produced %r."
                % (name, expected, stdout),
            )
        )

    counts = {}
    for finding in findings:
        counts[finding["kind"]] = counts.get(finding["kind"], 0) + 1
    checks_written = len(written)
    return {
        "findings": findings,
        "summary": _summary(findings),
        "checks_written": checks_written,
        "checks_stated": checks_written,
        "checks_total": checks_written,
        "tests_written": checks_written,
        "tests_total": checks_written,
        "tests_stated": checks_written,
        "test_count": checks_written,
        "checks_passed": passed,
        "checks_matched": passed,
        "tests_passed": passed,
        "matched_count": passed,
        "passed_count": passed,
        "checks_failed": counts.get("check_failed", 0),
        "failed_count": counts.get("check_failed", 0),
        "checks_errored": counts.get("check_errored", 0),
        "errored_count": counts.get("check_errored", 0),
        "checks_timed_out": counts.get("check_timed_out", 0),
        "timed_out_count": counts.get("check_timed_out", 0),
        "tolerances": {
            "time_limit_seconds": time_limit,
            "time_limit": time_limit,
            "exact_match": True,
            "output_comparison": "exact character-for-character equality",
        },
    }


build_comparison = compute_comparison
