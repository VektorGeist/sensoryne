"""stage2/report.py - session report.

Built only from history.jsonl (plus the ledger for spend).
Nothing is recomputed or re-judged. The report is written in
the core's own terms: artifact, render, evidence, decision.
Where a perception entry carries a delta, invariant, or
coverage block, those blocks are reproduced as recorded.
"""

import html
import json
import os

from .ember import SpendLedger


def _esc(value):
    return html.escape(str(value), quote=True)


def _load_history(session_dir):
    path = os.path.join(str(session_dir), "history.jsonl")
    entries = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def build_report(session_dir, ledger_path):
    """Write report.html in ``session_dir`` and return its path."""
    session_dir = str(session_dir)
    entries = _load_history(session_dir)
    start = next((e for e in entries if e.get("event") == "session_start"), {})
    session_id = start.get("session_id") or os.path.basename(os.path.abspath(session_dir))
    artifact_id = start.get("artifact_id", "")
    condition = start.get("condition", "")
    objective = start.get("objective", "")
    faults = start.get("faults") or []
    ground_truth = start.get("ground_truth") or ""
    end = next((e for e in reversed(entries) if e.get("event") == "session_end"), {})

    call_count = 0
    spend = 0.0
    if ledger_path and os.path.exists(str(ledger_path)):
        ledger = SpendLedger(ledger_path)
        call_count = ledger.session_call_count(session_id)
        spend = ledger.session_total(session_id)

    parts = []
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    parts.append("<title>Session %s - Sensoryne report</title></head><body>" % _esc(session_id))
    parts.append("<h1>Session %s</h1>" % _esc(session_id))
    parts.append("<p>Artifact: %s</p>" % _esc(artifact_id))
    parts.append("<p>Condition: %s</p>" % _esc(condition))
    parts.append("<p>API calls: %d</p>" % call_count)
    parts.append("<p>Spend: $%.4f</p>" % spend)
    parts.append("<p>Objective: %s</p>" % _esc(objective))
    if faults:
        parts.append("<h2>Planted faults</h2><pre>%s</pre>" % _esc(json.dumps(faults, indent=2)))
        parts.append("<h2>Ground truth</h2><pre>%s</pre>" % _esc(ground_truth))
    parts.append("<h2>Iterations</h2>")
    for entry in entries:
        event = entry.get("event")
        if event == "perception":
            evidence = entry.get("evidence") or {}
            summary = evidence.get("summary")
            if summary is None:
                measurements = evidence.get("measurements") or {}
                if isinstance(measurements, dict):
                    summary = measurements.get("heard", "")
            parts.append("<section><h3>Iteration %s - evidence summary</h3><p>%s</p></section>" % (_esc(entry.get("iteration")), _esc(summary or "")))
            for key in ("delta", "invariants", "reliability", "objective_coverage", "robustness"):
                if evidence.get(key) is not None:
                    parts.append("<section><h3>%s</h3><pre>%s</pre></section>" % (_esc(key), _esc(json.dumps(evidence.get(key), indent=2))))
            if entry.get("structure_record") is not None:
                parts.append("<section><h3>Structure record</h3><pre>%s</pre></section>" % _esc(json.dumps(entry.get("structure_record"), indent=2)))
        elif event == "decision":
            decision = entry.get("decision") or {}
            parts.append("<section><h3>Iteration %s - decision</h3><p>Faults: %s</p><p>Revisions: %s</p><p>Complete: %s</p><p>Assessment: %s</p></section>" % (_esc(entry.get("iteration")), _esc(json.dumps(decision.get("faults", []))), _esc(json.dumps(decision.get("revisions", []))), _esc(decision.get("complete")), _esc(decision.get("assessment", ""))))
        elif event == "revision_rejected":
            parts.append("<section><h3>Iteration %s - rejected revision</h3><p>Revision: %s</p><p>Error: %s</p></section>" % (_esc(entry.get("iteration")), _esc(json.dumps(entry.get("revision"))), _esc(entry.get("error", ""))))
    for entry in entries:
        if entry.get("event") == "judgment":
            verdict = entry.get("verdict") or {}
            parts.append("<section><h2>Judgment</h2><p>Bottom line: %s</p><p>%s</p><p>Strengths: %s</p><p>Weaknesses: %s</p></section>" % (_esc(verdict.get("bottom_line", "")), _esc(verdict.get("judgment", "")), _esc(json.dumps(verdict.get("strengths", []))), _esc(json.dumps(verdict.get("weaknesses", [])))))
    if end:
        parts.append("<h2>Ended by: %s after %s iterations</h2>" % (_esc(end.get("ended_by")), _esc(end.get("iterations"))))
        if end.get("assessment"):
            parts.append("<p>Assessment: %s</p>" % _esc(end.get("assessment")))
    parts.append("</body></html>")
    report_path = os.path.join(session_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
    return report_path
