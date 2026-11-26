"""stage2/decision.py - the platform-wide decision contract.

Every application uses the same decision shape. Revisions are
opaque to the core: the core hands them to the application's
revision engine and records what that engine applied or
rejected, with errors.
"""

import json

DECISION_SHAPE_TEXT = """Answer with one JSON object and nothing else:
{"faults": [{"description": "...", "where": "..."}], "revisions": [...], "complete": true or false, "assessment": "judgment of the work against the objective"}

An empty revisions list means no change is needed. complete is true only when you judge the rendered work meets the objective."""


def extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in reply")
    return json.loads(text[start:end + 1])


def parse_decision(text):
    decision = extract_json_object(text)
    if not isinstance(decision, dict):
        raise ValueError("decision must be a JSON object")
    for key in ("faults", "revisions", "complete", "assessment"):
        if key not in decision:
            raise ValueError("decision missing required key: %s" % key)
    if not isinstance(decision["faults"], list):
        raise ValueError("decision 'faults' must be a list")
    if not isinstance(decision["revisions"], list):
        raise ValueError("decision 'revisions' must be a list")
    return decision


def parse_artifact(text):
    artifact = extract_json_object(text)
    if not isinstance(artifact, dict):
        raise ValueError("artifact must be a JSON object")
    return artifact


def build_session_so_far(history_entries, evidence_summaries=None):
    """Build the 'Session so far' block from recorded history."""
    lines = ["Session so far:"]
    evidence_summaries = list(evidence_summaries or [])
    summary_index = 0
    for entry in history_entries:
        event = entry.get("event")
        if event == "decision":
            decision = entry.get("decision") or {}
            lines.append(
                "Iteration %s faults you reported: %s"
                % (entry.get("iteration"), json.dumps(decision.get("faults", [])))
            )
            lines.append(
                "Iteration %s assessment: %s"
                % (entry.get("iteration"), decision.get("assessment", ""))
            )
        elif event == "revision_applied":
            lines.append(
                "Revision applied at iteration %s: %s"
                % (entry.get("iteration"), json.dumps(entry.get("revision")))
            )
        elif event == "revision_rejected":
            lines.append(
                "Revision rejected at iteration %s: %s (error: %s)"
                % (
                    entry.get("iteration"),
                    json.dumps(entry.get("revision")),
                    entry.get("error"),
                )
            )
        elif event == "perception" and summary_index < len(evidence_summaries):
            lines.append(
                "Earlier evidence summary (iteration %s): %s"
                % (
                    entry.get("iteration"),
                    json.dumps(evidence_summaries[summary_index]),
                )
            )
            summary_index += 1
    if len(lines) == 1:
        lines.append("(no earlier iterations)")
    return "\n".join(lines)
