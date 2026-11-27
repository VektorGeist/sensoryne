"""stage2/judge.py - the judge seat, disk-only.

Reads a finished session's record (history.jsonl plus the
artifact store record) and appends judgment events to its
history. Imports no application module at module level,
renders nothing, perceives nothing, and never touches a
live session context. Roots are resolved from the registry
pending context exactly as controller.open_session does.
"""
import importlib
import json
import os
from datetime import datetime, timezone

from .store import ArtifactStore

_BOTTOM_LINES = ("good", "mixed", "not_good")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _pending(reg):
    for a in ("pending_context", "host_context", "default_context"):
        p = getattr(reg, a, None)
        if isinstance(p, dict) and p:
            return p
    s = getattr(reg, "sessions", None)
    if isinstance(s, dict):
        for k in ("__pending__", "pending"):
            if isinstance(s.get(k), dict):
                return s[k]
    return {}


def _roots(reg):
    pending = _pending(reg)
    store = pending.get("store") or ArtifactStore(pending.get("store_root") or os.path.join(os.getcwd(), "sessions", "artifacts"))
    root = str(pending.get("session_root") or os.path.join(os.getcwd(), "sessions"))
    return pending, store, root


def _read_history(path):
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _record(store, artifact_id):
    if not artifact_id:
        return None
    try:
        return store.load(artifact_id)
    except Exception:
        return None


def _application(store, artifact_id, session_id):
    record = _record(store, artifact_id)
    if record is not None:
        meta = record.get("metadata") if isinstance(record, dict) else None
        if isinstance(meta, dict) and meta.get("application"):
            return meta["application"]
    return str(session_id).split("-", 1)[0]


def _start_end(entries):
    start = next((e for e in entries if e.get("event") == "session_start"), {})
    ends = [e for e in entries if e.get("event") == "session_end"]
    end = ends[-1] if ends else {}
    return start, end


def list_sessions(reg):
    _pending_ctx, store, root = _roots(reg)
    sessions = []
    if not os.path.isdir(root):
        return {"sessions": sessions}
    for name in sorted(os.listdir(root)):
        sdir = os.path.join(root, name)
        hist = os.path.join(sdir, "history.jsonl")
        if not os.path.isdir(sdir) or not os.path.exists(hist):
            continue
        try:
            entries = _read_history(hist)
        except Exception:
            continue
        start, end = _start_end(entries)
        artifact_id = start.get("artifact_id")
        sessions.append({
            "session_id": name,
            "artifact_id": artifact_id,
            "application": _application(store, artifact_id, name),
            "condition": start.get("condition"),
            "objective": start.get("objective"),
            "ended_by": end.get("ended_by") if end else None,
            "iterations": sum(1 for e in entries if e.get("event") == "perception"),
            "judgments": sum(1 for e in entries if e.get("event") == "judgment"),
        })
    sessions.sort(key=lambda s: s["session_id"])
    return {"sessions": sessions}


def _judge_brief(pending, application):
    try:
        app = pending.get("app") if isinstance(pending, dict) else None
        if app is None:
            app = importlib.import_module("stage2.apps.%s.app" % application)
        fn = getattr(app, "judge_brief", None)
        return fn() if callable(fn) else ""
    except Exception:
        return ""


def open_judgment(reg, session_id):
    pending, store, root = _roots(reg)
    sdir = os.path.join(root, str(session_id))
    hist = os.path.join(sdir, "history.jsonl")
    if not os.path.isdir(sdir) or not os.path.exists(hist):
        raise ValueError("unknown session_id: %r" % (session_id,))
    try:
        entries = _read_history(hist)
    except Exception as exc:
        raise ValueError("unknown session_id: %r" % (session_id,)) from exc
    start, end = _start_end(entries)
    artifact_id = start.get("artifact_id") or session_id
    application = _application(store, artifact_id, session_id)
    written_work = []
    record = _record(store, artifact_id)
    if record is not None:
        for version in record.get("versions", []) or []:
            vid = version.get("version_id") if isinstance(version, dict) else None
            if vid is None:
                continue
            try:
                source = store.load_version_source(artifact_id, vid)
            except Exception:
                continue
            written_work.append({"version_id": vid, "source": source})
    iterations = []
    for entry in entries:
        if entry.get("event") != "perception":
            continue
        evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
        iterations.append({
            "iteration": entry.get("iteration"),
            "evidence_summary": evidence.get("summary"),
            "comparison": evidence.get("comparison"),
            "objective_coverage": evidence.get("objective_coverage"),
        })
    pack = {
        "session_id": session_id,
        "artifact_id": artifact_id,
        "application": application,
        "condition": start.get("condition"),
        "objective": start.get("objective"),
        "written_work": written_work,
        "iterations": iterations,
        "decisions": [e.get("decision") for e in entries if e.get("event") == "decision"],
        "ended_by": end.get("ended_by") if end else None,
        "final_assessment": end.get("assessment") if end else None,
        "judgments": [e.get("verdict") for e in entries if e.get("event") == "judgment"],
        "judge_brief": _judge_brief(pending, application),
    }
    if len(written_work) > 2 and len(json.dumps(pack, ensure_ascii=False, default=str)) > 150000:
        pack["written_work"] = [written_work[0], written_work[-1]]
    return pack


def _validate_verdict(verdict):
    if not isinstance(verdict, dict):
        raise ValueError("verdict must be an object")
    if verdict.get("bottom_line") not in _BOTTOM_LINES:
        raise ValueError("verdict.bottom_line must be one of 'good', 'mixed', 'not_good'")
    if not isinstance(verdict.get("judgment"), str) or not verdict["judgment"].strip():
        raise ValueError("verdict.judgment must be a non-empty string")
    for key in ("strengths", "weaknesses"):
        value = verdict.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError("verdict.%s must be a list of strings" % key)


def record_verdict(reg, session_id, verdict):
    _validate_verdict(verdict)
    _pending_ctx, _store, root = _roots(reg)
    sdir = os.path.join(root, str(session_id))
    hist = os.path.join(sdir, "history.jsonl")
    if not os.path.isdir(sdir) or not os.path.exists(hist):
        raise ValueError("unknown session_id: %r" % (session_id,))
    entry = {"ts": _now(), "event": "judgment", "session_id": session_id, "verdict": verdict}
    with open(hist, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"recorded": True, "session_id": session_id}
