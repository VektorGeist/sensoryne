"""stage2/controller.py - session operations as callable services.

The model reaches this module through the tool surface. The
operations here are the platform side of each tool call:
session opening, validation with retry text on failure, the
render seam calling the supplied renderer, perception by
condition, decision application with per-operation records,
termination labels, and history logging. This module assembles
no prompts of its own and decides nothing. It contains no
application logic and no application vocabulary except
_robustness, one music-specific path in the core, which
renders the shared robustness variations for the music
application through the generic variation and rendering
seams.

Renderer output contract: out_path is a single file path. Every
renderer writes its primary output file exactly at out_path and
returns "path" equal to out_path. A renderer that produces
several files derives the additional file paths from out_path:
animation frames as <stem>_frame_NNN<ext>, and the code
application's delivered files in the directory <out_path>_files.

Version naming: versions are identified everywhere by the
store's version identifier (v1, v2, ...). A delta's
from_version and to_version use that identifier at both ends.
A probe delta names the hypothetical end as the next store
identifier with a -probe suffix, for example v2-probe.

Delta rule: a delta compares measurements that were actually
perceived. Every measurement path in a delta is rooted at
measurements, as a path in the evidence document: changed,
added, removed, and unchanged measurement paths all carry the
measurements. prefix. Perceptual versions after the first
carry a delta in their evidence and in the version's render
record. Control versions carry no delta, because nothing was
perceived; no comparison against an empty document is stored
as a delta. A probe in the control condition returns the
control submission shape for the hypothetical artifact, with
probe true at the top level, and no delta.
"""

import copy
import hashlib
import importlib
import json
import os
from datetime import datetime, timezone

from . import decision as decision_module
from .delta import compute_delta
from .shape import shape_document, shape_evidence
from .store import ArtifactStore


def _now():
    return datetime.now(timezone.utc).isoformat()


def _canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _hash(v):
    return hashlib.sha256(_canon(v).encode()).hexdigest()


def _next_version_id(store, artifact_id):
    try:
        record = store.load(artifact_id)
        return "v%d" % (len(record.get("versions", [])) + 1)
    except Exception:
        return "v1"


def _version_number(version_id):
    if isinstance(version_id, str) and version_id.startswith("v"):
        try:
            return int(version_id[1:])
        except ValueError:
            return 1
    return 1


def _rooted_measurements(delta):
    """Root every measurement path in a delta at measurements.

    compute_delta compares the measurements documents. In the
    evidence document those measurements live under the
    measurements key, so the stored and returned delta names
    them by their place in that document.
    """
    if not isinstance(delta, dict):
        return delta
    rooted = copy.deepcopy(delta)
    for entry in rooted.get("changed_measurements", []) or []:
        if isinstance(entry, dict) and entry.get("path"):
            entry["path"] = "measurements." + str(entry["path"])
    for key in ("added_measurements", "removed_measurements", "unchanged_keys"):
        values = rooted.get(key) or []
        rooted[key] = ["measurements." + str(v) for v in values]
    return rooted


class _History:
    def __init__(self, path):
        self.path = path
        self.entries = []

    def log(self, event, **f):
        e = {"ts": _now(), "event": event}
        e.update(f)
        self.entries.append(e)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(e, ensure_ascii=False) + "\n")
        return e

    def rewrite(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            for e in self.entries:
                handle.write(json.dumps(e, ensure_ascii=False) + "\n")


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


def _app(name, pending):
    if isinstance(pending, dict) and pending.get("app") is not None:
        return pending["app"]
    return importlib.import_module("stage2.apps.%s.app" % name)


def _renderer(name, pending):
    r = pending.get("renderer") if isinstance(pending, dict) else None
    if callable(r):
        return r
    m = importlib.import_module("%s_renderer" % name)
    r = getattr(m, "renderer")
    if not callable(r):
        raise ValueError("no renderer callable")
    return r


def _extras(app):
    # The application's own invariants module is preferred when
    # it exists, so that its judgments, reliability block, and
    # coverage are the ones reported. An application that
    # carries those functions on itself is its own extras.
    if getattr(app, "NAME", None):
        try:
            return importlib.import_module("stage2.apps.%s.invariants" % app.NAME)
        except Exception:
            pass
    if hasattr(app, "evaluate_invariants") or hasattr(app, "reliability_block") or hasattr(app, "objective_coverage"):
        return app
    name = getattr(app, "NAME", None)
    if name:
        try:
            return importlib.import_module("stage2.apps.%s.invariants" % name)
        except Exception:
            pass
    return app


def _summary(ev):
    if not isinstance(ev, dict):
        return {"summary": str(ev), "measurements": {}}
    s = ev.get("summary")
    if s is None and isinstance(ev.get("measurements"), dict):
        s = ev["measurements"].get("heard")
    return {"summary": s, "measurements": ev.get("measurements", {})}


def _shape_fields(ctx):
    fields = {"decision_shape_text": decision_module.DECISION_SHAPE_TEXT}
    app = ctx.get("app") if isinstance(ctx, dict) else None
    text = getattr(app, "REVISION_SHAPE_TEXT", None)
    if isinstance(text, str) and text:
        fields["revision_shape_text"] = text
    return fields

def _with_obj(ctx, art):
    c = copy.deepcopy(art)
    o = ctx.get("objective")
    if isinstance(c, dict) and isinstance(o, dict) and "objective" not in c:
        c["objective"] = copy.deepcopy(o)
    return c


def _structure(ctx, art):
    return ctx["app"].structure_record(_with_obj(ctx, art))


def _presentation(ctx, ev, ins, st):
    app = ctx["app"]
    if not hasattr(app, "perceptual_prompt"):
        return None
    try:
        return app.perceptual_prompt(ctx.get("objective"), ev, ins, st)
    except TypeError:
        try:
            return app.perceptual_prompt(ctx.get("objective"), ev, ins)
        except TypeError:
            return None


def _store_entry(store, aid, src, obj, meta):
    try:
        return store.load(aid)
    except FileNotFoundError:
        return store.create(aid, src, obj, metadata=meta or {})


def _update_src(store, aid, src, obj):
    r = store.load(aid)
    r["source"] = copy.deepcopy(src)
    if obj is not None:
        r["objective"] = copy.deepcopy(obj)
    from .store import _write_json
    _write_json(store._record_path(aid), r)
    return r


def _comparison(ctx, art, rec, ev):
    """The comparison block for this application, if it has one.

    Preferred from the application's own invariants module,
    then from a comparison module beside the application, then
    from the application module itself. Returns None when the
    application has no comparison, so applications without one
    store no comparison key.
    """
    name = getattr(ctx["app"], "NAME", None)
    candidates = []
    seen = set()
    for modname in (
        ("stage2.apps.%s.invariants" % name) if name else None,
        ("stage2.apps.%s.comparison" % name) if name else None,
    ):
        if modname and modname not in seen:
            seen.add(modname)
            try:
                candidates.append(importlib.import_module(modname))
            except Exception:
                pass
    candidates.append(ctx["app"])
    for mod in candidates:
        fn = getattr(mod, "compute_comparison", None) or getattr(mod, "build_comparison", None) or getattr(mod, "comparison_block", None)
        if fn is None:
            continue
        try:
            return fn(_with_obj(ctx, art), rec, ev)
        except TypeError:
            try:
                return fn(_with_obj(ctx, art), rec)
            except TypeError:
                return fn(_with_obj(ctx, art))
    return None


def _invariants(ctx, art, rec, ev):
    ex = _extras(ctx["app"])
    fn = getattr(ex, "evaluate_invariants", None) or getattr(ctx["app"], "evaluate_invariants", None)
    if fn is None:
        return {"checked": [], "summary": {"total_checked": 0, "held": 0, "broken": 0}}
    r = copy.deepcopy(rec)
    r["original_artifact"] = copy.deepcopy(ctx.get("original_source"))
    r["objective"] = copy.deepcopy(ctx.get("objective"))
    try:
        out = fn(_with_obj(ctx, art), r, ev)
    except TypeError:
        out = fn(_with_obj(ctx, art), r)
    checked = list(out.get("checked", []) if isinstance(out, dict) else out or [])
    held = sum(1 for c in checked if c.get("held") is True)
    summary = {"total_checked": len(checked), "held": held, "broken": len(checked) - held}
    if isinstance(out, dict) and isinstance(out.get("summary"), dict):
        for k in summary:
            if k in out["summary"]:
                summary[k] = out["summary"][k]
    return {"checked": checked, "summary": summary}


def _reliability(ctx):
    ex = _extras(ctx["app"])
    fn = getattr(ex, "reliability_block", None) or getattr(ctx["app"], "reliability_block", None)
    if fn is None:
        return {"instrument": getattr(ctx["app"], "NAME", "unknown"), "limits": []}
    try:
        return fn()
    except TypeError:
        return fn(ctx["app"])


def _coverage(ctx, art, rec, ev):
    ex = _extras(ctx["app"])
    fn = getattr(ex, "objective_coverage", None) or getattr(ctx["app"], "objective_coverage", None)
    if fn is None:
        return []
    r = copy.deepcopy(rec)
    r["objective"] = copy.deepcopy(ctx.get("objective"))
    try:
        return list(fn(_with_obj(ctx, art), r, ev) or [])
    except TypeError:
        return list(fn(_with_obj(ctx, art), r) or [])


def _robustness(ctx, art, rec, ev):
    ex = _extras(ctx["app"])
    app = ctx["app"]
    fn = getattr(ex, "robustness_variations", None) or getattr(app, "robustness_variations", None)
    if fn is None:
        return None
    try:
        variations = fn()
    except TypeError:
        variations = fn(app)
    if not variations:
        return None
    tf = getattr(app, "transform_artifact_for_variation", None) or getattr(ex, "transform_artifact_for_variation", None)
    if tf is None:
        return None
    entries = []
    pitches = []
    louds = []
    counts = {}
    for var in variations:
        name = var.get("name") if isinstance(var, dict) else str(var)
        try:
            transformed = tf(copy.deepcopy(art), var)
        except TypeError:
            transformed = tf(copy.deepcopy(art), name)
        if transformed is None:
            continue
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))
        out = os.path.join(ctx["session_dir"], "robustness", "%s_%d.wav" % (safe, ctx.get("iteration", 0)))
        r2 = ctx["renderer"](copy.deepcopy(transformed), out, copy.deepcopy(ctx.get("faults") or []))
        if not isinstance(r2, dict):
            r2 = {"path": str(r2)}
        r2.setdefault("path", out)
        r2["_artifact"] = copy.deepcopy(transformed)
        r2["objective"] = copy.deepcopy(ctx.get("objective"))
        try:
            ev2 = ctx["app"].perceive(r2)
        except Exception:
            continue
        m = ev2.get("measurements", {}) if isinstance(ev2, dict) else {}
        pitch = m.get("pitch_hz", m.get("f0"))
        loud = m.get("loudness", m.get("loud"))
        if isinstance(pitch, (int, float)) and not isinstance(pitch, bool):
            pitches.append(float(pitch))
        if isinstance(loud, (int, float)) and not isinstance(loud, bool):
            louds.append(float(loud))
        st = ctx["app"].structure_record(_with_obj(ctx, transformed))
        nc = {}
        for part in st.get("parts", []) if isinstance(st, dict) else []:
            if isinstance(part, dict) and "part" in part:
                nc[part["part"]] = part.get("note_count")
        for k, v in nc.items():
            counts.setdefault(k, []).append(v)
        entries.append({"variation": name, "pitch_hz": pitch, "loudness": loud, "note_counts": nc})
    if not entries:
        return None

    def spread(vals):
        return {"min": min(vals), "max": max(vals)} if vals else {"min": None, "max": None}

    return {"variations": entries, "spread": {"pitch_hz": spread(pitches), "loudness": spread(louds), "note_counts": {k: spread([float(x) for x in v if isinstance(x, (int, float))]) for k, v in counts.items()}}}


def _memory_path(ctx_like, application):
    roots = []
    if isinstance(ctx_like, dict):
        if ctx_like.get("session_root"):
            roots.append(os.path.join(str(ctx_like["session_root"]), "episodic_memory", "%s.jsonl" % application))
        st = ctx_like.get("store")
        if st is not None and getattr(st, "root", None):
            roots.append(os.path.join(str(st.root), "episodic_memory", "%s.jsonl" % application))
    for r in roots:
        if os.path.exists(r):
            return r
    return roots[0] if roots else os.path.join(os.getcwd(), "sessions", "episodic_memory", "%s.jsonl" % application)


def _append_memory(ctx, ended_by):
    if ended_by != "model_complete" or not ctx.get("_work_submitted"):
        return
    appname = ctx.get("application") or getattr(ctx["app"], "NAME", "unknown")
    path = _memory_path(ctx, appname)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    structure = ctx.get("current_structure") or (_structure(ctx, ctx["current_source"]) if ctx.get("current_source") is not None else None)
    summaries = ctx.get("evidence_summaries") or []
    rec = {"application": appname, "objective": copy.deepcopy(ctx.get("objective")), "structure_record": copy.deepcopy(structure), "evidence_summary": copy.deepcopy(summaries[-1]) if summaries else {"summary": None, "measurements": {}}, "termination": ended_by, "ended_by": ended_by, "session_id": ctx.get("session_id")}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _end(ctx, ended_by, assessment=None):
    ctx["ended_by"] = ended_by
    ctx["assessment"] = assessment
    ctx["history"].log("session_end", session_id=ctx["session_id"], iterations=ctx.get("iteration", 0), ended_by=ended_by, assessment=assessment)
    _append_memory(ctx, ended_by)
    report = None
    if ctx.get("ledger_path"):
        try:
            from .report import build_report
            report = build_report(ctx["session_dir"], ctx["ledger_path"])
        except Exception:
            report = None
    return {"session_id": ctx["session_id"], "ended_by": ended_by, "assessment": assessment, "report": report, "session_dir": ctx["session_dir"]}


def _render_perceive(ctx):
    """Render the current artifact, store its version, perceive it.

    One render, one version. The version identifier is the
    store's next identifier, so the delta for this version can
    be computed before the version is stored and stored with
    it. A delta is computed only for perceived versions.
    """
    app = ctx["app"]
    src = copy.deepcopy(ctx["current_source"])
    sh = _hash(src)
    ctx["iteration"] = int(ctx.get("iteration", 0)) + 1
    it = ctx["iteration"]
    out = os.path.join(ctx["session_dir"], "render_%d" % it)
    rec = ctx["renderer"](copy.deepcopy(src), out, copy.deepcopy(ctx.get("faults") or []))
    if not isinstance(rec, dict):
        rec = {"path": str(rec)}
    rec.setdefault("path", out)
    prec = copy.deepcopy(rec)
    prec["_artifact"] = copy.deepcopy(src)
    prec["objective"] = copy.deepcopy(ctx.get("objective"))
    ev = None
    if ctx.get("condition") == "perceptual":
        ev = app.perceive(prec)
        ctx.setdefault("evidence_summaries", []).append(_summary(ev))
    structure = _structure(ctx, src)
    # For Music & Sound, the comparison is computed and
    # attached before invariants and coverage run, so those
    # blocks read the same comparison figures that are
    # stored in this pack. Other applications keep the
    # original order, so their blocks are unchanged.
    comp = None
    comp_computed = False
    if ev is not None and (getattr(app, "NAME", None) == "music" or ctx.get("application") == "music"):
        comp = _comparison(ctx, src, rec, ev)
        comp_computed = True
        if comp is not None:
            rec["comparison"] = comp
            ev["comparison"] = comp
    inv = _invariants(ctx, src, rec, ev)
    rel = _reliability(ctx)
    cov = _coverage(ctx, src, rec, ev)
    rob = _robustness(ctx, src, rec, ev) if ev is not None else None
    # The comparison is computed from the hearing's evidence
    # (and the performance record) against the written work,
    # and attached to both the performance record and the
    # evidence before either is stored, so the stored record
    # and the returned copy carry the same block.
    if not comp_computed:
        comp = _comparison(ctx, src, rec, ev)
    rec["invariants"] = inv
    rec["reliability"] = rel
    rec["objective_coverage"] = cov
    if comp is not None:
        rec["comparison"] = comp
    if rob is not None:
        rec["robustness"] = rob
    if ev is not None:
        ev["invariants"] = inv
        ev["reliability"] = rel
        ev["objective_coverage"] = cov
        if comp is not None:
            ev["comparison"] = comp
        if rob is not None:
            ev["robustness"] = rob
    vid = _next_version_id(ctx["store"], ctx["artifact_id"])
    delta = None
    if ev is not None and ctx.get("current_version_id") is not None:
        delta = _rooted_measurements(compute_delta(ctx["current_version_id"], vid, ctx.get("current_measurements") or {}, ev.get("measurements", {}), ctx.get("current_structure") or {}, structure))
        ev["delta"] = delta
        rec["delta"] = delta
    for k in list(rec):
        if k.startswith("_"):
            del rec[k]
    ver = ctx["store"].add_version(ctx["artifact_id"], rec["path"], copy.deepcopy(rec), copy.deepcopy(src))
    if ver["version_id"] != vid:
        raise RuntimeError("version identifier mismatch during render")
    stored = ctx["store"].load_version_source(ctx["artifact_id"], vid)
    if _hash(stored) != sh:
        raise RuntimeError("version integrity: stored source mismatch")
    record = ctx["store"].load(ctx["artifact_id"])
    if not record.get("versions") or record["versions"][-1].get("version_id") != vid:
        raise RuntimeError("version integrity: not latest")
    rec["source_hash"] = sh
    rec["version_id"] = vid
    ctx["history"].log("render", iteration=it, version_id=vid, render_info=copy.deepcopy(rec), structure_record=copy.deepcopy(structure))
    if ctx.get("condition") == "perceptual":
        ctx["history"].log("perception", iteration=it, evidence=ev, structure_record=copy.deepcopy(structure))
    else:
        ctx["history"].log("perception_skipped", iteration=it, reason="control condition: the model evaluates the artifact alone and cannot perceive the render", structure_record=copy.deepcopy(structure))
    ctx["current_version_id"] = vid
    ctx["current_structure"] = copy.deepcopy(structure)
    ctx["current_evidence"] = copy.deepcopy(ev) if ev is not None else ctx.get("current_evidence")
    ctx["current_measurements"] = copy.deepcopy(ev.get("measurements", {})) if isinstance(ev, dict) else ctx.get("current_measurements")
    return rec, ev, structure, delta


def _plant(ctx):
    if not ctx.get("test") or ctx.get("_faults_planted"):
        return
    faults = copy.deepcopy(list(ctx["app"].plant_faults(ctx["original_source"]) or []))
    gt = ctx["app"].generate_ground_truth(faults)
    ctx["faults"] = faults
    ctx["ground_truth"] = gt
    ctx["_faults_planted"] = True
    h = ctx["history"]
    for e in h.entries:
        if e.get("event") == "session_start":
            e["faults"] = copy.deepcopy(faults)
            e["ground_truth"] = gt
            break
    h.rewrite()
    h.log("faults_planted", faults=copy.deepcopy(faults), ground_truth=gt)


def open_session(reg, args):
    args = copy.deepcopy(args or {})
    application = args.get("application")
    objective = args.get("objective")
    condition = args.get("condition")
    test = bool(args.get("test"))
    if not isinstance(application, str) or not application:
        raise ValueError("application required")
    if condition not in ("control", "perceptual"):
        raise ValueError("condition must be control or perceptual")
    pending = _pending(reg)
    app = _app(application, pending)
    renderer = _renderer(application, pending)
    store = pending.get("store") or ArtifactStore(pending.get("store_root") or os.path.join(os.getcwd(), "sessions", "artifacts"))
    root = str(pending.get("session_root") or os.path.join(os.getcwd(), "sessions"))
    # Mint the session id from what is already on disk, not
    # from an in-memory counter alone, so a fresh process
    # cannot reuse an earlier session's id. Both the session
    # root and the artifact store root are checked, and the
    # candidate moves forward until it names neither an
    # existing session directory nor an existing artifact.
    def _on_disk_ids():
        ids = set()
        for base in (root, getattr(store, "root", None)):
            if base and os.path.isdir(str(base)):
                try:
                    ids.update(os.listdir(str(base)))
                except OSError:
                    pass
        try:
            ids.update((getattr(reg, "sessions", {}) or {}).keys())
        except Exception:
            pass
        return ids

    def _minted_id():
        prefix = "%s-%s-" % (application, condition)
        existing = _on_disk_ids()
        highest = 0
        try:
            highest = max(highest, int(getattr(reg, "_session_counter", 0) or 0))
        except Exception:
            pass
        for name in existing:
            if isinstance(name, str) and name.startswith(prefix):
                suffix = name[len(prefix):]
                if suffix.isdigit():
                    highest = max(highest, int(suffix))
        number = highest + 1
        while "%s-%s-%d" % (application, condition, number) in existing:
            number += 1
        try:
            reg._session_counter = number
        except Exception:
            pass
        return "%s-%s-%d" % (application, condition, number)

    sid = pending.get("session_id")
    if not sid or sid in _on_disk_ids():
        sid = _minted_id()
    sdir = os.path.join(root, sid)
    aid = pending.get("artifact_id") or sid
    if aid != sid:
        # An explicit artifact id that already names an
        # artifact on disk would append this session's
        # versions to an earlier session's artifact; give
        # this session its own artifact instead.
        try:
            _exists = os.path.exists(store._record_path(aid))
        except Exception:
            _exists = aid in _on_disk_ids()
        if _exists:
            aid = sid
    os.makedirs(sdir, exist_ok=True)
    faults = copy.deepcopy(list(pending.get("faults") or []))
    gt = pending.get("ground_truth")
    if gt is None:
        gt = app.generate_ground_truth(faults) if faults else []
    hist = _History(os.path.join(sdir, "history.jsonl"))
    ctx = {"store": store, "app": app, "renderer": renderer, "session_root": root, "session_dir": sdir, "session_id": sid, "artifact_id": aid, "application": application, "objective": copy.deepcopy(objective), "condition": condition, "test": test, "faults": faults, "ground_truth": gt, "max_iterations": int(pending.get("max_iterations") or 4), "iteration": 0, "current_source": None, "original_source": None, "evidence_summaries": [], "history": hist, "ended_by": None, "assessment": None, "ledger_path": pending.get("ledger_path"), "_faults_planted": bool(faults), "_work_submitted": False, "_artifact_hashes": [], "_revision_pairs": [], "current_version_id": None, "current_structure": None, "current_evidence": None, "current_measurements": None}
    reg.register(sid, ctx)
    _store_entry(store, aid, {}, objective, {"application": application, "session_id": sid})
    hist.log("session_start", session_id=sid, artifact_id=aid, condition=condition, objective=copy.deepcopy(objective), faults=copy.deepcopy(faults), ground_truth=gt, structure_record=None, input_summary=None)
    result = {"session_id": sid, "application": application, "condition": condition, "test": test, "composition_prompt": app.composition_prompt(objective)}
    if condition == "control":
        result["information_rules"] = "Control condition: you will receive the structure record and the artifact as written. You will not receive evidence about the render, and you cannot perceive the render in this condition."
    hist.log("tool_call", tool="sensoryne_open_session", arguments=args)
    hist.log("tool_result", tool="sensoryne_open_session", session_id=sid, condition=condition, test=test)
    return result


def submit_work(ctx, artifact):
    if ctx.get("_work_submitted"):
        raise ValueError("work already submitted")
    app = ctx["app"]
    h = ctx["history"]
    # A work may arrive as JSON text from a model client;
    # extract its object the same way as the decision path
    # and continue with the resulting dict. A string that
    # carries no parseable object is left as-is, so
    # validation still fails on it as it does now.
    if isinstance(artifact, str):
        try:
            artifact = decision_module.parse_artifact(artifact)
        except Exception:
            pass
    h.log("tool_call", tool="sensoryne_submit_work", session_id=ctx["session_id"], artifact=artifact)
    errors = app.validate(artifact)
    if errors:
        result = {"valid": False, "session_id": ctx["session_id"], "errors": list(errors), "retry_prompt": app.composition_retry_prompt(ctx.get("objective"), errors, artifact)}
        result.update(_shape_fields(ctx))
        h.log("tool_result", tool="sensoryne_submit_work", session_id=ctx["session_id"], valid=False, errors=list(errors))
        return result
    ctx["original_source"] = copy.deepcopy(artifact)
    ctx["current_source"] = copy.deepcopy(artifact)
    ctx["_work_submitted"] = True
    ctx["_artifact_hashes"] = [_hash(artifact)]
    ctx["_revision_pairs"] = []
    _update_src(ctx["store"], ctx["artifact_id"], artifact, ctx.get("objective"))
    _plant(ctx)
    structure = _structure(ctx, artifact)
    start = next((e for e in h.entries if e.get("event") == "session_start"), None)
    if start is not None:
        start["structure_record"] = copy.deepcopy(structure)
        start["input_summary"] = copy.deepcopy(app.input_summary(artifact))
        h.rewrite()
    _rec, ev, structure, _delta = _render_perceive(ctx)
    ins = app.input_summary(artifact)
    if ctx.get("condition") == "perceptual":
        result = {"valid": True, "session_id": ctx["session_id"], "iteration": ctx["iteration"], "evidence": ev, "structure_record": structure, "input_summary": ins, "presentation": _presentation(ctx, ev, ins, structure)}
    else:
        result = {"valid": True, "session_id": ctx["session_id"], "iteration": ctx["iteration"], "structure_record": structure, "artifact": copy.deepcopy(ctx["current_source"]), "perception": "perception_skipped"}
    result.update(_shape_fields(ctx))
    h.log("tool_result", tool="sensoryne_submit_work", session_id=ctx["session_id"], valid=True, iteration=ctx["iteration"])
    # The stored session record keeps the complete evidence
    # (logged in _render_perceive); only the copy handed back
    # through this return is shaped.
    return shape_document(result)


def _apply_atomic(ctx, revisions, log=True):
    app = ctx["app"]
    h = ctx["history"]
    applied = []
    rejected = []
    seen = set(ctx.get("_artifact_hashes") or [])
    pairs = set(tuple(p) for p in ctx.get("_revision_pairs") or [])
    for rev in revisions:
        it = ctx.get("iteration", 0)
        if not isinstance(rev, dict) or not isinstance(rev.get("op"), str) or not rev.get("op"):
            err = "revision must be an object with a non-empty op"
            rejected.append({"revision": rev, "error": err})
            if log:
                h.log("revision_rejected", iteration=it, revision=rev, error=err)
            continue
        before = copy.deepcopy(ctx["current_source"])
        bh = _hash(before)
        try:
            cand = app.apply_revision(copy.deepcopy(before), copy.deepcopy(rev))
            if cand is None:
                raise ValueError("revision engine returned no artifact")
            errs = list(app.validate(cand) or [])
            if errs:
                raise ValueError("resulting artifact failed validation: %s" % "; ".join(str(e) for e in errs))
        except Exception as exc:
            ctx["current_source"] = before
            rejected.append({"revision": rev, "error": str(exc)})
            if log:
                h.log("revision_rejected", iteration=it, revision=rev, error=str(exc))
            continue
        ch = _hash(cand)
        key = _canon(rev)
        if ch == bh:
            err = "revision is a no-op: the resulting artifact is unchanged"
            rejected.append({"revision": rev, "error": err})
            ctx["current_source"] = before
            if log:
                h.log("revision_rejected", iteration=it, revision=rev, error=err)
            continue
        pair = (key, ch)
        if pair in pairs:
            err = "revision has already been applied with this result"
            rejected.append({"revision": rev, "error": err})
            ctx["current_source"] = before
            if log:
                h.log("revision_rejected", iteration=it, revision=rev, error=err)
            continue
        ctx["current_source"] = copy.deepcopy(cand)
        seen.add(ch)
        pairs.add(pair)
        applied.append(rev)
        if log:
            h.log("revision_applied", iteration=it, revision=rev)
    ctx["_artifact_hashes"] = sorted(seen)
    ctx["_revision_pairs"] = [list(p) for p in sorted(pairs)]
    return applied, rejected


def _well_formed(d):
    return isinstance(d, dict) and isinstance(d.get("faults"), list) and isinstance(d.get("revisions"), list) and isinstance(d.get("complete"), bool) and isinstance(d.get("assessment"), str)


def submit_decision(ctx, decision):
    h = ctx["history"]
    if not ctx.get("_work_submitted"):
        raise ValueError("no work submitted")
    h.log("tool_call", tool="sensoryne_submit_decision", session_id=ctx["session_id"], decision=decision)
    try:
        # A decision may arrive as JSON text from a model
        # client; parse it the same way as the object form
        # below and continue with the resulting dict. A
        # genuinely unparseable string still ends the session
        # as decision_unparseable through the handler below.
        if isinstance(decision, str):
            decision = decision_module.parse_decision(decision)
        parsed = decision_module.parse_decision(json.dumps(decision))
        if not _well_formed(parsed):
            raise ValueError("decision fields have the wrong types")
    except Exception as exc:
        h.log("decision_unparseable", iteration=ctx.get("iteration", 0), error=str(exc))
        _end(ctx, "decision_unparseable", None)
        h.log("tool_result", tool="sensoryne_submit_decision", session_id=ctx["session_id"], ended_by="decision_unparseable")
        unparseable_result = {"session_id": ctx["session_id"], "ended_by": "decision_unparseable", "error": str(exc)}
        unparseable_result.update(_shape_fields(ctx))
        return unparseable_result
    h.log("decision", iteration=ctx.get("iteration", 0), decision=parsed)
    revisions = parsed.get("revisions") or []
    applied, rejected = _apply_atomic(ctx, revisions)
    structure = _structure(ctx, ctx["current_source"])
    ev = None
    if applied:
        _rec, ev, structure, _d = _render_perceive(ctx)
    else:
        for e in reversed(h.entries):
            if e.get("event") == "perception":
                ev = e.get("evidence")
                break
    sofar = decision_module.build_session_so_far(h.entries, ctx.get("evidence_summaries") or [])
    if ctx.get("condition") == "perceptual":
        result = {"session_id": ctx["session_id"], "iteration": ctx["iteration"], "evidence": ev, "structure_record": copy.deepcopy(structure), "artifact": copy.deepcopy(ctx["current_source"]), "applied": applied, "rejected": rejected, "session_so_far": sofar}
    else:
        result = {"session_id": ctx["session_id"], "iteration": ctx["iteration"], "structure_record": copy.deepcopy(structure), "artifact": copy.deepcopy(ctx["current_source"]), "perception": "perception_skipped", "applied": applied, "rejected": rejected, "session_so_far": sofar}
    result.update(_shape_fields(ctx))
    if parsed.get("complete") is True:
        _end(ctx, "model_complete", parsed.get("assessment"))
        result["ended_by"] = "model_complete"
        result["assessment"] = parsed.get("assessment")
    elif not revisions:
        _end(ctx, "model_no_revisions", parsed.get("assessment"))
        result["ended_by"] = "model_no_revisions"
        result["assessment"] = parsed.get("assessment")
    elif int(ctx.get("iteration", 0)) >= int(ctx.get("max_iterations") or 4):
        _end(ctx, "iteration_limit", parsed.get("assessment"))
        result["ended_by"] = "iteration_limit"
    h.log("tool_result", tool="sensoryne_submit_decision", session_id=ctx["session_id"], iteration=ctx["iteration"], ended_by=ctx.get("ended_by"))
    # The stored session record keeps the complete evidence;
    # only the copy handed back through this return is shaped.
    return shape_document(result)


def probe(ctx, decision):
    """Hypothetical revision probe.

    Returns exactly what a submission of the hypothetical
    artifact returns in this condition, with probe true at the
    top level. In the perceptual condition that is the
    evidence document, carrying a delta against the current
    version. In the control condition it is the control
    submission shape, with no evidence and no delta. A probe
    creates no version and cannot end a session.
    """
    h = ctx["history"]
    if not ctx.get("_work_submitted"):
        raise ValueError("no work submitted")
    h.log("tool_call", tool="sensoryne_probe", session_id=ctx["session_id"], decision=decision)
    # A decision may arrive as JSON text from a model client;
    # extract its object the same way and continue with the
    # resulting dict. An unparseable string is left as-is, so
    # the revisions-list check below still answers with its
    # errors list.
    if isinstance(decision, str):
        try:
            decision = decision_module.extract_json_object(decision)
        except Exception:
            pass
    if not isinstance(decision, dict) or not isinstance(decision.get("revisions"), list):
        result = {"probe": True, "valid": False, "errors": ["probe decision must carry a revisions list"]}
        result.update(_shape_fields(ctx))
        h.log("tool_result", tool="sensoryne_probe", session_id=ctx["session_id"], valid=False)
        return result
    app = ctx["app"]
    hypo = copy.deepcopy(ctx["current_source"])
    errors = []
    rejected = []
    for rev in decision.get("revisions") or []:
        try:
            cand = app.apply_revision(copy.deepcopy(hypo), copy.deepcopy(rev))
            errs = list(app.validate(cand) or [])
            if errs:
                raise ValueError("resulting artifact failed validation: %s" % "; ".join(str(e) for e in errs))
            hypo = cand
        except Exception as exc:
            errors.append(str(exc))
            rejected.append({"revision": rev, "error": str(exc)})
    if errors:
        result = {"probe": True, "valid": False, "errors": errors, "rejected": rejected}
        result.update(_shape_fields(ctx))
        h.log("tool_result", tool="sensoryne_probe", session_id=ctx["session_id"], valid=False, errors=errors)
        return result
    out = os.path.join(ctx["session_dir"], "probe_render_%d" % (int(ctx.get("iteration", 0)) + 1))
    rec = ctx["renderer"](copy.deepcopy(hypo), out, copy.deepcopy(ctx.get("faults") or []))
    if not isinstance(rec, dict):
        rec = {"path": str(rec)}
    rec.setdefault("path", out)
    prec = copy.deepcopy(rec)
    prec["_artifact"] = copy.deepcopy(hypo)
    prec["objective"] = copy.deepcopy(ctx.get("objective"))
    structure = _structure(ctx, hypo)
    inv = _invariants(ctx, hypo, rec, None)
    rel = _reliability(ctx)
    cov = _coverage(ctx, hypo, rec, None)
    if ctx.get("condition") == "perceptual":
        ev = app.perceive(prec)
        _comp = None
        _comp_computed = False
        if getattr(app, "NAME", None) == "music" or ctx.get("application") == "music":
            _comp = _comparison(ctx, hypo, rec, ev)
            _comp_computed = True
            if _comp is not None:
                rec["comparison"] = _comp
                ev["comparison"] = _comp
        ev["invariants"] = _invariants(ctx, hypo, rec, ev)
        ev["reliability"] = rel
        ev["objective_coverage"] = _coverage(ctx, hypo, rec, ev)
        if not _comp_computed:
            _comp = _comparison(ctx, hypo, rec, ev)
        if _comp is not None:
            ev["comparison"] = _comp
        next_id = "v%d-probe" % (_version_number(ctx.get("current_version_id")) + 1)
        ev["delta"] = _rooted_measurements(compute_delta(ctx.get("current_version_id") or "v1", next_id, ctx.get("current_measurements") or {}, ev.get("measurements", {}), ctx.get("current_structure") or {}, structure))
        ev["probe"] = True
        ev["structure_record"] = copy.deepcopy(structure)
        ev["hypothetical_artifact"] = copy.deepcopy(hypo)
        result = ev
    else:
        result = {"probe": True, "structure_record": copy.deepcopy(structure), "artifact": copy.deepcopy(hypo), "perception": "perception_skipped", "invariants": inv, "reliability": rel, "objective_coverage": cov}
    result.update(_shape_fields(ctx))
    h.log("tool_result", tool="sensoryne_probe", session_id=ctx["session_id"], probe=True, valid=True)
    # A probe stores nothing, so there is no stored copy to
    # keep whole: the returned hypothetical evidence is shaped
    # as a bare evidence document, stating inside itself that
    # it is a shaped copy and where the complete evidence
    # lives (the current version's session record).
    if ctx.get("condition") == "perceptual":
        return shape_evidence(result)
    return shape_document(result)


def resume_session(reg, session_id):
    """Resume a session from its record on disk.

    The session is located using the same roots open_session
    resolves ( the pending store and session_root ), its
    history.jsonl is read, and a session that ended as
    model_complete is refused. Every other state is resumable:
    no session_end at all, or an error ending such as
    decision_unparseable. The live session state is rebuilt
    from the record and registered under the existing id by
    the same registration open_session performs, so the other
    working tools operate on it unchanged. A session that is
    already live in this process is returned as it stands:
    it is not re-registered and, unless it had ended by an
    error and must be made continuable again, no event is
    logged for it.
    """
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id required")
    pending = _pending(reg)
    store = pending.get("store") or ArtifactStore(pending.get("store_root") or os.path.join(os.getcwd(), "sessions", "artifacts"))
    root = str(pending.get("session_root") or os.path.join(os.getcwd(), "sessions"))
    sdir = os.path.join(root, session_id)
    hist_path = os.path.join(sdir, "history.jsonl")
    if not os.path.isdir(sdir) or not os.path.exists(hist_path):
        raise ValueError("unknown session_id: %r" % (session_id,))
    entries = []
    try:
        with open(hist_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as exc:
        raise ValueError("unknown session_id: %r" % (session_id,)) from exc
    start = next((e for e in entries if e.get("event") == "session_start"), None)
    if start is None:
        raise ValueError("cannot resume session %r: no session_start in its record" % (session_id,))
    ends = [e for e in entries if e.get("event") == "session_end"]
    prior_ended_by = ends[-1].get("ended_by") if ends else None
    if prior_ended_by == "model_complete":
        raise ValueError("cannot resume session %r: it ended as model_complete" % (session_id,))
    sessions_map = getattr(reg, "sessions", None)
    live = sessions_map.get(session_id) if isinstance(sessions_map, dict) else None
    if live is not None:
        hist_live = live.get("history")
        entries_live = getattr(hist_live, "entries", []) if hist_live is not None else []
        if live.get("ended_by") is not None:
            live["ended_by"] = None
            live["assessment"] = None
            if hist_live is not None:
                hist_live.log("session_resumed", session_id=session_id, application=live.get("application"), version_resumed_at=live.get("current_version_id"), version_id=live.get("current_version_id"), prior_ended_by=prior_ended_by, iteration=live.get("iteration", 0))
        sofar_live = decision_module.build_session_so_far(entries_live, live.get("evidence_summaries") or [])
        result_live = {"session_id": session_id, "application": live.get("application"), "version_resumed_at": live.get("current_version_id"), "version_id": live.get("current_version_id"), "iteration": live.get("iteration", 0), "condition": live.get("condition"), "test": live.get("test"), "prior_ended_by": prior_ended_by, "session_so_far": sofar_live}
        result_live.update(_shape_fields(live))
        return result_live
    artifact_id = start.get("artifact_id") or session_id
    objective = copy.deepcopy(start.get("objective"))
    condition = start.get("condition")
    test_flag = start.get("test")
    application = start.get("application")
    record = None
    try:
        record = store.load(artifact_id)
    except Exception:
        record = None
    if not application and isinstance(record, dict):
        meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else None
        if meta and meta.get("application"):
            application = meta.get("application")
    for e in entries:
        args_e = e.get("arguments") if isinstance(e.get("arguments"), dict) else None
        if e.get("tool") == "sensoryne_open_session" and isinstance(args_e, dict):
            if not application:
                application = args_e.get("application")
            if condition not in ("control", "perceptual"):
                condition = args_e.get("condition")
            if test_flag is None and "test" in args_e:
                test_flag = args_e.get("test")
    for e in entries:
        if e.get("tool") == "sensoryne_open_session" and e.get("event") == "tool_result":
            if condition not in ("control", "perceptual"):
                condition = e.get("condition")
            if test_flag is None and "test" in e:
                test_flag = e.get("test")
    if not application:
        raise ValueError("cannot resume session %r: application is not recorded" % (session_id,))
    if condition not in ("control", "perceptual"):
        condition = "perceptual"
    if test_flag is None:
        test_flag = False
    app = _app(application, pending)
    renderer = _renderer(application, pending)
    faults = copy.deepcopy(list(start.get("faults") or []))
    gt = copy.deepcopy(start.get("ground_truth"))
    if gt is None:
        try:
            gt = app.generate_ground_truth(faults) if faults else []
        except Exception:
            gt = []
    versions = list(record.get("versions") or []) if isinstance(record, dict) else []
    latest_version_id = versions[-1].get("version_id") if versions and isinstance(versions[-1], dict) else None
    current_source = None
    original_source = None
    version_hashes = []
    if versions:
        try:
            original_source = store.load_version_source(artifact_id, versions[0].get("version_id"))
        except Exception:
            original_source = None
        if latest_version_id:
            try:
                current_source = store.load_version_source(artifact_id, latest_version_id)
            except Exception:
                current_source = None
        for v in versions:
            try:
                version_hashes.append(_hash(store.load_version_source(artifact_id, v.get("version_id"))))
            except Exception:
                pass
    if record is None:
        try:
            _store_entry(store, artifact_id, current_source or {}, objective, {"application": application, "session_id": session_id})
        except Exception:
            pass
    iterations = []
    for e in entries:
        for key in ("iteration", "iterations"):
            val = e.get(key)
            if isinstance(val, int) and not isinstance(val, bool):
                iterations.append(val)
    iteration = max(iterations) if iterations else 0
    renders = [e for e in entries if e.get("event") == "render"]
    if renders and iteration == 0:
        iteration = len(renders)
    perceptions = [e for e in entries if e.get("event") == "perception"]
    work_submitted = bool(renders or perceptions or any(e.get("event") == "perception_skipped" for e in entries))
    last_structure = None
    for e in reversed(entries):
        if e.get("event") in ("render", "perception", "perception_skipped") and e.get("structure_record") is not None:
            last_structure = copy.deepcopy(e.get("structure_record"))
            break
    last_evidence = copy.deepcopy(perceptions[-1].get("evidence")) if perceptions else None
    evidence_summaries = [_summary(e.get("evidence")) for e in perceptions]
    hist = _History(hist_path)
    hist.entries = entries
    if original_source is None and work_submitted and current_source is not None:
        original_source = copy.deepcopy(current_source)
    ctx = {"store": store, "app": app, "renderer": renderer, "session_root": root, "session_dir": sdir, "session_id": session_id, "artifact_id": artifact_id, "application": application, "objective": objective, "condition": condition, "test": bool(test_flag), "faults": faults, "ground_truth": gt, "max_iterations": int(pending.get("max_iterations") or 4), "iteration": int(iteration), "current_source": copy.deepcopy(current_source) if current_source is not None else None, "original_source": copy.deepcopy(original_source) if original_source is not None else None, "evidence_summaries": evidence_summaries, "history": hist, "ended_by": None, "assessment": None, "ledger_path": pending.get("ledger_path"), "_faults_planted": bool(faults) or any(e.get("event") == "faults_planted" for e in entries), "_work_submitted": bool(work_submitted), "_artifact_hashes": sorted(set(version_hashes)), "_revision_pairs": [], "current_version_id": latest_version_id, "current_structure": last_structure, "current_evidence": last_evidence, "current_measurements": copy.deepcopy(last_evidence.get("measurements", {})) if isinstance(last_evidence, dict) else None}
    if ctx["current_structure"] is None and ctx["current_source"] is not None:
        try:
            ctx["current_structure"] = _structure(ctx, ctx["current_source"])
        except Exception:
            ctx["current_structure"] = None
    revision_pairs = []
    replay_hashes = []
    if original_source is not None:
        hypo = copy.deepcopy(original_source)
        replay_hashes = [_hash(hypo)]
        for e in entries:
            if e.get("event") != "revision_applied":
                continue
            rev = e.get("revision")
            try:
                hypo = app.apply_revision(copy.deepcopy(hypo), copy.deepcopy(rev))
            except Exception:
                break
            ch = _hash(hypo)
            revision_pairs.append([_canon(rev), ch])
            replay_hashes.append(ch)
    ctx["_revision_pairs"] = revision_pairs
    ctx["_artifact_hashes"] = sorted(set(list(ctx["_artifact_hashes"]) + replay_hashes))
    if hasattr(reg, "register"):
        reg.register(session_id, ctx)
    else:
        reg.sessions[session_id] = ctx
    hist.log("session_resumed", session_id=session_id, application=application, version_resumed_at=latest_version_id, version_id=latest_version_id, prior_ended_by=prior_ended_by, iteration=int(iteration))
    sofar = decision_module.build_session_so_far(hist.entries, evidence_summaries)
    result = {"session_id": session_id, "application": application, "version_resumed_at": latest_version_id, "version_id": latest_version_id, "iteration": int(iteration), "condition": condition, "test": bool(test_flag), "prior_ended_by": prior_ended_by, "session_so_far": sofar}
    result.update(_shape_fields(ctx))
    return result


def recall_memory(reg, application, limit):
    if not isinstance(application, str) or not application:
        raise ValueError("application required")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    pending = _pending(reg)
    path = _memory_path(pending, application)
    if not os.path.exists(path):
        return {"application": application, "entries": []}
    entries = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.reverse()
    return {"application": application, "entries": entries[:limit]}


def finish_session(ctx):
    h = ctx["history"]
    h.log("tool_call", tool="sensoryne_finish_session", session_id=ctx["session_id"])
    summary = _end(ctx, "model_complete", ctx.get("assessment"))
    h.log("tool_result", tool="sensoryne_finish_session", session_id=ctx["session_id"], ended_by="model_complete")
    summary["session_id"] = ctx["session_id"]
    return summary
