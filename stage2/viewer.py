"""stage2/viewer.py - Sensoryne replay viewer server.

Serves the viewer and session files on localhost only, using
the standard library. ``python3 -m stage2.viewer``. The viewer
displays evidence and decisions exactly as recorded; it
detects, and judges nothing, and needs no API key. Spend and
call counts are read from each run's own spend.jsonl inside
its run directory. Each session's structure record and each
stored delta are read from the session_start, perception, and
render entries of its history.jsonl.
"""

import argparse
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INDEX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "viewer", "index.html"
)


def _sessions_root():
    candidate = os.path.join(_PROJECT_ROOT, "sessions")
    return candidate if os.path.isdir(candidate) else os.path.join(os.getcwd(), "sessions")


def _is_within(root, target):
    root = os.path.realpath(root)
    target = os.path.realpath(target)
    return target == root or target.startswith(root + os.sep)


def _read_spend_file(spend_path):
    spend_by_session = {}
    calls_by_session = {}
    if not os.path.exists(spend_path):
        return spend_by_session, calls_by_session
    with open(spend_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            sid = entry.get("session_id")
            spend_by_session[sid] = spend_by_session.get(sid, 0.0) + float(
                entry.get("cost_usd", 0.0)
            )
            calls_by_session[sid] = calls_by_session.get(sid, 0) + 1
    return spend_by_session, calls_by_session


def _find_run_dir(sessions_root, dirpath):
    root = os.path.realpath(sessions_root)
    current = os.path.realpath(dirpath)
    while True:
        if os.path.exists(os.path.join(current, "spend.jsonl")):
            return current
        if current == root:
            return root
        parent = os.path.dirname(current)
        if parent == current:
            return root
        current = parent


def _discover_sessions(sessions_root):
    sessions = []
    if not os.path.isdir(sessions_root):
        return sessions
    for dirpath, _dirs, files in os.walk(sessions_root):
        if "history.jsonl" not in files:
            continue
        history_path = os.path.join(dirpath, "history.jsonl")
        session_id = os.path.basename(dirpath)
        events = []
        with open(history_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        start = next((e for e in events if e.get("event") == "session_start"), {})
        structure_record = start.get("structure_record")
        if structure_record is None:
            perception = next(
                (
                    e
                    for e in events
                    if e.get("event") == "perception"
                    and e.get("structure_record") is not None
                ),
                None,
            )
            if perception is not None:
                structure_record = perception.get("structure_record")
        # Delta blocks as stored, keyed by iteration, for the
        # delta view between consecutive versions.
        deltas = {}
        for entry in events:
            if entry.get("event") == "perception":
                evidence = entry.get("evidence") or {}
                if isinstance(evidence, dict) and evidence.get("delta") is not None:
                    deltas[entry.get("iteration")] = evidence.get("delta")
            elif entry.get("event") == "render":
                info = entry.get("render_info") or {}
                if isinstance(info, dict) and info.get("delta") is not None:
                    deltas.setdefault(entry.get("iteration"), info.get("delta"))
        run_dir = _find_run_dir(sessions_root, dirpath)
        spend_by_session, calls_by_session = _read_spend_file(
            os.path.join(run_dir, "spend.jsonl")
        )
        run_id = ""
        if os.path.realpath(run_dir) != os.path.realpath(sessions_root):
            run_id = os.path.basename(run_dir)
        rel_dir = os.path.relpath(dirpath, sessions_root).replace(os.sep, "/")
        renders = sorted(
            f
            for f in files
            if f.startswith("render_") and f not in ("render_info",)
        )
        sid = start.get("session_id", session_id)
        sessions.append(
            {
                "session_id": sid,
                "artifact_id": start.get("artifact_id", ""),
                "condition": start.get("condition", ""),
                "objective": start.get("objective", ""),
                "faults": start.get("faults", []),
                "ground_truth": start.get("ground_truth", ""),
                "structure_record": structure_record,
                "deltas": deltas,
                "run_id": run_id,
                "rel_dir": rel_dir,
                "history_url": "/sessions/%s/history.jsonl" % rel_dir,
                "renders": [
                    {"name": r, "url": "/sessions/%s/%s" % (rel_dir, r)}
                    for r in renders
                ],
                "events": events,
                "api_calls": calls_by_session.get(sid, 0),
                "spend_usd": spend_by_session.get(sid, 0.0),
            }
        )
    sessions.sort(key=lambda s: s["session_id"])
    return sessions


class _Handler(BaseHTTPRequestHandler):
    server_version = "Sensoryne/1.0"

    def log_message(self, *args):
        pass

    def _send_file(self, path):
        if not os.path.isfile(path):
            self.send_error(404)
            return
        content_type, _ = mimetypes.guess_type(path)
        self.send_response(200)
        self.send_header(
            "Content-Type", content_type or "application/octet-stream"
        )
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.end_headers()
        with open(path, "rb") as handle:
            chunk = handle.read(65536)
            while chunk:
                self.wfile.write(chunk)
                chunk = handle.read(65536)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in ("/", "/index.html"):
            self._send_file(_INDEX_PATH)
            return
        if path == "/api/sessions":
            body = json.dumps(
                _discover_sessions(_sessions_root())
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/sessions/"):
            rel = path[len("/sessions/"):]
            root = os.path.abspath(_sessions_root())
            target = os.path.abspath(os.path.join(root, rel))
            if not _is_within(root, target):
                self.send_error(403)
                return
            self._send_file(target)
            return
        self.send_error(404)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sensoryne replay viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print("Sensoryne viewer on http://%s:%d/" % (args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
