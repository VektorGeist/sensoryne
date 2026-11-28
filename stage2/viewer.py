"""stage2/viewer.py - Sensoryne replay viewer server.

Serves the viewer and session files on localhost only, using
the standard library. ``python3 -m stage2.viewer``. The viewer
displays evidence and decisions exactly as recorded,
computes nothing of its own, and needs no API key. Spend and
call counts are read from each run's own spend.jsonl inside
its run directory. Each session's structure record and each
stored delta are read from the session_start, perception, and
render entries of its history.jsonl.

The page (stage2/viewer/index.html) requires three server
behaviours beyond the original contract, all implemented here
without changing any session data:

* ``--sessions-root PATH`` (or the SENSORYNE_SESSIONS_ROOT
  environment variable) selects the sessions folder to serve,
  so the proof-run sessions can be viewed without moving
  them. The default remains ./sessions.
* Extensionless render files (render_1, render_1.png for
  typography, robustness/*.wav, frame PNGs) are served with
  a content type inferred from the render kind recorded in
  the session history when the file name has no extension.
* A single trailing .png/.wav suffix may be omitted in a
  /sessions/ URL: /sessions/<dir>/render_1 serves the file
  render_1.png when only that file exists. This is how the
  renderer's own save path is addressed.
* Extensionless session files such as history.jsonl are
  served as application/json.
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
_ICON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "viewer", "sensoryne-icon.webp"
)

# Set by main() from --sessions-root / the environment. Kept as a
# module global so _Handler (which has no config of its own) can
# resolve the root the same way _discover_sessions does.
_SESSIONS_ROOT_OVERRIDE = None

# Settings keys live beside the sessions folder: the settings file
# is sensoryne-settings.json in the parent directory of the
# sessions root the viewer was started with. It is created on
# first save and its absence is tolerated. Only booleans are
# ever returned to the page; stored values are never sent back.
_SETTINGS_FILENAME = "sensoryne-settings.json"


def _settings_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(_sessions_root())),
        _SETTINGS_FILENAME,
    )


def _read_settings():
    try:
        with open(_settings_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _settings_status(data):
    return {
        "worker_key_set": bool(data.get("worker_key")),
        "judge_key_set": bool(data.get("judge_key")),
    }


def _write_settings(data):
    path = _settings_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)



def _sessions_root():
    if _SESSIONS_ROOT_OVERRIDE:
        return _SESSIONS_ROOT_OVERRIDE
    env = os.environ.get("SENSORYNE_SESSIONS_ROOT")
    if env:
        return env
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


# Render-kind -> content type, for extensionless render files.
# Derived from the renderer contract in the *-renderer modules:
# audio renders are WAV, frame/image renders are PNG, text
# renders are plain text, and the code render is a directory.
_KIND_CONTENT_TYPES = {
    "audio": "audio/wav",
    "image": "image/png",
    "frames": "image/png",
    "text": "text/plain; charset=utf-8",
    "files": "application/octet-stream",
}


def _render_kind_for(rel_path):
    """Best-effort render kind for a /sessions/ relative path.

    Reads the owning session's history.jsonl render entry whose
    recorded path ends with the requested file name. Returns
    None when no kind can be determined; the caller then falls
    back to the extension guess.
    """
    parts = rel_path.split("/")
    if len(parts) < 2:
        return None
    target_name = parts[-1]
    if len(parts) >= 3 and parts[-2] == "robustness":
        session_dir = os.path.join(_sessions_root(), parts[0])
    else:
        session_dir = os.path.join(_sessions_root(), *parts[:-1])
    history_path = os.path.join(session_dir, "history.jsonl")
    if not os.path.isfile(history_path):
        return None
    try:
        with open(history_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if entry.get("event") != "render":
                    continue
                info = entry.get("render_info") or {}
                if not isinstance(info, dict):
                    continue
                recorded = str(info.get("path", "")).replace("\\", "/")
                if recorded.endswith("/" + target_name) or recorded == target_name:
                    return info.get("kind")
                # Frame files: the recorded path is the stem.
                if info.get("kind") == "frames" and target_name.startswith(
                    os.path.basename(recorded) + "_frame_"
                ):
                    return "frames"
    except OSError:
        return None
    return None


def _content_type_for(path, rel_path):
    # history.jsonl and other .jsonl records are JSON.
    if path.endswith(".jsonl") or path.endswith(".json"):
        return "application/json"
    content_type, _ = mimetypes.guess_type(path)
    if content_type:
        return content_type
    # Extensionless render file: use the recorded render kind.
    kind = _render_kind_for(rel_path)
    if kind in _KIND_CONTENT_TYPES:
        return _KIND_CONTENT_TYPES[kind]
    # Robustness WAVs and other extensioned audio already have a
    # guessed type; anything else stays octet-stream.
    return "application/octet-stream"


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

    def _send_file(self, path, rel_path=None):
        if not os.path.isfile(path):
            self.send_error(404)
            return
        content_type = _content_type_for(path, rel_path or "")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.end_headers()
        with open(path, "rb") as handle:
            chunk = handle.read(65536)
            while chunk:
                self.wfile.write(chunk)
                chunk = handle.read(65536)

    def _send_icon(self):
        if not os.path.isfile(_ICON_PATH):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/webp")
        self.send_header("Content-Length", str(os.path.getsize(_ICON_PATH)))
        self.end_headers()
        with open(_ICON_PATH, "rb") as handle:
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
        if path in ("/sensoryne-icon.webp", "/viewer/sensoryne-icon.webp"):
            self._send_icon()
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
        if path == "/api/settings":
            body = json.dumps(_settings_status(_read_settings())).encode("utf-8")
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
            if os.path.isfile(target):
                self._send_file(target, rel)
                return
            # One optional trailing media suffix: renderers save
            # e.g. render_1.png for an extensionless render_1
            # path, and wav renders may be addressed either way.
            for suffix in (".png", ".wav"):
                candidate = target + suffix
                if not _is_within(root, candidate):
                    continue
                if os.path.isfile(candidate):
                    self._send_file(candidate, rel)
                    return
            self.send_error(404)
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path != "/api/settings":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        settings = _read_settings()
        changed = False
        # A non-empty key value stores it; an empty string or an
        # absent field leaves that key unchanged.
        if isinstance(data.get("worker_key"), str) and data["worker_key"]:
            if settings.get("worker_key") != data["worker_key"]:
                settings["worker_key"] = data["worker_key"]
                changed = True
        if isinstance(data.get("judge_key"), str) and data["judge_key"]:
            if settings.get("judge_key") != data["judge_key"]:
                settings["judge_key"] = data["judge_key"]
                changed = True
        if data.get("remove") in ("worker", "judge"):
            field = "worker_key" if data["remove"] == "worker" else "judge_key"
            if field in settings:
                del settings[field]
                changed = True
        if changed:
            _write_settings(settings)
        body = json.dumps(_settings_status(settings)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv=None):
    global _SESSIONS_ROOT_OVERRIDE
    parser = argparse.ArgumentParser(description="Sensoryne replay viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--sessions-root",
        default=None,
        help="sessions folder to serve (default: ./sessions)",
    )
    args = parser.parse_args(argv)
    if args.sessions_root:
        _SESSIONS_ROOT_OVERRIDE = os.path.abspath(args.sessions_root)
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print("Sensoryne viewer on http://%s:%d/ (sessions: %s)" % (args.host, args.port, _sessions_root()))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
