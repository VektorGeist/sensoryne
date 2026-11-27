"""stage2/apps/code/instrument.py - execution instrument.

Runs the delivered program in a closed directory, once per
stated test, with a 10 second limit per test.  Records exit
status, standard output, standard error, run time, whether the
limit was hit, and whether the output matches the expected
output exactly.  It reports those facts; it does not judge.
"""

import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import time

METHOD = (
    "delivered files copied into a closed temporary directory; "
    "the entry file (main.py if present, otherwise the first "
    "file) is run once per stated test with the test's input "
    "fed on standard input, in a subprocess whose socket module "
    "is disabled before the program starts; 10 second limit per "
    "test; standard output compared to the expected output "
    "exactly, character for character"
)

TIME_LIMIT_SECONDS = 10

_LAUNCHER = (
    "import runpy, socket, sys\n"
    "class _BlockedSocket(socket.socket):\n"
    "    def __init__(self, *a, **k):\n"
    "        raise OSError('network access disabled by the instrument')\n"
    "socket.socket = _BlockedSocket\n"
    "socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(OSError('network access disabled by the instrument'))\n"
    "sys.argv = ['main.py']\n"
    "runpy.run_path('main.py', run_name='__main__')\n"
)


def _read_delivered_files(path):
    files = {}
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full) and name != "_instrument_launcher.py":
                with open(full, "r", encoding="utf-8", errors="replace") as handle:
                    files[name] = handle.read()
    elif os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            files[os.path.basename(path)] = handle.read()
    return files


def _entry_file(files):
    if "main.py" in files:
        return "main.py"
    return sorted(files)[0] if files else None


def run_tests(files, tests):
    """Run the delivered program once per test. Returns results."""
    results = []
    entry = _entry_file(files)
    for test in tests or []:
        test_input = test.get("input", "") if isinstance(test, dict) else ""
        expected = test.get("expected") if isinstance(test, dict) else None
        name = test.get("name", "") if isinstance(test, dict) else ""
        if entry is None:
            results.append(
                {
                    "name": name,
                    "input": test_input,
                    "expected": expected,
                    "entry_file": None,
                    "exit_status": None,
                    "stdout": "",
                    "stderr": "no delivered files to run",
                    "runtime_seconds": 0.0,
                    "limit_hit": False,
                    "matches_expected": False,
                }
            )
            continue
        with tempfile.TemporaryDirectory(prefix="sensoryne_closed_") as closed_dir:
            for file_name, text in files.items():
                target = os.path.join(closed_dir, file_name)
                os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write(text)
            with open(os.path.join(closed_dir, "_instrument_launcher.py"), "w", encoding="utf-8") as handle:
                handle.write(_LAUNCHER)
            started = time.perf_counter()
            limit_hit = False
            exit_status = None
            stdout = ""
            stderr = ""
            try:
                completed = subprocess.run(
                    [sys.executable, "_instrument_launcher.py"],
                    input=str(test_input),
                    capture_output=True,
                    text=True,
                    timeout=TIME_LIMIT_SECONDS,
                    cwd=closed_dir,
                    env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
                )
                exit_status = completed.returncode
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
            except subprocess.TimeoutExpired as exc:
                limit_hit = True
                stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            runtime = time.perf_counter() - started
        results.append(
            {
                "name": name,
                "input": test_input,
                "expected": expected,
                "entry_file": entry,
                "exit_status": exit_status,
                "stdout": stdout,
                "stderr": stderr,
                "runtime_seconds": round(float(runtime), 6),
                "limit_hit": limit_hit,
                "matches_expected": bool(expected is not None and stdout == str(expected)),
            }
        )
    return results


def analyze_delivered(path, tests=None):
    files = _read_delivered_files(path)
    results = run_tests(files, tests or [])
    matched = sum(1 for item in results if item["matches_expected"])
    if results:
        summary = (
            "%d of %d stated tests matched the expected output exactly. "
            "Entry file: %s. Per test, exit status, standard output, "
            "standard error, run time, limit hit, and exact match are "
            "recorded."
            % (matched, len(results), results[0].get("entry_file") or "none")
        )
    else:
        summary = "0 stated tests were run; no delivered program was executed."
    measurements = {
        "instrument": "execution",
        "method": METHOD,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "entry_file": _entry_file(files),
        "delivered_files": sorted(files),
        "test_count": len(results),
        "matched_count": matched,
        "results": results,
    }
    full = {
        "instrument": "execution",
        "method": METHOD,
        "render_path": str(path),
        "delivered_files": files,
        "measurements": measurements,
        "interpretation": None,
    }
    return {
        "summary": summary,
        "measurements": measurements,
        "full": full,
        "render_path": str(path),
        "instrument": "execution",
        "interpretation": None,
    }


def perceive_delivered(path, tests=None):
    return analyze_delivered(path, tests)
