"""code_renderer.py - composing-side code renderer.

renderer(artifact, out_path, faults): writes the delivered
program as files under out_path (a directory), with delivery
faults applied to the delivered copy only; the written program
is unchanged.

Path containment: delivered file names are model-provided, so
every name is validated before it is used and again
immediately before writing.  Absolute paths, parent
traversal, drive-letter and UNC paths, unsafe components, and
symlink escapes are rejected, and the resolved target must
remain inside the designated render root.
"""

import copy
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

__all__ = ["renderer"]

_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_UNSAFE_CHARS = set('<>:"|?*')
_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _resolve(path):
    try:
        return path.resolve(strict=False)
    except TypeError:
        return path.resolve()


def _is_within(root, target):
    root = _resolve(Path(root))
    target = _resolve(Path(target))
    try:
        return os.path.commonpath([str(root), str(target)]) == str(root)
    except ValueError:
        return False


def _validate_relative_name(name):
    """Validate a delivered file name and return its safe parts.

    Rejects absolute paths, parent traversal, drive-letter and
    UNC paths, mixed-separator traversal, empty/current
    components, and unsafe filename components.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("delivered file name must be a non-empty string: %r" % (name,))
    if "\x00" in name or any(ord(char) < 32 for char in name):
        raise ValueError("unsafe delivered file name: %r" % (name,))
    if name.startswith(("/", "\\")):
        raise ValueError("absolute delivered file paths are rejected: %r" % (name,))
    if _DRIVE_RE.match(name):
        raise ValueError("drive-letter delivered file paths are rejected: %r" % (name,))

    windows = PureWindowsPath(name)
    posix = PurePosixPath(name)
    if windows.drive or windows.is_absolute() or posix.is_absolute():
        raise ValueError("absolute delivered file paths are rejected: %r" % (name,))

    # Treat both separators alike so mixed-separator traversal
    # such as "a/..\\evil" cannot slip past a POSIX-only check.
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        raise ValueError("absolute delivered file paths are rejected: %r" % (name,))

    parts = normalized.split("/")
    for part in parts:
        if part in ("", ".", ".."):
            raise ValueError("unsafe delivered file name component in %r" % (name,))
        if _DRIVE_RE.match(part) or ":" in part:
            raise ValueError("drive-letter delivered file paths are rejected: %r" % (name,))
        if any(char in _UNSAFE_CHARS for char in part):
            raise ValueError("unsafe delivered file name component in %r" % (name,))
        if part != part.strip() or part.endswith(".") or part.endswith(" "):
            raise ValueError("unsafe delivered file name component in %r" % (name,))
        if part.lower() in _RESERVED_NAMES:
            raise ValueError("reserved delivered file name component in %r" % (name,))
    return parts


def _check_no_symlink_components(render_root, parts):
    """Reject if any existing component below the root is a symlink."""
    current = Path(render_root)
    for part in parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink escape rejected in delivered file path")
    return current


def _safe_target(render_root, name):
    """Return the resolved target path for a delivered file name,
    verified to be contained inside render_root."""
    parts = _validate_relative_name(name)
    root = _resolve(Path(render_root))
    _check_no_symlink_components(root, parts)
    candidate = root.joinpath(*parts)
    if candidate.is_symlink():
        raise ValueError("symlink escape rejected in delivered file path: %r" % (name,))
    resolved = _resolve(candidate)
    if resolved == root or not _is_within(root, resolved):
        raise ValueError("delivered file path escapes the render root: %r" % (name,))
    return resolved


def _apply_faults(files, faults):
    files = copy.deepcopy(files)
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_file":
            files = [f for f in files if f.get("name") != fault.get("name")]
        elif kind == "truncate_file":
            for entry in files:
                if entry.get("name") == fault.get("name"):
                    lines = str(entry.get("text", "")).splitlines()
                    count = fault.get("lines")
                    if isinstance(count, int) and count >= 0:
                        entry["text"] = "\n".join(lines[:count]) + ("\n" if lines[:count] else "")
        elif kind == "swap_lines":
            for entry in files:
                if entry.get("name") == fault.get("name"):
                    lines = str(entry.get("text", "")).splitlines()
                    first, second = fault.get("first"), fault.get("second")
                    if (
                        isinstance(first, int) and isinstance(second, int)
                        and 1 <= first <= len(lines) and 1 <= second <= len(lines)
                    ):
                        lines[first - 1], lines[second - 1] = lines[second - 1], lines[first - 1]
                        entry["text"] = "\n".join(lines) + "\n"
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return files


def _structure_record(artifact, delivered):
    return {
        "files": [
            {"name": f.get("name"), "line_count": len(str(f.get("text", "")).splitlines())}
            for f in delivered
        ],
        "purpose": copy.deepcopy(artifact.get("purpose")),
        "tests": copy.deepcopy(artifact.get("tests", [])),
    }


def renderer(artifact, out_path, faults):
    delivered = _apply_faults(artifact.get("files", []), faults)
    render_root = Path(out_path)
    if render_root.exists() and not render_root.is_dir():
        raise ValueError("render path exists and is not a directory: %r" % (str(out_path),))
    if render_root.is_symlink():
        raise ValueError("render path must not be a symlink")
    render_root.mkdir(parents=True, exist_ok=True)
    root = _resolve(render_root)

    for entry in delivered:
        name = entry.get("name")
        # Validate before use, create parents inside the root
        # only, and validate again immediately before writing.
        target = _safe_target(root, name)
        parent = target.parent
        if not _is_within(root, parent):
            raise ValueError("delivered file path escapes the render root: %r" % (name,))
        _check_no_symlink_components(root, _validate_relative_name(name))
        parent.mkdir(parents=True, exist_ok=True)
        target = _safe_target(root, name)
        if target.is_symlink():
            raise ValueError("symlink escape rejected in delivered file path: %r" % (name,))
        if not _is_within(root, _resolve(target.parent)):
            raise ValueError("delivered file path escapes the render root: %r" % (name,))
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(str(entry.get("text", "")))

    return {
        "path": str(out_path),
        "kind": "files",
        "files": [f.get("name") for f in delivered],
        "structure_record": _structure_record(artifact, delivered),
        "tests": copy.deepcopy(artifact.get("tests", [])),
        "faults_applied": copy.deepcopy(list(faults or [])),
    }
