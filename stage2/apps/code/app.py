"""stage2/apps/code/app.py - the code application.

Sensoryne receives the delivered program produced on the
composing side.  Inside Sensoryne this application keeps the
artifact format and validation, revisions, the execution
instrument adapter, the input summary and structure record,
delivery-fault definitions and ground-truth text, and the
prompts.  Delivery faults are definitions recorded here and
applied by the composing side to the delivered copy; the
program Sensoryne receives already carries them.  No revision
may edit the tests or the purpose.

File names are model-provided and are validated as safe
relative paths wherever they are used, including in revision
operations, so a revision cannot introduce a path that
escapes the artifact directory when the delivered copy is
written.
"""

import ast
import copy
import json
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from ...decision import DECISION_SHAPE_TEXT
from .instrument import perceive_delivered

NAME = "code"

REVISION_SHAPE_TEXT = """Revisions are operations applied to the program, in order. References are checked, text must be non-empty, file names must be safe relative paths, the resulting artifact is re-validated before it becomes the new version, and a rejected revision leaves the current program unchanged. No revision may edit the tests or the purpose.
- {"op": "replace_function", "file": "main.py", "function": "solve", "text": "..."}: replace the named function in the named file with the given text.
- {"op": "replace_lines", "file": "main.py", "from": 1, "to": 3, "text": "..."}: replace lines from..to (1-based, inclusive) in the named file with the given text.
- {"op": "add_file", "name": "helper.py", "text": "..."}: add a new file. The name must be a safe relative path: no absolute path, no parent traversal, no drive letter, no unsafe characters.
- {"op": "remove_file", "name": "helper.py"}: remove the named file."""

_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_UNSAFE_CHARS = set('<>:"|?*')
_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _validate_file_name(name):
    """Validate a model-provided file name as a safe relative path."""
    if not isinstance(name, str) or not name:
        raise ValueError("file name must be a non-empty string: %r" % (name,))
    if "\x00" in name or any(ord(char) < 32 for char in name):
        raise ValueError("unsafe file name: %r" % (name,))
    if name.startswith(("/", "\\")):
        raise ValueError("absolute file paths are rejected: %r" % (name,))
    if _DRIVE_RE.match(name):
        raise ValueError("drive-letter file paths are rejected: %r" % (name,))
    windows = PureWindowsPath(name)
    posix = PurePosixPath(name)
    if windows.drive or windows.is_absolute() or posix.is_absolute():
        raise ValueError("absolute file paths are rejected: %r" % (name,))
    parts = name.replace("\\", "/").split("/")
    for part in parts:
        if part in ("", ".", ".."):
            raise ValueError("unsafe file name component in %r" % (name,))
        if _DRIVE_RE.match(part) or ":" in part:
            raise ValueError("drive-letter file paths are rejected: %r" % (name,))
        if any(char in _UNSAFE_CHARS for char in part):
            raise ValueError("unsafe file name component in %r" % (name,))
        if part != part.strip() or part.endswith(".") or part.endswith(" "):
            raise ValueError("unsafe file name component in %r" % (name,))
        if part.lower() in _RESERVED_NAMES:
            raise ValueError("reserved file name component in %r" % (name,))
    if not parts:
        raise ValueError("unsafe file name: %r" % (name,))
    return name


def _safe_target(render_root, name):
    """Resolve a target inside render_root or raise ValueError."""
    _validate_file_name(name)
    root = Path(render_root).resolve()
    parts = name.replace("\\", "/").split("/")
    candidate = root.joinpath(*parts)
    resolved = candidate.resolve()
    if os.path.commonpath([str(root), str(resolved)]) != str(root):
        raise ValueError("file path escapes the render root: %r" % (name,))
    return resolved


def _file_name_errors(name):
    try:
        _validate_file_name(name)
    except ValueError as exc:
        return str(exc)
    return None


def validate(artifact):
    """Structure only: an object with a files list, at least
    one file, every file name a safe relative path, no
    duplicate file names, every file holding text."""
    errors = []
    if not isinstance(artifact, dict) or "files" not in artifact:
        return ["program must be an object with a 'files' list"]
    files = artifact.get("files")
    if not isinstance(files, list) or not files:
        errors.append("program must contain at least one file")
    else:
        seen = set()
        for index, entry in enumerate(files, start=1):
            if not isinstance(entry, dict):
                errors.append("file %d must be an object" % index)
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                errors.append("file %d has no name" % index)
            else:
                name_error = _file_name_errors(name)
                if name_error is not None:
                    errors.append(name_error)
                elif name in seen:
                    errors.append("duplicate file name %r" % name)
                else:
                    seen.add(name)
            if not isinstance(entry.get("text"), str) or not entry["text"].strip():
                errors.append("file %r is empty" % (name,))
    return errors


def _objective_text(objective):
    if isinstance(objective, dict):
        return json.dumps(objective, indent=2)
    return str(objective)


def composition_prompt(objective):
    return """Write a program meeting the objective's purpose. Answer with the artifact JSON only, in this format and nothing else:
{"files": [{"name": "main.py", "text": "..."}], "purpose": "what the program is supposed to do", "tests": [{"name": "...", "input": "...", "expected": "..."}]}

File names must be safe relative paths: no absolute path, no parent traversal, no drive letter, no unsafe characters. The tests are stated by you as part of the work: input fed to the program on standard input, expected output.

Objective:
%s

Answer with one program JSON object in the format {"files": [{"name": "main.py", "text": "..."}], "purpose": "...", "tests": [...]} and nothing else.""" % _objective_text(objective)


def composition_retry_prompt(objective, errors, rejected):
    return """Your previous program did not pass validation. Please provide a corrected program.

Objective:
%s

Failed checks:
%s

Rejected program:
%s

Answer with one corrected program JSON object in the format {"files": [{"name": "main.py", "text": "..."}], "purpose": "...", "tests": [...]} and nothing else.""" % (
        _objective_text(objective),
        "\n".join("- " + str(e) for e in errors),
        json.dumps(rejected, indent=2),
    )


def control_prompt(objective, artifact):
    return """You are evaluating a program you composed. You cannot execute it in this condition; judge from the program alone.

Objective:
%s

Program:
%s

State any faults you find in the program itself, order revisions if needed, and judge whether the work meets the objective.

%s

%s""" % (
        _objective_text(objective),
        json.dumps(artifact, indent=2),
        DECISION_SHAPE_TEXT,
        REVISION_SHAPE_TEXT,
    )


def perceptual_prompt(objective, evidence, input_summary, structure_record):
    return """You are evaluating a program you composed, given evidence from an independent execution instrument about the delivered program, the input summary, and the structure record of what you wrote.

Objective:
%s

INPUT (what was composed): the program exactly as written, including its stated tests and purpose, below.

Input summary:
%s

Structure record (files with line counts, functions and top-level parts in each file by name, purpose, and test list):
%s

OUTPUT (what was delivered): the delivered program, with the instrument's full evidence about it below. The instrument executes the program and records the facts; it does not judge. Judge the delivered program against what was written and the objective.

Evidence:
%s

The evidence's comparison section is Sensoryne's judgment of the delivered program against the objective's checks, and the decision must state how it answers that judgment.

State any faults you find, order revisions if needed, and judge whether the delivered work meets the objective.

%s

%s""" % (
        _objective_text(objective),
        json.dumps(input_summary, indent=2),
        json.dumps(structure_record, indent=2),
        json.dumps(evidence, indent=2),
        DECISION_SHAPE_TEXT,
        REVISION_SHAPE_TEXT,
    )


def input_summary(artifact):
    """The program exactly as written, tests and purpose included."""
    return copy.deepcopy(artifact)


def _top_level_parts(text):
    parts = []
    try:
        tree = ast.parse(text or "")
    except Exception:
        return parts
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parts.append({"name": node.name, "kind": type(node).__name__})
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    parts.append({"name": target.id, "kind": "Assign"})
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            parts.append({"name": elt.id, "kind": "Assign"})
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            parts.append({"name": node.target.id, "kind": "AnnAssign"})
        elif isinstance(node, ast.Import):
            parts.append({"name": node.names[0].name if node.names else "", "kind": "Import"})
        elif isinstance(node, ast.ImportFrom):
            parts.append({"name": node.module or "", "kind": "ImportFrom"})
    return parts


def structure_record(artifact):
    entries = []
    for entry in artifact.get("files", []):
        text = entry.get("text", "")
        entries.append(
            {
                "name": entry.get("name"),
                "line_count": len(text.splitlines()) if isinstance(text, str) else 0,
                "top_level_parts": _top_level_parts(text),
            }
        )
    return {
        "files": entries,
        "file_count": len(entries),
        "purpose": copy.deepcopy(artifact.get("purpose")),
        "tests": copy.deepcopy(artifact.get("tests", [])),
    }


def generate_ground_truth(faults):
    """One ground-truth string per fault, from the definitions."""
    lines = []
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "drop_file":
            lines.append(
                "drop_file: file %r is omitted from the delivered copy, "
                "so a written file does not reach the instrument."
                % fault.get("name")
            )
        elif kind == "truncate_file":
            lines.append(
                "truncate_file: only the first %s lines of file %r are "
                "delivered."
                % (fault.get("lines"), fault.get("name"))
            )
        elif kind == "swap_lines":
            lines.append(
                "swap_lines: lines %s and %s of file %r exchange "
                "positions in the delivered copy."
                % (fault.get("first"), fault.get("second"), fault.get("name"))
            )
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return lines


def plant_faults(artifact):
    """Return the fault set for the given work, deterministically,
    using only defined kinds and only files/lines that exist."""
    files = artifact.get("files", []) if isinstance(artifact, dict) else []
    faults = []
    if files:
        faults.append({"kind": "drop_file", "name": files[0].get("name")})
    multi_line = None
    for entry in files:
        text = entry.get("text", "")
        if isinstance(text, str) and len(text.splitlines()) >= 2:
            multi_line = entry
            break
    if multi_line is not None:
        faults.append(
            {"kind": "truncate_file", "name": multi_line.get("name"), "lines": 1}
        )
        faults.append(
            {
                "kind": "swap_lines",
                "name": multi_line.get("name"),
                "first": 1,
                "second": 2,
            }
        )
    return faults


def perceive(render_record):
    tests = []
    record = render_record.get("structure_record")
    if isinstance(record, dict):
        tests = list(record.get("tests") or [])
    if not tests and isinstance(render_record.get("tests"), list):
        tests = list(render_record["tests"])
    return perceive_delivered(render_record["path"], tests)


def _file_entry(revised, name):
    if not isinstance(name, str) or not name:
        raise ValueError("a file name is required")
    _validate_file_name(name)
    for entry in revised.get("files", []):
        if entry.get("name") == name:
            return entry
    raise ValueError("file %r does not exist" % (name,))


def _require_text(revision):
    text = revision.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return text


def _require_int(revision, key):
    value = revision.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("%s requires an integer %r" % (revision.get("op"), key))
    return value


def _replace_function_in_text(text, function, replacement):
    try:
        tree = ast.parse(text or "")
    except SyntaxError as exc:
        raise ValueError("file does not parse: %s" % exc)
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == function
        ),
        None,
    )
    if node is None:
        raise ValueError("function %r does not exist" % (function,))
    lines = text.splitlines()
    start = node.lineno - 1
    end = node.end_lineno or len(lines)
    lines[start:end] = replacement.splitlines()
    new_text = "\n".join(lines) + "\n"
    try:
        ast.parse(new_text)
    except SyntaxError as exc:
        raise ValueError("replacement does not parse: %s" % exc)
    return new_text


def apply_revision(artifact, revision):
    """Return a revised deep copy; input never modified.

    No revision may edit the tests or the purpose; those fields
    are carried over unchanged.  File names in revisions are
    validated as safe relative paths, and the resulting
    artifact is validated as a whole, so a revision cannot
    introduce a path that escapes the artifact directory.
    """
    if not isinstance(revision, dict) or "op" not in revision:
        raise ValueError("revision must be an object with an 'op'")
    op = revision["op"]
    if op not in ("replace_function", "replace_lines", "add_file", "remove_file"):
        raise ValueError("unknown revision op: %r" % (op,))
    revised = copy.deepcopy(artifact)
    tests_before = copy.deepcopy(artifact.get("tests"))
    purpose_before = copy.deepcopy(artifact.get("purpose"))

    if op == "replace_function":
        entry = _file_entry(revised, revision.get("file"))
        function = revision.get("function")
        if not isinstance(function, str) or not function:
            raise ValueError("replace_function requires a 'function'")
        entry["text"] = _replace_function_in_text(
            entry.get("text", ""), function, _require_text(revision)
        )
    elif op == "replace_lines":
        entry = _file_entry(revised, revision.get("file"))
        start_line = _require_int(revision, "from")
        end_line = _require_int(revision, "to")
        lines = entry["text"].splitlines()
        if start_line < 1 or end_line < start_line or end_line > len(lines):
            raise ValueError("line range %s..%s is out of range" % (start_line, end_line))
        lines[start_line - 1:end_line] = _require_text(revision).splitlines()
        entry["text"] = "\n".join(lines) + "\n"
    elif op == "add_file":
        name = revision.get("name")
        _validate_file_name(name)
        if any(entry.get("name") == name for entry in revised.get("files", [])):
            raise ValueError("file %r already exists" % name)
        revised.setdefault("files", []).append({"name": name, "text": _require_text(revision)})
    elif op == "remove_file":
        name = revision.get("name")
        entry = _file_entry(revised, name)
        revised["files"].remove(entry)

    if revised.get("tests") != tests_before or revised.get("purpose") != purpose_before:
        raise ValueError("no revision may edit the tests or the purpose")
    errors = validate(revised)
    if errors:
        raise ValueError(
            "resulting program failed validation: %s" % "; ".join(errors)
        )
    return revised


def apply_revisions(artifact, revisions):
    current = copy.deepcopy(artifact)
    for revision in revisions or []:
        current = apply_revision(current, revision)
    return current
