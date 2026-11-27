"""stage2/shape.py - shape the copy of evidence returned to the model.

The stored session record (history.jsonl, via the controller)
keeps the complete evidence document exactly as perceived. Only
the copy handed back through a tool result is shaped, at the
tool return boundary, so a tool result stays bounded in size.

What the shaped copy keeps in full: the summary, the measurements
(including the token pack's per-segment stated figures - peak,
RMS, zero crossings, summary values), the invariants with their
values, the reliability block, the objective coverage, the delta
where present, the structure record, and the input summary. The
whole tool result (evidence plus structure_record, input_summary,
artifact, session_so_far, presentation, and any other top-level
fields the tool returns) is shaped as one document under one
budget; the protected sections above are never bulk-replaced
except under the oversize fallback below.

What is replaced by a reference: a raw bulk block - a waveform
sample array, a spectrogram or field array, an image payload, or
a token pack segment's raw data. Each replacement is a dict
``{"_reference": "<what it is>; <size> chars; session record"}``,
stating what the block is, its size, and that its full data is
held in the session record. The shaped copy also carries, at its
top level, ``shaped_copy`` (True), ``shaping_note`` and
``complete_evidence_location`` stating that it is a shaped copy
and that the complete evidence is in the session record, plus a
compact ``shaping_replacements`` summary (counts by replacement
kind, with an example path per kind) so the model can tell what
was replaced and where the whole of it lives.

Deciding what to replace, in order:

1. Duplicate aliases. ``oscilloscope`` and
   ``oscilloscope_block`` hold the same block twice, as do the
   envelope keys inside it and the ``summary`` / ``heard`` pair
   on each segment. The canonical copy stays in full; the
   identical duplicate is replaced by a reference to it. No
   distinct figure is lost.

2. Bulk by name. Any node whose key names raw payload data -
   waveform / sample / spectrogram / field / image / pixel /
   trajectory arrays, envelope arrays, and the raw field arrays
   that carry the same data under material names (``descriptor``,
   ``median_place``, ``mode_weights`` and its alias,
   ``aggregate``, harmonic ``scores``) - is replaced wherever it
   occurs, including inside ``full``, inside the token pack, and
   inside per-segment readings. Names are matched exactly, so
   measurement scalars that merely share a word with a bulk
   name are untouched, and the oscilloscope's stated peak, RMS
   and zero-crossing figures stay.

3. Repeat copies. The evidence carries the same measurements
   several times over: ``measurements``, the top-level
   ``token_pack``, ``full.measurements`` and ``full.token_pack``.
   The first (canonical) copy stays in full; each later repeat
   of that whole subtree is replaced by a reference to the
   canonical path, stating that the full data is held in the
   session record. The canonical copy carries every stated
   figure, so no measurement is lost.

4. Budget enforcement. After 1-3, if the shaped copy still
   exceeds the budget (default 50,000 characters as JSON), the
   largest remaining subtrees are reference-replaced, largest
   first, re-measuring after each replacement, until the copy
   fits or nothing replaceable remains. The replacement never
   takes a protected container whole: the evidence document,
   its summary, measurements, invariants, reliability block,
   objective coverage and delta, the token pack, its segment
   list and each segment dict, each oscilloscope dict, the
   invariants' checked list and entries, the structure record
   and the input summary keep their shape, counts and scalar
   figures. Only a subtree inside such a container is replaced,
   and only when it saves more than its reference costs: a
   per-segment reading, a mode or polyphony list, a per-note
   violations list, or a per-note structure list. Each such
   replacement states what the block is, its size, and that
   its full data is held in the session record. Protected
   scalar sections (summary, reliability, objective coverage,
   delta) are never replaced as a whole; the invariants keep
   each checked entry's id, description, held flag and value.

The budget applies per section. The evidence copy itself is
enforced to the budget on its own sub-document, so the
structure record and input summary - each kept in full - cannot
push the evidence copy over budget, and the evidence copy
cannot force them to be cut. The structure record and the input
summary are each enforced to the budget only when that section
alone is over budget, replacing largest subtrees inside it.
Any other top-level field (presentation, session_so_far,
artifact, hypothetical artifact) is unprotected and is enforced
together under the budget on its own sub-document, so a large
presentation string, session-so-far block or artifact cannot
push the whole result over the proven size either. Shaping
never mutates its input; it returns a deep copy.

The model token pack is exempt from shaping in whole and in
part: shaping never replaces the pack, a segment, or any field
inside a segment, at any budget. Bulk-by-name, duplicate-alias,
and budget enforcement all carry the pack over byte-identical;
every reduction needed to fit the budget comes from the other
sections.
"""

import copy
import json

# Aim: the shaped copy (the whole tool result document) stays
# comfortably under the size proven to pass (about 99,000
# characters), at roughly 50,000 characters.
EVIDENCE_BUDGET = 50000

# Keys that name raw payload data wherever they occur. Matched
# exactly; a measurement scalar with a similar name is not
# matched unless it is listed here. The oscilloscope's stated
# figures (peak, RMS, zero crossings) are not listed.
_BULK_KEYS = frozenset(
    {
        # waveform sample arrays and spectrogram / field arrays
        "waveform_image",
        "waveform",
        "samples",
        "sample_array",
        "spectrogram_image",
        "spectrogram",
        "field",
        "field_array",
        "place_time_field",
        "place",
        "place_vector",
        "trajectory",
        "trajectories",
        # envelope arrays derived from the waveform
        "peak_envelope_20bins",
        "envelope",
        # image payloads
        "chladni_image",
        "plate_image",
        "image",
        "image_array",
        "image_payload",
        "images",
        "labeled_images",
        "pixels",
        "pixel_array",
        "frame_image",
        # raw field arrays carried under material names
        "descriptor",
        "median_place",
        "mode_weights",
        "chladni_weights",
        "aggregate",
        "scores",
        # formant / frame tracks are per-frame raw series
        "formant_track",
        "formant_transitions",
        "frames",
    }
)

# Top-level evidence subtrees whose later repeats are replaced
# by a reference to the canonical (first) copy.
_REPEAT_PATHS = (
    ("evidence", "token_pack"),
    ("evidence", "full", "measurements"),
    ("evidence", "full", "token_pack"),
    ("evidence", "full", "reading"),
)

# Sections whose scalar content is protected during budget
# enforcement. Only an oversize array inside them may be
# reference-replaced, and only when the sections alone would
# exceed the budget.
_PROTECTED_TOP = frozenset({"summary", "reliability", "objective_coverage", "delta"})

_PROTECTED_EVIDENCE_KEYS = frozenset(
    {"summary", "reliability", "objective_coverage", "delta"}
)


def _size(value):
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return len(repr(value))


def _shape_of(value):
    if isinstance(value, list):
        if value and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        ):
            return "number array"
        if value and all(isinstance(item, list) for item in value):
            cols = len(value[0]) if value else 0
            return "2-D array (%d x %d)" % (len(value), cols)
        return "array (%d entries)" % len(value)
    if isinstance(value, dict):
        return "object (%d keys)" % len(value)
    return type(value).__name__


def _reference(what, value):
    """A compact reference: what it is, its size, where it lives."""
    return {
        "_reference": "%s; %d chars; full data in session record"
        % (what, _size(value))
    }


def _is_bulk_key(key):
    return isinstance(key, str) and key in _BULK_KEYS


def _is_model_pack_path(path):
    """True for paths inside the model-facing token pack."""
    parts = path.split(".") if path else []
    if not parts:
        return False
    # evidence.measurements.token_pack and evidence.token_pack
    # are the model-facing copies. evidence.full.token_pack and
    # evidence.full.measurements.token_pack are the stored
    # copies and are not protected here.
    if parts[0] == "evidence" and len(parts) >= 3 and parts[1] == "measurements" and parts[2] == "token_pack":
        return True
    if parts[0] == "evidence" and len(parts) >= 2 and parts[1] == "token_pack":
        return True
    return False


def _replace_bulk(node, path, replacements):
    """Replace bulk-keyed subtrees in place (on a copy)."""
    if isinstance(node, dict):
        for key in list(node.keys()):
            child_path = "%s.%s" % (path, key) if path else str(key)
            # The structure record and input summary are kept
            # in full: bulk replacement does not apply inside
            # them; only their own budget enforcement (when a
            # section alone is over budget) may replace their
            # largest subtrees.
            if child_path in ("structure_record", "input_summary"):
                continue
            # The model-facing token pack is carried whole:
            # its segment envelopes, mode weights, and
            # descriptors are stated figures, not raw payload,
            # and are never bulk-replaced. Only the stored
            # token pack under evidence.full may be replaced.
            if _is_model_pack_path(child_path):
                continue
            if _is_bulk_key(key):
                value = node[key]
                if value is None:
                    continue
                node[key] = _reference(key, value)
                replacements.append(
                    {"path": child_path, "what": "raw %s data" % key}
                )
            else:
                _replace_bulk(node[key], child_path, replacements)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _replace_bulk(item, "%s.%d" % (path, index), replacements)


def _replace_duplicates(node, path, replacements):
    """Replace identical duplicate alias subtrees by a reference.

    ``oscilloscope_block`` duplicates ``oscilloscope``;
    ``envelope`` duplicates ``peak_envelope_20bins`` inside an
    oscilloscope block; ``heard`` duplicates ``summary`` on a
    segment. The first occurrence stays in full. Comparison is
    by exact equality of the two sibling values, so distinct
    figures that merely share a name are kept.
    """
    if path in ("structure_record", "input_summary"):
        return
    if _is_model_pack_path(path):
        return
    if isinstance(node, dict):
        # Top-level heard duplicates the evidence summary.
        ev = node.get("evidence")
        if isinstance(ev, dict) and isinstance(ev.get("measurements"), dict):
            meas = ev["measurements"]
            if (
                "heard" in meas
                and "summary" in ev
                and meas["heard"] == ev["summary"]
                and meas["heard"] is not None
            ):
                meas["heard"] = _reference(
                    "duplicate heard (summary)", meas["heard"]
                )
                replacements.append(
                    {
                        "path": "evidence.measurements.heard",
                        "what": "duplicate heard data",
                    }
                )
        # Sibling duplicates inside this dict.
        for alias, canonical in (
            ("oscilloscope_block", "oscilloscope"),
            ("envelope", "peak_envelope_20bins"),
            ("heard", "summary"),
        ):
            if alias in node and canonical in node:
                if node[alias] == node[canonical] and node[alias] is not None:
                    child_path = "%s.%s" % (path, alias) if path else alias
                    canonical_path = (
                        "%s.%s" % (path, canonical) if path else canonical
                    )
                    node[alias] = _reference(
                        "duplicate %s (%s)" % (alias, canonical),
                        node[alias],
                    )
                    replacements.append(
                        {
                            "path": child_path,
                            "what": "duplicate %s data" % alias,
                        }
                    )
        # A segment reading repeats its own heard sentence:
        # reading.sentence and reading.summary both hold the
        # segment's heard text, which the segment summary also
        # holds. The segment summary stays in full; each
        # identical reading copy is a reference to it.
        reading = node.get("reading")
        if isinstance(reading, dict) and isinstance(node.get("summary"), str):
            for rkey in ("sentence", "summary"):
                if reading.get(rkey) == node.get("summary") and reading.get(rkey) is not None:
                    rpath = "%s.reading.%s" % (path, rkey) if path else "reading.%s" % rkey
                    reading[rkey] = _reference(
                        "duplicate reading.%s (summary)" % rkey,
                        reading[rkey],
                    )
                    replacements.append(
                        {
                            "path": rpath,
                            "what": "duplicate reading.%s data" % rkey,
                        }
                    )
        for key in list(node.keys()):
            child_path = "%s.%s" % (path, key) if path else str(key)
            _replace_duplicates(node[key], child_path, replacements)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _replace_duplicates(item, "%s.%d" % (path, index), replacements)


def _get(node, *path):
    current = node
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None, False
        current = current[key]
    return current, True


def _set(node, path, value):
    current = node
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def _replace_repeats(shaped, replacements):
    """Replace later repeats of canonical subtrees by a reference."""
    for path in _REPEAT_PATHS:
        value, found = _get(shaped, *path)
        if not found or value is None:
            continue
        if isinstance(value, dict) and "_reference" in value:
            continue
        if path[-1] == "token_pack":
            canonical = "evidence.measurements.token_pack"
        else:
            canonical = "evidence.measurements"
        dotted = ".".join(path)
        _set(
            shaped,
            path,
            {
                "_reference": "repeat of %s; canonical %s; full data in session record"
                % (dotted, canonical)
            },
        )
        replacements.append(
            {"path": dotted, "what": "repeat copy; canonical copy at %s" % canonical}
        )
    return shaped


def _is_protected_section(path):
    parts = path.split(".") if path else []
    if not parts:
        return True
    if parts[0] in _PROTECTED_TOP:
        return True
    if parts[0] == "evidence" and len(parts) > 1 and parts[1] in _PROTECTED_EVIDENCE_KEYS:
        return True
    return False


def _is_protected_container(path):
    """Containers never replaced whole during budget enforcement."""
    parts = path.split(".") if path else []
    if _is_model_pack_path(path):
        return True
    if path in (
        "",
        "evidence",
        "evidence.measurements",
        "evidence.invariants",
        "evidence.reliability",
        "evidence.objective_coverage",
        "evidence.delta",
        "evidence.robustness",
        "evidence.full",
        "evidence.measurements.token_pack",
        "evidence.token_pack",
        "structure_record",
        "input_summary",
    ):
        return True
    if parts and parts[-1] == "segments_data":
        return True
    # Each segment dict keeps its stated figures; its children
    # are the candidates.
    if len(parts) >= 2 and parts[-2] == "segments_data":
        return True
    # The invariants' checked list and each entry keep their
    # id, description, held flag and value.
    if parts and parts[-1] == "checked":
        return True
    if len(parts) >= 2 and parts[-2] == "checked":
        return True
    # Each oscilloscope dict keeps its stated peak/RMS/ZCR.
    if parts and parts[-1] in ("oscilloscope", "oscilloscope_block"):
        return True
    # Each reading dict keeps its stated figures; its children
    # (already bulk-replaced where raw) are the candidates.
    if parts and parts[-1] == "reading":
        return True
    # Structure record and input summary containers keep their
    # shape; only their largest child lists are candidates.
    if parts and parts[0] in ("structure_record", "input_summary"):
        return True
    # Protected scalar sections are containers too: only an
    # oversize array inside them may be replaced.
    if _is_protected_section(path):
        return True
    return False


def _collect_candidates(node, path, out):
    """Collect (size, path) for replaceable subtrees."""
    if _is_model_pack_path(path):
        # The model token pack is exempt in whole and in part:
        # no node inside it is ever a budget candidate.
        return
    if isinstance(node, str):
        # A lone oversize string (for example a long
        # presentation) is replaceable; ordinary short strings
        # are not budget candidates.
        if path and not _is_protected_section(path) and _size(node) > 50000:
            out.append((_size(node), path))
        return
    if isinstance(node, dict):
        if "_reference" in node:
            return
        if not _is_protected_container(path):
            out.append((_size(node), path))
        for key, value in node.items():
            child = "%s.%s" % (path, key) if path else str(key)
            _collect_candidates(value, child, out)
    elif isinstance(node, list):
        if not _is_protected_container(path):
            out.append((_size(node), path))
        for index, item in enumerate(node):
            _collect_candidates(item, "%s.%d" % (path, index), out)


def _resolve_parent(root, path):
    parts = path.split(".")
    parent = root
    for key in parts[:-1]:
        if isinstance(parent, dict):
            if key not in parent:
                return None, None
            parent = parent[key]
        elif isinstance(parent, list):
            try:
                parent = parent[int(key)]
            except Exception:
                return None, None
        else:
            return None, None
    return parent, parts[-1]


def _node_at(root, path):
    parent, last = _resolve_parent(root, path)
    if parent is None:
        return None
    if isinstance(parent, dict):
        return parent.get(last)
    if isinstance(parent, list):
        try:
            return parent[int(last)]
        except Exception:
            return None
    return None


def _replace_at(root, path):
    parent, last = _resolve_parent(root, path)
    if parent is None:
        return None
    if isinstance(parent, dict):
        if last not in parent:
            return None
        original = parent[last]
        parent[last] = _reference(
            _shape_of(original), original
        )
        return original
    if isinstance(parent, list):
        try:
            idx = int(last)
            original = parent[idx]
            parent[idx] = _reference(
                _shape_of(original), original
            )
            return original
        except Exception:
            return None
    return None


def _enforce_budget(shaped, budget, replacements):
    """Reference-replace largest subtrees until the document fits."""
    guard = {
        "shaped_copy",
        "shaping_note",
        "complete_evidence_location",
        "shaping_replacements",
    }
    for _ in range(600):
        if _size(shaped) <= budget:
            break
        candidates = []
        _collect_candidates(shaped, "", candidates)
        candidates = [
            (size, path)
            for size, path in candidates
            if path and path.split(".")[0] not in guard
        ]
        if not candidates:
            break
        candidates.sort(key=lambda item: (_is_protected_section(item[1]), -item[0]))
        chosen = None
        for size, path in candidates:
            if _is_protected_section(path):
                continue
            # Only replace when the saving beats the reference:
            # a subtree no larger than its reference would make
            # the copy bigger, not smaller, and must stay in full.
            node = _node_at(shaped, path)
            if node is None:
                continue
            ref_size = _size(_reference(_shape_of(node), node))
            if size - ref_size <= 0:
                continue
            chosen = (size, path)
            break
        if chosen is None:
            # Only protected sections remain: take the largest
            # list inside one, never the section itself.
            for size, path in candidates:
                if not _is_protected_section(path):
                    continue
                node2 = _node_at(shaped, path)
                if node2 is None:
                    continue
                ref_size = _size(_reference(_shape_of(node2), node2))
                if size - ref_size <= 0:
                    continue
                node = _node_at(shaped, path)
                if isinstance(node, list) and len(node) > 1:
                    chosen = (size, path)
                    break
            if chosen is None:
                # The model token pack is never a candidate in
                # any form: no segment entry inside it may be
                # replaced to meet the budget. The budget then
                # cannot be met from the evidence copy and the
                # loop stops with the pack whole.
                break
        size, path = chosen
        before = _size(shaped)
        original = _replace_at(shaped, path)
        if original is None:
            break
        if _size(shaped) >= before:
            # The reference did not save space; do not loop on
            # replacements that make the copy larger.
            break
        replacements.append(
            {
                "path": path,
                "what": "oversize %s (budget %d)"
                % (_shape_of(original), budget),
            }
        )
    return shaped


def _summarize_replacements(replacements):
    """Compact per-kind counts with an example path per kind.

    A per-occurrence list for a 20-segment token pack would
    itself exceed the budget; counts plus example paths tell
    the model what was replaced and where the whole lives
    without reintroducing the bulk.
    """
    counts = {}
    examples = {}
    for entry in replacements:
        what = entry.get("what", "replacement")
        counts[what] = counts.get(what, 0) + 1
        examples.setdefault(what, entry.get("path", ""))
    summary = []
    for what in sorted(counts):
        summary.append(
            {"what": what, "count": counts[what], "example": examples[what]}
        )
    return summary


def shape_document(document, budget=EVIDENCE_BUDGET, location="session record"):
    """Return a shaped deep copy of ``document`` under ``budget``.

    ``document`` is the whole tool result dict. The input is
    not mutated. The returned copy carries ``shaped_copy``,
    ``shaping_note``, ``complete_evidence_location`` and a
    compact ``shaping_replacements`` summary at its top level.
    The model token pack is carried whole at any budget.
    """
    if not isinstance(document, dict):
        return document
    shaped = copy.deepcopy(document)
    replacements = []
    # 1. Repeat copies first, so bulk and duplicate handling
    # runs once on the canonical copy only; then duplicate
    # aliases, then bulk by name.
    _replace_repeats(shaped, replacements)
    _replace_duplicates(shaped, "", replacements)
    _replace_bulk(shaped, "", replacements)
    # Shaping metadata first, so the budget below covers the
    # whole returned copy including this metadata. It states,
    # inside the copy itself, that the copy is shaped and where
    # the complete evidence lives.
    shaped["shaped_copy"] = True
    shaped["shaping_note"] = (
        "This is a shaped copy of the evidence. Raw bulk blocks (waveform "
        "sample arrays, spectrogram and field arrays, image payloads, and "
        "token pack segment raw data) are replaced by references stating "
        "what the block is and its size. The summary, measurements, "
        "invariants with their values, reliability block, objective "
        "coverage, delta, structure record, and input summary are kept in "
        "full. The complete evidence is held in the session record."
    )
    shaped["complete_evidence_location"] = (
        "The complete evidence document is stored in the session record "
        "(history.jsonl); this tool result is a shaped copy only."
        if location == "session record"
        else str(location)
    )
    shaped["shaping_replacements"] = []
    # 2. Budget enforcement, per section. The evidence copy
    # itself is the bounded section: it is enforced to ``budget``
    # on its own sub-document, so the structure record and the
    # input summary - each kept in full - cannot push the
    # evidence copy over budget, and the evidence copy cannot
    # force the structure record or input summary to be cut.
    # The structure record and the input summary are each
    # enforced to ``budget`` only when that section alone is
    # over budget, replacing largest subtrees inside it. Any
    # other top-level field (presentation, session_so_far,
    # artifact, hypothetical artifact) is unprotected: those are
    # enforced together under ``budget`` on their own
    # sub-document, so a large presentation string or artifact
    # cannot push the whole result over budget either.
    if "evidence" in shaped:
        # Enforce on the evidence sub-document alone: the
        # shaping metadata at the top level is not part of the
        # evidence copy's budget.
        ev_wrapped = {"evidence": shaped["evidence"]}
        # Leave room for the shaping metadata that sits on the
        # shaped copy itself (note, location, replacements
        # summary), so the whole shaped evidence copy - metadata
        # included - stays under ``budget``.
        _enforce_budget(ev_wrapped, budget - 4500, replacements)
        shaped["evidence"] = ev_wrapped["evidence"]
    shaped["shaping_replacements"] = _summarize_replacements(replacements)
    for section in ("structure_record", "input_summary"):
        if section not in shaped:
            continue
        section_doc = {section: shaped[section]}
        if _size(section_doc[section]) > budget:
            _enforce_budget(section_doc, budget, replacements)
            shaped[section] = section_doc[section]
            shaped["shaping_replacements"] = _summarize_replacements(
                replacements
            )
    other = {
        key: value
        for key, value in shaped.items()
        if key
        not in (
            "evidence",
            "structure_record",
            "input_summary",
            "shaped_copy",
            "shaping_note",
            "complete_evidence_location",
            "shaping_replacements",
        )
    }
    if other and _size(other) > budget:
        other_doc = dict(other)
        _enforce_budget(other_doc, budget, replacements)
        for key in list(other_doc):
            shaped[key] = other_doc[key]
        shaped["shaping_replacements"] = _summarize_replacements(replacements)
    return shaped


def shape_evidence(evidence, budget=EVIDENCE_BUDGET):
    """Shape a bare evidence document (probe returns it directly).

    The evidence is wrapped as ``{"evidence": ...}`` so the
    repeat-copy and budget rules apply identically, then
    unwrapped. The shaping metadata is kept on the evidence
    itself, so the returned copy states inside itself that it
    is a shaped copy.
    """
    if not isinstance(evidence, dict):
        return evidence
    wrapped = shape_document({"evidence": evidence}, budget=budget)
    shaped = wrapped.pop("evidence")
    replacements = wrapped.pop("shaping_replacements", [])
    if isinstance(shaped, dict):
        shaped["shaped_copy"] = True
        shaped["shaping_note"] = wrapped.get("shaping_note")
        shaped["complete_evidence_location"] = wrapped.get(
            "complete_evidence_location"
        )
        shaped["shaping_replacements"] = replacements
    return shaped


__all__ = [
    "EVIDENCE_BUDGET",
    "shape_document",
    "shape_evidence",
]
