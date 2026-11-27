"""stage2/store.py - artifact store.

One subdirectory per artifact containing ``artifact.json`` and a
``versions/`` folder. A revision never overwrites a version; it
adds one. The artifact's source representation and each rendered
output are always kept as distinct records, and every version
carries its own copy of the source it was rendered from. The
render record stored with a version carries the delta, invariant,
reliability, and coverage blocks computed for that version.
"""

import copy
import json
import os
from datetime import datetime, timezone


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


class ArtifactStore:
    """An artifact store rooted at a directory."""

    def __init__(self, root):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _artifact_dir(self, artifact_id):
        return os.path.join(self.root, artifact_id)

    def _record_path(self, artifact_id):
        return os.path.join(self._artifact_dir(artifact_id), "artifact.json")

    def create(self, artifact_id, source, objective, metadata=None):
        """Store a new artifact record and return it."""
        if os.path.exists(self._record_path(artifact_id)):
            raise FileExistsError("artifact already exists: %s" % artifact_id)
        record = {
            "artifact_id": artifact_id,
            "created": _utc_now(),
            "objective": objective,
            "source": copy.deepcopy(source),
            "metadata": copy.deepcopy(metadata) if metadata is not None else {},
            "versions": [],
        }
        os.makedirs(
            os.path.join(self._artifact_dir(artifact_id), "versions"),
            exist_ok=True,
        )
        _write_json(self._record_path(artifact_id), record)
        return copy.deepcopy(record)

    def add_version(self, artifact_id, render_path, render_info, source):
        """Append a new version and write its own source copy."""
        record = self.load(artifact_id)
        version_id = "v%d" % (len(record.get("versions", [])) + 1)
        version = {
            "version_id": version_id,
            "created": _utc_now(),
            "render_path": str(render_path),
            "render_info": copy.deepcopy(render_info)
            if render_info is not None
            else {},
            "source_file": "versions/%s/source.json" % version_id,
        }
        version_dir = os.path.join(
            self._artifact_dir(artifact_id), "versions", version_id
        )
        os.makedirs(version_dir, exist_ok=True)
        _write_json(
            os.path.join(version_dir, "source.json"), copy.deepcopy(source)
        )
        record.setdefault("versions", []).append(version)
        _write_json(self._record_path(artifact_id), record)
        return copy.deepcopy(version)

    def load(self, artifact_id):
        """Return the record for ``artifact_id``."""
        path = self._record_path(artifact_id)
        if not os.path.exists(path):
            raise FileNotFoundError("artifact not found: %s" % artifact_id)
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_version_source(self, artifact_id, version_id):
        """Return the source copy stored with one version."""
        path = os.path.join(
            self._artifact_dir(artifact_id),
            "versions",
            version_id,
            "source.json",
        )
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_artifacts(self):
        if not os.path.isdir(self.root):
            return []
        return sorted(
            name
            for name in os.listdir(self.root)
            if os.path.exists(self._record_path(name))
            and os.path.isdir(self._artifact_dir(name))
        )
