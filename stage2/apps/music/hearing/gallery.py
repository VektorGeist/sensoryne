"""stage2/apps/music/hearing/gallery.py - gallery of known sounds.

Sensoryne's own remembered sounds for the music application.
Sounds are remembered by name as exemplars holding a
descriptor and a trajectory.  Recognition matches a new sound
against the gallery, by trajectory when trajectories are held
and otherwise by descriptor distance, returning the nearest
name, its distance, and whether it counts as known under the
threshold.

The gallery remembers and matches only.  It judges nothing
and names a sound only as the nearest remembered name.
"""

import math

import numpy as np

from .cochlea import PLACE_COUNT
from .trajectory import PLACES, STEPS, dtw_distance

__all__ = [
    "DEFAULT_THRESHOLD",
    "descriptor_distance",
    "Gallery",
    "KnownSounds",
    "remember_sound",
    "recognize_sound",
]

DEFAULT_THRESHOLD = 0.25

_EPS = 1e-12


def _as_descriptor(descriptor):
    values = np.asarray(descriptor, dtype=np.float64)
    if values.size != PLACE_COUNT:
        raise ValueError(
            "descriptor must have %d places, got %d" % (PLACE_COUNT, values.size)
        )
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    total = float(np.sum(values))
    if total > _EPS:
        values = values / total
    return values


def _as_trajectory(trajectory):
    if trajectory is None:
        return None
    values = np.asarray(trajectory, dtype=np.float64)
    if values.size == 0:
        return None
    if values.shape != (STEPS, PLACES):
        raise ValueError(
            "trajectory must have shape (%d, %d), got %r"
            % (STEPS, PLACES, values.shape)
        )
    return values


def descriptor_distance(first, second):
    """Euclidean distance between two place descriptors.

    Both descriptors are normalized to unit sum before the
    distance is taken, so two identical descriptors have
    distance 0.0 and descriptors concentrated on different
    places have distance approaching sqrt(2).
    """
    a = _as_descriptor(first)
    b = _as_descriptor(second)
    return float(np.linalg.norm(a - b))


class Gallery:
    """Exemplars of known sounds, remembered by name.

    Each exemplar holds a name, a descriptor, an optional
    trajectory, and an optional context.  Recognition returns
    the nearest exemplar's name, the distance to it, and
    whether the distance counts as known under the threshold.
    """

    def __init__(self, threshold=DEFAULT_THRESHOLD, exemplars=None):
        if threshold is None or float(threshold) < 0.0:
            raise ValueError("threshold must be a non-negative number")
        self.threshold = float(threshold)
        self.exemplars = []
        for exemplar in exemplars or []:
            self.remember(
                exemplar["name"],
                exemplar["descriptor"],
                exemplar.get("trajectory"),
                exemplar.get("context"),
            )

    def __len__(self):
        return len(self.exemplars)

    def remember(self, name, descriptor, trajectory=None, context=None):
        """Remember a sound by name as an exemplar.

        ``trajectory`` is stored as a plain nested list so the
        exemplar stays JSON-serializable.  Returns the stored
        exemplar.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("a known sound needs a non-empty name")
        descriptor_values = _as_descriptor(descriptor)
        trajectory_values = _as_trajectory(trajectory)
        exemplar = {
            "name": name,
            "descriptor": [float(v) for v in descriptor_values],
            "trajectory": (
                [[float(v) for v in row] for row in trajectory_values]
                if trajectory_values is not None
                else None
            ),
            "context": context,
        }
        self.exemplars.append(exemplar)
        return exemplar

    # Consolidation promotes episodes into the gallery under
    # this name.
    promote = remember

    def set_threshold(self, threshold):
        if threshold is None or float(threshold) < 0.0:
            raise ValueError("threshold must be a non-negative number")
        self.threshold = float(threshold)
        return self.threshold

    def distance_to(self, exemplar, descriptor, trajectory=None):
        """Distance from a new sound to one exemplar.

        Trajectory distance when both the new sound and the
        exemplar hold trajectories; descriptor distance
        otherwise.
        """
        new_trajectory = _as_trajectory(trajectory)
        held_trajectory = _as_trajectory(exemplar.get("trajectory"))
        if new_trajectory is not None and held_trajectory is not None:
            return float(dtw_distance(new_trajectory, held_trajectory)), "trajectory"
        return (
            descriptor_distance(descriptor, exemplar["descriptor"]),
            "descriptor",
        )

    def rank(self, descriptor, trajectory=None):
        """All exemplars ranked by distance, nearest first."""
        ranked = []
        for exemplar in self.exemplars:
            distance, method = self.distance_to(exemplar, descriptor, trajectory)
            ranked.append(
                {
                    "name": exemplar["name"],
                    "distance": float(distance),
                    "method": method,
                    "context": exemplar.get("context"),
                }
            )
        ranked.sort(key=lambda entry: entry["distance"])
        return ranked

    def recognize(self, descriptor, trajectory=None, threshold=None):
        """Match a new sound against the gallery.

        Returns the nearest name, its distance, the method
        used, and whether it counts as known under the
        threshold.  An empty gallery has no nearest name.
        """
        limit = self.threshold if threshold is None else float(threshold)
        if not self.exemplars:
            return {
                "name": None,
                "nearest": None,
                "distance": None,
                "known": False,
                "threshold": float(limit),
                "method": None,
                "candidates": [],
            }
        ranked = self.rank(descriptor, trajectory)
        nearest = ranked[0]
        return {
            "name": nearest["name"],
            "nearest": nearest["name"],
            "distance": float(nearest["distance"]),
            "known": bool(float(nearest["distance"]) <= limit),
            "threshold": float(limit),
            "method": nearest["method"],
            "candidates": ranked,
        }

    # Descriptive alias.
    match = recognize
    nearest = recognize

    def to_dict(self):
        return {
            "threshold": self.threshold,
            "exemplars": [
                {
                    "name": exemplar["name"],
                    "descriptor": list(exemplar["descriptor"]),
                    "trajectory": exemplar.get("trajectory"),
                    "context": exemplar.get("context"),
                }
                for exemplar in self.exemplars
            ],
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("gallery data must be an object")
        return cls(
            threshold=data.get("threshold", DEFAULT_THRESHOLD),
            exemplars=data.get("exemplars", []),
        )


KnownSounds = Gallery


def remember_sound(gallery, name, descriptor, trajectory=None, context=None):
    """Remember a sound in a gallery and return the exemplar."""
    return gallery.remember(name, descriptor, trajectory, context)


def recognize_sound(gallery, descriptor, trajectory=None, threshold=None):
    """Match a sound against a gallery and return the result."""
    return gallery.recognize(descriptor, trajectory, threshold)
