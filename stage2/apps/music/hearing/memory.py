"""stage2/apps/music/hearing/memory.py - episodic memory.

Sensoryne's own episodic memory for the music application.
Every hearing can be imprinted as an episode holding its
descriptor, its reading, its context, its note, and a
salience score combining loudness with novelty against the
gallery.  A new sound can recall the nearest past episodes
with their distances.  Consolidation promotes episodes into
the gallery when they are salient enough or repeated enough.

Salience is stated, not assumed.  Loudness in dB is
normalized as (loudness + 60) / 60, clipped to 0..1, so
-60 dB and quieter contribute 0 and 0 dB contributes 1.
Novelty is the distance to the nearest gallery exemplar
(descriptor distance, or trajectory distance when both sides
hold trajectories), clipped to 0..1; an empty gallery gives
novelty 1.0, a wholly new sound.  Salience is the mean of
the two: 0.5 * loudness_norm + 0.5 * novelty.

Consolidation is by rule, stated here: an episode is
promoted into the gallery when its salience is at least the
salience threshold, or when at least the repeat threshold of
episodes lie within the repeat radius of its descriptor.
Promotion remembers the episode under its note when it has
one, otherwise under its episode name.  Episodes already
promoted are not promoted again, and a name already in the
gallery is not remembered twice.

This module remembers and recalls only.  It judges nothing,
and a recall distance is a distance, not a claim of identity.
"""

import copy

import numpy as np

from .cochlea import PLACE_COUNT
from .gallery import descriptor_distance
from .trajectory import PLACES, STEPS, dtw_distance

__all__ = [
    "DEFAULT_SALIENCE_THRESHOLD",
    "DEFAULT_REPEAT_THRESHOLD",
    "REPEAT_RADIUS",
    "EpisodicMemory",
    "Memory",
    "loudness_norm",
    "compute_salience",
    "novelty_against_gallery",
]

DEFAULT_SALIENCE_THRESHOLD = 0.6
DEFAULT_REPEAT_THRESHOLD = 3
REPEAT_RADIUS = 0.25

_EPS = 1e-12


def loudness_norm(loudness):
    """Normalize a dB loudness to 0..1: (-60 dB -> 0, 0 dB -> 1)."""
    if loudness is None:
        return 0.0
    try:
        value = float(loudness)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, (value + 60.0) / 60.0)))


def _as_descriptor(descriptor):
    if isinstance(descriptor, dict) and "descriptor" in descriptor:
        descriptor = descriptor["descriptor"]
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


def novelty_against_gallery(gallery, descriptor, trajectory=None):
    """Novelty of a sound against a gallery, as (novelty, distance).

    Novelty is the nearest-exemplar distance clipped to 0..1.
    An empty or absent gallery gives novelty 1.0 and no
    distance.  When both the sound and the nearest exemplar
    hold trajectories, the distance is the trajectory
    distance; otherwise it is the descriptor distance.
    """
    if gallery is None or len(gallery) == 0:
        return 1.0, None
    place_values = _as_descriptor(descriptor)
    new_trajectory = _as_trajectory(trajectory)
    best = None
    for exemplar in gallery.exemplars:
        held = _as_trajectory(exemplar.get("trajectory"))
        if new_trajectory is not None and held is not None:
            distance = float(dtw_distance(new_trajectory, held))
        else:
            distance = descriptor_distance(place_values, exemplar["descriptor"])
        if best is None or distance < best:
            best = distance
    if best is None:
        return 1.0, None
    return float(min(1.0, max(0.0, best))), float(best)


def compute_salience(loudness, novelty):
    """Salience combining loudness with novelty: their mean."""
    if novelty is None:
        novelty = 1.0
    return float(0.5 * loudness_norm(loudness) + 0.5 * float(novelty))


class EpisodicMemory:
    """Episodes imprinted from hearings, recalled by distance,
    and consolidated into a gallery by rule."""

    def __init__(self, gallery=None, episodes=None):
        self.gallery = gallery
        self.episodes = []
        self._next_id = 1
        for episode in episodes or []:
            self.episodes.append(self._coerce_episode(episode))

    def __len__(self):
        return len(self.episodes)

    def _coerce_episode(self, episode):
        stored = copy.deepcopy(episode)
        if not isinstance(stored, dict) or "descriptor" not in stored:
            raise ValueError("episode must be an object holding a descriptor")
        stored["descriptor"] = [float(v) for v in _as_descriptor(stored["descriptor"])]
        trajectory = _as_trajectory(stored.get("trajectory"))
        stored["trajectory"] = (
            [[float(v) for v in row] for row in trajectory]
            if trajectory is not None
            else None
        )
        stored.setdefault("reading", None)
        stored.setdefault("context", None)
        stored.setdefault("note", None)
        stored.setdefault("loudness", None)
        stored.setdefault("novelty", None)
        stored.setdefault("salience", 0.0)
        stored.setdefault("promoted", False)
        if "episode_id" not in stored:
            stored["episode_id"] = self._next_id
        self._next_id = max(self._next_id, int(stored["episode_id"]) + 1)
        return stored

    def imprint(self, descriptor, reading=None, context=None, note=None,
                trajectory=None, loudness=None):
        """Imprint a hearing as an episode and return it.

        ``descriptor`` is the hearing's place descriptor (a
        reading dict carrying one is also accepted).
        ``reading`` is the hearing's plain language reading
        record, stored as given.  When ``note`` or ``loudness``
        are not given separately they are read from the
        reading.  Salience is computed at imprint time against
        the gallery as it stands then.
        """
        if isinstance(descriptor, dict) and reading is None and "descriptor" in descriptor:
            reading = descriptor
        place_values = _as_descriptor(descriptor)
        if reading is not None and not isinstance(reading, dict):
            reading = {"reading": reading}
        if note is None and isinstance(reading, dict):
            note = reading.get("note")
        if loudness is None and isinstance(reading, dict):
            if reading.get("loudness") is not None:
                loudness = reading.get("loudness")
            elif reading.get("loud") is not None:
                loudness = reading.get("loud")
        novelty, distance = novelty_against_gallery(
            self.gallery, place_values, trajectory
        )
        salience = compute_salience(loudness, novelty)
        trajectory_values = _as_trajectory(trajectory)
        episode = {
            "episode_id": self._next_id,
            "descriptor": [float(v) for v in place_values],
            "trajectory": (
                [[float(v) for v in row] for row in trajectory_values]
                if trajectory_values is not None
                else None
            ),
            "reading": copy.deepcopy(reading),
            "context": copy.deepcopy(context),
            "note": note,
            "loudness": (float(loudness) if loudness is not None else None),
            "novelty": float(novelty),
            "novelty_distance": distance,
            "salience": float(salience),
            "promoted": False,
        }
        self._next_id += 1
        self.episodes.append(episode)
        return copy.deepcopy(episode)

    def distance_to(self, episode, descriptor, trajectory=None):
        """Distance from a new sound to one episode, and the
        method used: trajectory when both hold trajectories,
        descriptor otherwise."""
        new_trajectory = _as_trajectory(trajectory)
        held = _as_trajectory(episode.get("trajectory"))
        if new_trajectory is not None and held is not None:
            return float(dtw_distance(new_trajectory, held)), "trajectory"
        return descriptor_distance(descriptor, episode["descriptor"]), "descriptor"

    def recall(self, descriptor, trajectory=None, k=3):
        """Recall the nearest past episodes, nearest first.

        Returns up to ``k`` dicts holding the episode, its
        distance, and the method used.  An empty memory
        recalls nothing.
        """
        if k is None or int(k) <= 0:
            return []
        place_values = _as_descriptor(descriptor)
        ranked = []
        for episode in self.episodes:
            distance, method = self.distance_to(episode, place_values, trajectory)
            ranked.append(
                {
                    "episode": copy.deepcopy(episode),
                    "episode_id": episode["episode_id"],
                    "distance": float(distance),
                    "method": method,
                    "note": episode.get("note"),
                    "salience": float(episode.get("salience", 0.0)),
                }
            )
        ranked.sort(key=lambda entry: entry["distance"])
        return ranked[: int(k)]

    # Descriptive alias.
    nearest = recall

    def repeat_count(self, episode, radius=REPEAT_RADIUS):
        """How many episodes, this one included, lie within the
        repeat radius of the given episode's descriptor."""
        target = _as_descriptor(episode["descriptor"])
        count = 0
        for other in self.episodes:
            if descriptor_distance(target, other["descriptor"]) <= float(radius):
                count += 1
        return count

    def consolidate(self, gallery=None, salience_threshold=DEFAULT_SALIENCE_THRESHOLD,
                    repeat_threshold=DEFAULT_REPEAT_THRESHOLD, radius=REPEAT_RADIUS):
        """Promote episodes into the gallery by rule.

        An episode is promoted when its salience is at least
        ``salience_threshold`` or when at least
        ``repeat_threshold`` episodes lie within ``radius`` of
        its descriptor.  Returns the list of promoted entries,
        each holding the gallery name and the episode id.
        """
        target_gallery = gallery if gallery is not None else self.gallery
        if target_gallery is None:
            raise ValueError("consolidation needs a gallery")
        if salience_threshold is None or float(salience_threshold) < 0.0:
            raise ValueError("salience_threshold must be a non-negative number")
        if repeat_threshold is None or int(repeat_threshold) < 1:
            raise ValueError("repeat_threshold must be at least 1")
        existing_names = {exemplar["name"] for exemplar in target_gallery.exemplars}
        promoted = []
        for episode in self.episodes:
            if episode.get("promoted"):
                continue
            salient = float(episode.get("salience", 0.0)) >= float(salience_threshold)
            repeated = self.repeat_count(episode, radius) >= int(repeat_threshold)
            if not (salient or repeated):
                continue
            name = episode.get("note") or "episode_%d" % int(episode["episode_id"])
            if name in existing_names:
                episode["promoted"] = True
                continue
            target_gallery.remember(
                name,
                episode["descriptor"],
                episode.get("trajectory"),
                context=copy.deepcopy(episode.get("context")),
            )
            existing_names.add(name)
            episode["promoted"] = True
            promoted.append(
                {
                    "name": name,
                    "episode_id": episode["episode_id"],
                    "salience": float(episode.get("salience", 0.0)),
                    "repeat_count": self.repeat_count(episode, radius),
                    "reason": "salient" if salient else "repeated",
                }
            )
        return promoted

    def to_dict(self):
        return {
            "next_id": self._next_id,
            "episodes": copy.deepcopy(self.episodes),
        }

    @classmethod
    def from_dict(cls, data, gallery=None):
        if not isinstance(data, dict):
            raise ValueError("memory data must be an object")
        memory = cls(gallery=gallery, episodes=data.get("episodes", []))
        if isinstance(data.get("next_id"), int):
            memory._next_id = max(memory._next_id, int(data["next_id"]))
        return memory


Memory = EpisodicMemory
