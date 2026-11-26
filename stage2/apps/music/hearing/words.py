"""stage2/apps/music/hearing/words.py - word lexicon.

Sensoryne's own word lexicon for the music application.
Spoken word forms are remembered by name as trajectories and
matched by trajectory distance, returning the nearest word
and the full ranked cohort of candidates with distances.

Matching is by dynamic time warping over the same
spectrotemporal trajectory the rest of the hearing uses, so
a spoken word is remembered as its place-by-time shape.  A
word form with no trajectory is stored without one and takes
part in no trajectory matching until one is given; it is
never matched by any other measure.

This module remembers and matches only.  The nearest word is
the nearest remembered form by trajectory distance.  It is
not a claim about what was said.
"""

import numpy as np

from .trajectory import PLACES, STEPS, dtw_distance

__all__ = [
    "WordLexicon",
    "Lexicon",
    "remember_word",
    "match_word",
    "rank_words",
]

_EPS = 1e-12


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


class WordLexicon:
    """Spoken word forms remembered by name as trajectories."""

    def __init__(self, entries=None):
        self.forms = []
        for entry in entries or []:
            self.remember(entry["name"], entry.get("trajectory"), entry.get("context"))

    def __len__(self):
        return len(self.forms)

    def remember(self, name, trajectory=None, context=None):
        """Remember a spoken word form by name.

        ``trajectory`` is its spectrotemporal trajectory,
        stored as a plain nested list so the entry stays
        JSON-serializable.  Re-remembering a name replaces
        its stored form.  Returns the stored entry.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("a word form needs a non-empty name")
        values = _as_trajectory(trajectory)
        entry = {
            "name": name,
            "trajectory": (
                [[float(v) for v in row] for row in values]
                if values is not None
                else None
            ),
            "context": context,
        }
        for index, existing in enumerate(self.forms):
            if existing["name"] == name:
                self.forms[index] = entry
                return entry
        self.forms.append(entry)
        return entry

    def forget(self, name):
        """Remove a remembered form by name.  Returns True when
        one was removed."""
        before = len(self.forms)
        self.forms = [entry for entry in self.forms if entry["name"] != name]
        return len(self.forms) < before

    def rank(self, trajectory):
        """The full ranked cohort of candidates with distances.

        Every remembered form holding a trajectory is scored
        by dynamic time warping distance from the given
        trajectory, nearest first.  Forms without a trajectory
        take part in no matching and do not appear.
        """
        values = _as_trajectory(trajectory)
        if values is None:
            return []
        ranked = []
        for entry in self.forms:
            held = _as_trajectory(entry.get("trajectory"))
            if held is None:
                continue
            ranked.append(
                {
                    "name": entry["name"],
                    "distance": float(dtw_distance(values, held)),
                    "context": entry.get("context"),
                }
            )
        ranked.sort(key=lambda item: item["distance"])
        return ranked

    def match(self, trajectory):
        """Match a spoken trajectory against the lexicon.

        Returns the nearest word, its distance, and the full
        ranked cohort.  An empty match has no nearest word.
        """
        cohort = self.rank(trajectory)
        if not cohort:
            return {
                "name": None,
                "nearest": None,
                "distance": None,
                "cohort": [],
                "candidates": [],
            }
        nearest = cohort[0]
        return {
            "name": nearest["name"],
            "nearest": nearest["name"],
            "distance": float(nearest["distance"]),
            "cohort": cohort,
            "candidates": cohort,
        }

    # Descriptive alias.
    nearest = match

    def to_dict(self):
        return {"forms": [dict(entry) for entry in self.forms]}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("lexicon data must be an object")
        return cls(entries=data.get("forms", []))


Lexicon = WordLexicon


def remember_word(lexicon, name, trajectory=None, context=None):
    """Remember a spoken word form in a lexicon."""
    return lexicon.remember(name, trajectory, context)


def match_word(lexicon, trajectory):
    """Match a spoken trajectory, returning the nearest word
    and the full ranked cohort."""
    return lexicon.match(trajectory)


def rank_words(lexicon, trajectory):
    """Return the full ranked cohort of candidates only."""
    return lexicon.rank(trajectory)
