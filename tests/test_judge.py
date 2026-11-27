"""tests/test_judge.py - the judge seat reads finished sessions
from disk and appends verdicts to their history.

One story session is driven through the controller with a
temporary root, then listed, opened, and judged through
stage2.judge. No model calls are made.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2 import controller, judge
from stage2.apps.story import app as story_app
from stage2.store import ArtifactStore
from stage2.tools import ToolRegistry
from story_renderer import renderer


def _story():
    return {
        "objective": {
            "section_count": 2,
            "movements": ["opening", "ending"],
            "promised_events": ["a door opens", "a key turns", "rain falls"],
        },
        "sections": [
            {"title": "Opening", "text": "A door opens in the old house and Mara steps inside today."},
            {"title": "Closing", "text": "A key turns in the lock as rain falls on the roof all night long today."},
        ],
    }


def _open_finished(tmpdir):
    store = ArtifactStore(os.path.join(tmpdir, "artifacts"))
    registry = ToolRegistry()
    registry._session_counter = 0
    registry.pending_context = {
        "store": store,
        "app": story_app,
        "renderer": renderer,
        "session_root": os.path.join(tmpdir, "sessions"),
        "session_id": "story-perceptual-1",
        "artifact_id": "story-perceptual-1",
        "faults": [],
        "max_iterations": 4,
        "ledger_path": None,
    }
    controller.open_session(registry, {"application": "story", "objective": _story()["objective"], "condition": "perceptual", "test": False})
    context = registry.get("story-perceptual-1")
    controller.submit_work(context, _story())
    controller.submit_decision(context, {"faults": [], "revisions": [{"op": "replace_section", "section": 1, "text": "A door opens in the old house and Mara steps inside slowly today."}], "complete": False, "assessment": "revised once"})
    controller.submit_decision(context, {"faults": [], "revisions": [], "complete": True, "assessment": "finished"})
    assert context["ended_by"] == "model_complete"
    return registry


def _history(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class JudgeTests(unittest.TestCase):
    def test_list_open_and_record_verdict_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = _open_finished(tmpdir)
            listing = judge.list_sessions(registry)
            self.assertEqual(len(listing["sessions"]), 1)
            entry = listing["sessions"][0]
            self.assertEqual(entry["session_id"], "story-perceptual-1")
            self.assertEqual(entry["artifact_id"], "story-perceptual-1")
            self.assertEqual(entry["application"], "story")
            self.assertEqual(entry["condition"], "perceptual")
            self.assertEqual(entry["objective"], _story()["objective"])
            self.assertEqual(entry["ended_by"], "model_complete")
            self.assertGreaterEqual(entry["iterations"], 1)
            self.assertEqual(entry["judgments"], 0)
            pack = judge.open_judgment(registry, "story-perceptual-1")
            self.assertTrue(pack["written_work"])
            self.assertEqual(pack["written_work"][0]["source"]["sections"][0]["title"], "Opening")
            self.assertTrue(pack["iterations"])
            comparison = pack["iterations"][0]["comparison"]
            self.assertIn("findings", comparison)
            self.assertIsNotNone(pack["iterations"][0]["evidence_summary"])
            self.assertIsNotNone(pack["iterations"][0]["objective_coverage"])
            self.assertTrue(pack["decisions"])
            self.assertTrue(pack["judge_brief"])
            self.assertEqual(pack["ended_by"], "model_complete")
            self.assertEqual(pack["final_assessment"], "finished")
            verdict = {"bottom_line": "mixed", "judgment": "The story holds together, though its close is thin.", "strengths": ["promised events land"], "weaknesses": ["closing section is thin"]}
            self.assertEqual(judge.record_verdict(registry, "story-perceptual-1", verdict), {"recorded": True, "session_id": "story-perceptual-1"})
            path = os.path.join(tmpdir, "sessions", "story-perceptual-1", "history.jsonl")
            events = _history(path)
            judgments = [e for e in events if e.get("event") == "judgment"]
            self.assertEqual(len(judgments), 1)
            self.assertEqual(judgments[0]["verdict"], verdict)
            self.assertEqual(judgments[0]["session_id"], "story-perceptual-1")
            self.assertIn("ts", judgments[0])
            second = {"bottom_line": "good", "judgment": "On a second reading the work stands.", "strengths": [], "weaknesses": []}
            judge.record_verdict(registry, "story-perceptual-1", second)
            judgments = [e for e in _history(path) if e.get("event") == "judgment"]
            self.assertEqual(len(judgments), 2)
            self.assertEqual(judgments[1]["verdict"], second)
            self.assertEqual(judge.list_sessions(registry)["sessions"][0]["judgments"], 2)
            self.assertEqual(judge.open_judgment(registry, "story-perceptual-1")["judgments"], [verdict, second])

    def test_invalid_verdicts_and_unknown_sessions_raise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = _open_finished(tmpdir)
            with self.assertRaises(ValueError):
                judge.record_verdict(registry, "story-perceptual-1", {"bottom_line": "great", "judgment": "x", "strengths": [], "weaknesses": []})
            with self.assertRaises(ValueError):
                judge.record_verdict(registry, "story-perceptual-1", {"bottom_line": "good", "judgment": "", "strengths": [], "weaknesses": []})
            with self.assertRaises(ValueError):
                judge.record_verdict(registry, "story-perceptual-1", {"bottom_line": "good", "judgment": "x", "strengths": "yes", "weaknesses": []})
            with self.assertRaises(ValueError):
                judge.record_verdict(registry, "no-such-session", {"bottom_line": "good", "judgment": "x", "strengths": [], "weaknesses": []})
            with self.assertRaises(ValueError):
                judge.open_judgment(registry, "no-such-session")

    def test_every_application_module_exposes_a_facts_grounded_brief(self):
        for name in ("music", "poetry", "story", "drawing", "code", "typography", "animation", "sound_design"):
            module = __import__("stage2.apps.%s.app" % name, fromlist=["judge_brief"])
            brief = module.judge_brief()
            self.assertTrue(brief, "empty brief for %s" % name)
            self.assertIn("fact", brief.lower(), "brief for %s does not name the facts-grounding rule" % name)

    def test_viewer_page_source_contains_judgment_card_code(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage2", "viewer", "index.html")
        with open(path, encoding="utf-8") as handle:
            page = handle.read()
        self.assertIn("judgment-card", page)
        self.assertIn("renderJudgment", page)
        self.assertIn('e.event==="judgment"', page)
        self.assertIn("not good", page)


if __name__ == "__main__":
    unittest.main()
