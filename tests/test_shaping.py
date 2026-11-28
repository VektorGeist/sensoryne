"""tests/test_shaping.py - evidence return-shaping tests.

The stored session record keeps the complete evidence exactly
as perceived; only the copy handed back through a tool return
is shaped. A large stored evidence therefore returns with the
model token pack whole while the stored record stays complete,
and a small evidence passes through unshaped in substance (its
sections intact, only the shaping metadata added).

The model token pack is exempt from shaping in whole and in
part: shaping never replaces the pack, a segment, or any field
inside a segment, at any budget. Other sections take any
reduction, so the shaped copy's non-pack sections stay bounded
even though the pack itself is carried in full.

No model calls are made. The large case is a stored
evidence document built inside the test itself, in a
temporary directory, with the structure and scale of a
symphony-length music session record: the same evidence
sections, the same 20-segment model token pack, the same
bulk waveform, spectrogram and image arrays in the stored
full copy, the same repeat measurements, and the same
over-budget delta and comparison blocks, so the shaping
assertions exercise the same behaviour on every run.
"""

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2 import controller
from stage2.shape import EVIDENCE_BUDGET, shape_document, shape_evidence, _size
from stage2.store import ArtifactStore
from stage2.tools import ToolRegistry, dispatch


def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _large_model_pack():
    segments=[]
    for i in range(20):
        start=float(i*3.6)
        sentence=(f"Heard a tonal sound lasting 3.62 seconds: pitched at F3, 174.6 Hz, "
                  f"+0.3 cents from exact; loudness -15.1 dB; brightness 552 Hz; "
                  f"energy centered at 458 Hz in the low mid region; onset rate 1.93 per second; "
                  f"pitch trend rising; loudness trend steady. Segment {i+1} of 20 in a "
                  f"symphony-length render. Strongest plate modes: mode 2 (weight 0.186), "
                  f"mode 12 (weight 0.180), mode 13 (weight 0.139).")
        seg={
            "segment": i+1,
            "start_time": start,
            "end_time": start+3.62,
            "reading": {
                "sentence": sentence,
                "note": "F3",
                "cents": 0.25,
                "pitch_hz": 174.64,
                "loudness": -15.11,
                "brightness": 551.87,
                "energy_center_hz": 457.97,
                "region": "low mid",
                "pitch_trend": "rising",
                "loud_trend": "steady",
                "onset_rate": 1.93,
                "strongest_modes": [{"mode":2,"weight":0.186},{"mode":12,"weight":0.180},{"mode":13,"weight":0.139}],
                "silent": False, "tonal": True, "pitched": True,
            },
            "peak_band": "mid",
            "chladni_modes": [{"mode":m,"pair":[1,m%5+1],"frequency_hz":100.0+m*13.5,"weight":0.01+0.001*m} for m in range(2,32)],
            "mode_weights": [round(0.001*(k%17)+0.0005*k,6) for k in range(78)],
            "harmonic_pitch_hz": 174.64,
            "harmonic_midi": 53,
            "harmonic_confidence": 0.92,
            "polyphony_candidates": [{"midi":53+m,"weight":0.1+0.02*m,"pitch_hz":150.0+m*10.0} for m in range(8)],
            "gallery_match": {"name":"piano","score":0.87},
            "oscilloscope": {"peak":0.42,"rms":0.18,"zero_crossing_rate":0.05,
                             "peak_envelope_20bins":[round(0.1+0.02*((k*3)%10),4) for k in range(20)],
                             "trend":"steady"},
        }
        segments.append(seg)
    return {
        "segment_count": 20,
        "segments_data": segments,
        "duration": 73.1,
        "sample_rate": 44100,
        "phonetics": {"syllables":12,"formants":[500.0,1500.0,2500.0],"voicing":0.9,"method":"stated phonetic method for the segment pack"},
        "descriptor": [round(0.01*k,4) for k in range(32)],
        "mode_weights": [round(0.001*(k%17)+0.0005*k,6) for k in range(78)],
    }

def _large_full_pack():
    segments=[]
    for i in range(20):
        start=float(i*3.6)
        sentence=(f"Heard a tonal sound lasting 3.62 seconds: pitched at F3, 174.6 Hz, "
                  f"+0.3 cents from exact; loudness -15.1 dB; brightness 552 Hz; "
                  f"energy centered at 458 Hz in the low mid region; onset rate 1.93 per second; "
                  f"pitch trend rising; loudness trend steady. Segment {i+1}.")
        osc={"peak":0.42,"rms":0.18,"zero_crossing_rate":0.05,
             "peak_envelope_20bins":[round(0.1+0.02*((k*3)%10),4) for k in range(20)],"trend":"steady"}
        seg={
            "segment": i+1,
            "start_time": start, "end_time": start+3.62,
            "times": [start, start+3.62],
            "reading": {"sentence": sentence, "summary": sentence, "silent": False, "tonal": True, "pitched": True,
                        "note":"F3","cents":0.25,"pitch_hz":174.64,"f0":174.64,"loudness":-15.11,"brightness":551.87,
                        "centroid":457.97,"peak_band":"mid","region":"low mid","pitch_trend":"rising","loud_trend":"steady",
                        "onset_flux":0.12,"onset_rate":1.93},
            "summary": sentence, "heard": sentence,
            "pitch_hz":174.64,"f0":174.64,"note":"F3","cents":0.25,"loud":-15.11,"loudness":-15.11,
            "brightness":551.87,"centroid":457.97,"peak_band":"mid","region":"low mid",
            "pitch_trend":"rising","loud_trend":"steady","onset_flux":0.12,"onset_rate":1.93,
            "chladni_modes": [{"mode":m,"pair":[1,m%5+1],"frequency_hz":100.0+m*13.5,"weight":0.01} for m in range(2,32)],
            "mode_weights": [round(0.001*(k%17),6) for k in range(78)],
            "chladni_weights": [round(0.001*(k%17),6) for k in range(78)],
            "gallery_match": {"name":"piano","score":0.87},
            "harmonic_pitch":174.64,"harmonic_midi":53,"harmonic_confidence":0.92,
            "polyphony_candidates": [{"midi":53+m,"weight":0.1} for m in range(8)],
            "oscilloscope": copy.deepcopy(osc),
            "oscilloscope_block": copy.deepcopy(osc),
            # bulk arrays: waveform samples and a 2-D spectrogram
            "samples": [round(0.5*((k*7+i)%13)/13.0-0.25,5) for k in range(3000)],
            "spectrogram_image": [[round(0.01*((r*31+c*7+i)%17),4) for c in range(40)] for r in range(30)],
            "chladni_image": [[round(0.01*((r*13+c*5+i)%11),4) for c in range(20)] for r in range(20)],
            "descriptor": [round(0.01*k,4) for k in range(32)],
        }
        segments.append(seg)
    return {"segment_count":20,"segments":20,"segments_data":segments,"duration":73.1,"sample_rate":44100,
            "phonetics":{"syllables":12,"formants":[500.0,1500.0,2500.0],"voicing":0.9},
            "descriptor":[round(0.01*k,4) for k in range(32)],
            "mode_weights":[round(0.001*(k%17),6) for k in range(78)]}

def _build_large_evidence():
    summary=("Heard a tonal sound lasting 73.10 seconds: pitched at A4, 440.0 Hz, +0.2 cents from exact; "
             "loudness -15.8 dB; brightness 599 Hz; energy centered at 496 Hz in the low mid region; "
             "onset rate 1.78 per second; pitch trend falling; loudness trend steady. Strongest plate modes: "
             "mode 12 (weight 0.190), mode 2 (weight 0.185), mode 13 (weight 0.135).")
    pack=_large_model_pack()
    measurements={
        "heard": summary,
        "note":"A4","cents":0.2,"pitch_hz":440.0486,"loudness":-15.8122,"brightness":599.0,
        "energy_center_hz":496.0,"region":"low mid","peak_band":"mid","pitch_trend":"falling","loud_trend":"steady",
        "onset_rate":1.78,"duration":73.0964,"silent":False,"tonal":True,"pitched":True,
        "descriptor":[round(0.01*k,4) for k in range(32)],
        "mode_weights":[round(0.001*(k%17),6) for k in range(78)],
        "chladni_modes":[{"mode":m,"pair":[1,m%5+1],"frequency_hz":100.0+m*13.5,"weight":0.01} for m in range(2,32)],
        "harmonic_pitch_hz":440.05,"harmonic_midi":69,"harmonic_confidence":0.93,"harmonic_disagreement":False,
        "polyphony_candidates":[{"midi":69+m,"weight":0.1} for m in range(6)],
        "gallery_match":{"name":"piano","score":0.88},
        "token_pack": pack,
        "image_reading":{"kind":"plate","width":64,"height":64},
        # canonical bulk outside the pack, so bulk-by-name has work to do
        "waveform": [round(0.5*((k*7)%13)/13.0-0.25,5) for k in range(20000)],
        "spectrogram": [[round(0.01*((r*31+c*7)%17),4) for c in range(50)] for r in range(40)],
    }
    full_pack=_large_full_pack()
    changed=[]
    for i in range(900):
        changed.append({"path": f"evidence.measurements.segment_{i}.reading.pitch_hz",
                        "from": 100.0+i, "to": 101.0+i,
                        "segment": i, "note":"A4", "detail": "changed measurement for segment %d in the stored record"%i})
    findings=[]
    for i in range(250):
        findings.append({"kind":"dynamics_off","where":f"part lead start {i}.0 note A4","written":{"part":"lead","start":float(i),"midi":69},
                         "heard":{"difference_db":-5.0,"attack_loudness_db":-20.0},"statement":f"finding {i} for the stored comparison record"})
    evidence={
        "summary": summary,
        "measurements": measurements,
        "token_pack": copy.deepcopy(pack),
        "full": {"render_path":"large-evidence-render.wav","perceived_at":"2026-01-01T00:00:00+00:00","instrument":"piano",
                 "measurements": copy.deepcopy(measurements),
                 "token_pack": full_pack,
                 "reading":{"sentence":summary,"note":"A4"},
                 "harmonic":{"pitch_hz":440.05,"midi":69},
                 "interpretation": None},
        "render_path":"large-evidence-render.wav",
        "instrument":"piano",
        "interpretation": None,
        "invariants":{"checked":[
            {"id":"notes_heard_of_written","description":"Per part, the notes heard are reported against the notes written.","held":False,
             "value":[{"part":"Melody","notes_written":76,"notes_heard":76},{"part":"Harmony","notes_written":72,"notes_heard":72},{"part":"Bass","notes_written":24,"notes_heard":24}],
             "violations":[]},
            {"id":"notes_in_instrument_range","description":"Every written note lies in the instrument range.","held":True,
             "value":[{"part":"Melody","out_of_range":0}],"violations":[]},
            {"id":"note_count_per_part","description":"Note counts per part match the written score.","held":True,
             "value":[{"part":"Melody","notes_written":76},{"part":"Harmony","notes_written":72}],"violations":[]},
        ],"summary":{"total_checked":3,"held":2,"broken":1}},
        "reliability":{"instrument":"Sensoryne hearing","limits":[
            "Sample rate and FFT resolution are those of the handed-in render and the cochlea analysis block.",
            "Fundamental pitch is the strongest interpolated peak in the pitch band.",
            "Syllable and formant figures are estimates from the stated phonetic methods.",
            "The hearing works on handed-in renders only; there is no live capture.",
            "Comparison pitch tolerance is 50 cents; a written note is counted heard when its harmonic energy is at least 0.15 of the strongest."]},
        "objective_coverage":[{"item":"part Melody: 76 of 76 notes heard","covered":True},
                               {"item":"part Harmony: 72 of 72 notes heard","covered":True},
                               {"item":"part Bass: 24 of 24 notes heard","covered":True},
                               {"item":"instrument range piano","covered":True}],
        "comparison":{"findings":findings,"summary":{"total":len(findings),"counts":{"dynamics_off":len(findings)}},
                      "per_part":[{"part":"lead","notes_written":76,"notes_heard":76,"notes_not_heard":0}],
                      "tolerances":{"pitch_tolerance_cents":50.0,"dynamics_threshold_db":4.0}},
        "robustness":{"variations":[{"tempo":0.95,"spread_db":1.2},{"tempo":1.05,"spread_db":1.4}],"spread":{"tempo_db":1.3,"velocity_db":0.8}},
        "delta":{"from_version":"v1","to_version":"v2","changed_measurements":changed,
                 "added_measurements":[],"removed_measurements":[],"unchanged_keys":["summary","measurements.pitch_hz"],
                 "structural_diff":{"added":0,"removed":0}},
    }
    start={"event":"session_start","session_id":"large-evidence-session","artifact_id":"large-evidence-artifact",
           "condition":"perceptual",
           "objective":{"goal":"a symphony-length piano work"},
           "faults":[],"ground_truth":[],
           "structure_record":{"instrument":"piano","transposition":"non-transposing",
                               "written_range":{"low":{"name":"A0","midi":21,"freq_hz":27.5},"high":{"name":"C8","midi":108,"freq_hz":4186.0},"low_midi":21,"high_midi":108},
                               "parts":[{"name":"Melody","notes":76},{"name":"Harmony","notes":72},{"name":"Bass","notes":24}],
                               "duration_seconds":73.1,
                               "notes":[{"part":"Melody","start":float(i*0.9),"midi":69+(i%8),"dur":0.9} for i in range(76)]},
           "input_summary":[{"part":"Melody","start":float(i*0.9),"midi":69+(i%8),"dur":0.9} for i in range(76)]}
    return evidence, start

def _music_evidence(tmpdir):
    """Build the large stored evidence in a temporary directory.

    The document is written as a session history record
    (session_start, then perception) inside ``tmpdir`` and
    read back from that file, so the large case runs on a
    fresh clone with no outside fixture.
    """
    evidence, start = _build_large_evidence()
    session_dir = os.path.join(tmpdir, "large-evidence-session")
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, "history.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(start) + "\n")
        handle.write(json.dumps({"event": "perception", "evidence": evidence}) + "\n")
    read_start = None
    read_evidence = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("event") == "session_start":
                read_start = entry
            elif entry.get("event") == "perception":
                read_evidence = entry.get("evidence")
    return read_evidence, read_start


class StubApp:
    NAME = "stub"

    def validate(self, artifact):
        if not isinstance(artifact, dict) or "value" not in artifact:
            return ["artifact must hold a value"]
        value = artifact["value"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return ["value must be a non-negative integer"]
        return []

    def composition_prompt(self, objective):
        return "compose a stub artifact"

    def composition_retry_prompt(self, objective, errors, rejected):
        return "retry: %s" % "; ".join(errors)

    def control_prompt(self, objective, artifact):
        return "control prompt"

    def perceptual_prompt(self, objective, evidence, input_summary, structure_record):
        return "perceptual prompt"

    def input_summary(self, artifact):
        return copy.deepcopy(artifact)

    def structure_record(self, artifact):
        return {"value": artifact.get("value"), "kind": "stub"}

    def generate_ground_truth(self, faults):
        return []

    def plant_faults(self, artifact):
        return []

    def perceive(self, render_record):
        artifact = render_record.get("_artifact") or {}
        value = artifact.get("value", 0)
        # A large synthetic payload alongside the small stated
        # figures, so shaping has bulk to reference-replace.
        return {
            "summary": "value %s" % value,
            "measurements": {
                "value": value,
                "double": value * 2,
                "waveform": list(range(20000)),
            },
            "invariants": {
                "checked": [{"id": "value_ok", "held": True, "value": value}],
                "summary": {"total_checked": 1, "held": 1, "broken": 0},
            },
            "reliability": {"level": "high"},
            "objective_coverage": [{"goal": "value matters", "covered": True}],
            "full": {},
        }

    def apply_revision(self, artifact, revision):
        if not isinstance(revision, dict) or revision.get("op") != "set_value":
            raise ValueError("unknown revision op")
        value = revision.get("value")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("value must be a non-negative integer")
        revised = copy.deepcopy(artifact)
        revised["value"] = value
        return revised

    def apply_revisions(self, artifact, revisions):
        current = copy.deepcopy(artifact)
        for revision in revisions or []:
            current = self.apply_revision(current, revision)
        return current


def _stub_renderer(artifact, out_path, faults):
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("value=%s\n" % artifact.get("value"))
    return {"path": str(out_path), "kind": "text"}


def _open_stub(tmpdir):
    store = ArtifactStore(os.path.join(tmpdir, "artifacts"))
    registry = ToolRegistry()
    registry._session_counter = 0
    registry.pending_context = {
        "store": store,
        "app": StubApp(),
        "renderer": _stub_renderer,
        "session_root": os.path.join(tmpdir, "sessions"),
        "session_id": "stub-session",
        "artifact_id": "stub-artifact",
        "objective": {"goal": "value matters"},
        "condition": "perceptual",
        "test": False,
        "faults": [],
        "max_iterations": 4,
        "ledger_path": None,
    }
    controller.open_session(
        registry,
        {
            "application": "stub",
            "objective": {"goal": "value matters"},
            "condition": "perceptual",
            "test": False,
        },
    )
    return registry, registry.get("stub-session")


class ShapeFunctionTests(unittest.TestCase):
    def test_small_evidence_passes_through_unshaped_in_substance(self):
        evidence = {
            "summary": "a short heard summary",
            "measurements": {"value": 3, "double": 6},
            "invariants": {
                "checked": [{"id": "value_ok", "held": True, "value": 3}],
                "summary": {"total_checked": 1, "held": 1, "broken": 0},
            },
            "reliability": {"level": "high"},
            "objective_coverage": [{"goal": "g", "covered": True}],
            "delta": {"from_version": "v1", "to_version": "v2"},
        }
        original = copy.deepcopy(evidence)
        shaped = shape_evidence(evidence)
        # Input is not mutated and every original section is
        # intact in the shaped copy.
        self.assertEqual(evidence, original)
        for key in original:
            self.assertEqual(shaped[key], original[key])
        self.assertTrue(shaped["shaped_copy"])
        self.assertIn("shaped copy", shaped["shaping_note"])
        self.assertIn("session record", shaped["complete_evidence_location"])
        self.assertLess(_size(shaped), EVIDENCE_BUDGET)

    def test_shaping_never_mutates_its_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evidence, _start = _music_evidence(tmpdir)
            before = _size(evidence)
            shaped = shape_evidence(evidence)
            self.assertEqual(_size(evidence), before)
            self.assertLess(_size(shaped), before)


class MusicEvidenceShapingTests(unittest.TestCase):
    def test_large_music_evidence_returns_bounded_while_stored_stays_complete(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        evidence, start = _music_evidence(tmp.name)
        original_size = _size(evidence)
        self.assertGreater(original_size, 10 * EVIDENCE_BUDGET)

        shaped = shape_evidence(evidence)

        # The stored evidence is untouched at full size and the
        # shaped copy states that it is a shaped copy.
        self.assertEqual(_size(evidence), original_size)
        self.assertTrue(shaped["shaped_copy"])
        self.assertIn("shaped copy", shaped["shaping_note"].lower())
        self.assertIn("session record", shaped["complete_evidence_location"])

        # The pack is exempt and whole: byte-identical to the
        # stored pack, every segment in full, no reference
        # anywhere inside it. Other sections take any reduction.
        shaped_pack = shaped["measurements"]["token_pack"]
        original_pack = evidence["measurements"]["token_pack"]
        self.assertNotIn("_reference", shaped_pack)
        self.assertEqual(
            json.dumps(shaped_pack, sort_keys=True, ensure_ascii=False, default=str),
            json.dumps(original_pack, sort_keys=True, ensure_ascii=False, default=str),
        )
        shaped_segments = shaped_pack["segments_data"]
        original_segments = original_pack["segments_data"]
        self.assertEqual(len(shaped_segments), len(original_segments))
        for shaped_seg, original_seg in zip(shaped_segments, original_segments):
            self.assertNotIn("_reference", shaped_seg)
            self.assertEqual(shaped_seg, original_seg)
        # No reference anywhere inside the pack subtree.
        self.assertNotIn("_reference", json.dumps(shaped_pack, default=str))

        # Every other required section survived shaping in full
        # or as a bounded reference-replaced subtree, and the
        # non-pack shaped copy stays bounded: the stored record's
        # non-pack bulk (the full repeat copies) is replaced.
        self.assertEqual(shaped["summary"], evidence["summary"])
        for key in ("pitch_hz", "loudness", "duration"):
            if key in evidence["measurements"]:
                self.assertEqual(shaped["measurements"][key], evidence["measurements"][key])
        self.assertEqual(
            [c["id"] for c in shaped["invariants"]["checked"]],
            [c["id"] for c in evidence["invariants"]["checked"]],
        )
        self.assertEqual(
            [c["held"] for c in shaped["invariants"]["checked"]],
            [c["held"] for c in evidence["invariants"]["checked"]],
        )
        self.assertEqual(shaped["reliability"], evidence["reliability"])
        self.assertEqual(shaped["objective_coverage"], evidence["objective_coverage"])
        # The repeat copies of the measurements are references;
        # the stored record keeps them complete.
        self.assertIn("_reference", shaped["full"]["measurements"])
        self.assertNotIn("_reference", evidence["full"]["measurements"])
        self.assertNotIn("shaped_copy", evidence)

        # The whole tool-result document shapes the same way:
        # the pack stays whole, structure and input kept in full.
        document = {
            "valid": True,
            "session_id": "large-evidence-session",
            "iteration": 1,
            "evidence": copy.deepcopy(evidence),
            "structure_record": copy.deepcopy(start["structure_record"]),
            "input_summary": copy.deepcopy(start["input_summary"]),
        }
        shaped_doc = shape_document(document)
        self.assertTrue(shaped_doc["shaped_copy"])
        doc_pack = shaped_doc["evidence"]["measurements"]["token_pack"]
        self.assertEqual(
            json.dumps(doc_pack, sort_keys=True, ensure_ascii=False, default=str),
            json.dumps(original_pack, sort_keys=True, ensure_ascii=False, default=str),
        )
        self.assertEqual(shaped_doc["structure_record"], start["structure_record"])
        self.assertEqual(shaped_doc["input_summary"], start["input_summary"])


class ReturnBoundaryTests(unittest.TestCase):
    def test_submit_work_returns_shaped_copy_and_stores_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context = _open_stub(tmpdir)
            result = controller.submit_work(context, {"value": 2})
            self.assertTrue(result["shaped_copy"])
            self.assertIn("shaped copy", result["shaping_note"].lower())
            evidence = result["evidence"]
            self.assertEqual(evidence["summary"], "value 2")
            self.assertEqual(evidence["measurements"]["value"], 2)
            # Bulk waveform in the returned copy is a reference.
            self.assertIn("_reference", evidence["measurements"]["waveform"])
            # The stored record keeps the complete evidence.
            perception = [
                e for e in context["history"].entries if e.get("event") == "perception"
            ]
            stored = perception[-1]["evidence"]
            self.assertNotIn("shaped_copy", stored)
            self.assertEqual(stored["measurements"]["waveform"], list(range(20000)))

    def test_submit_decision_returns_shaped_copy_with_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _registry, context = _open_stub(tmpdir)
            controller.submit_work(context, {"value": 1})
            result = controller.submit_decision(
                context,
                {
                    "faults": [],
                    "revisions": [{"op": "set_value", "value": 4}],
                    "complete": False,
                    "assessment": "continue",
                },
            )
            self.assertTrue(result["shaped_copy"])
            self.assertEqual(result["evidence"]["delta"]["to_version"], "v2")
            self.assertEqual(result["evidence"]["measurements"]["value"], 4)
            perception = [
                e for e in context["history"].entries if e.get("event") == "perception"
            ]
            self.assertNotIn("shaped_copy", perception[-1]["evidence"])

    def test_probe_returns_shaped_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry, context = _open_stub(tmpdir)
            controller.submit_work(context, {"value": 1})
            result = dispatch(
                registry,
                "sensoryne_probe",
                {
                    "session_id": context["session_id"],
                    "decision": {
                        "faults": [],
                        "revisions": [{"op": "set_value", "value": 5}],
                        "complete": False,
                        "assessment": "hypothetical",
                    },
                },
            )
            self.assertTrue(result["probe"])
            self.assertTrue(result["shaped_copy"])
            self.assertEqual(result["measurements"]["value"], 5)
            self.assertEqual(result["delta"]["to_version"], "v2-probe")


if __name__ == "__main__":
    unittest.main()
