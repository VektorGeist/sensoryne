# Sensoryne

## Project Overview

Sensoryne is an independent perceptual evaluation and improvement platform, and a creative process for AI. A model makes a creative work. The finished render is handed in. An independent instrument perceives the render and returns evidence. The model criticizes the render against the work's structure record and its objective and revises in addressed operations, repeating until the model judges the work complete. The instrument perceives, the model evaluates and decides, the platform coordinates and preserves the record. Sensoryne makes nothing and hears nothing. Rendering happens on the composing side. The finished render is handed to Sensoryne.

The platform contains no scoring algorithm. Instruments report the measurable layer only. Every evidence document carries `interpretation: null`. Evidence keeps the outer shape of summary, measurements, and full record, with additional blocks described below.

## Problem

A model that only sees its own source representation cannot detect defects that exist only in what that source renders to. In music, a score can be written one way while the render differs in a part, a passage, or a written note. In text, a delivered page can omit a written line. In a page layout, a drawn picture, an animation, a sound work, or a delivered program, the delivered form can differ from the written form. Evaluation requires an independent sense: something that perceives the rendered output and reports what it perceives, leaving judgment to the model.

## Solution

The model is connected to the platform as a client through the Meta Model API and calls six tools while it works: `sensoryne_open_session`, `sensoryne_submit_work`, `sensoryne_submit_decision`, `sensoryne_finish_session`, `sensoryne_probe`, and `sensoryne_recall`. The host sends one opening task per session and then executes tool calls only.

The loop, with source kept distinct from render at every step:

1. `sensoryne_open_session` takes application, objective, condition, and test. The platform creates the session record and artifact store entry and returns the application's composition prompt text. In a test session, demo faults are planted from the application's `plant_faults` and ground truth is generated from the fault definitions at planting time.
2. `sensoryne_submit_work` takes session_id and artifact. The platform validates with the application's `validate`. On validation failure the result returns the failed checks and the retry prompt text, and no version is stored. On success it stores version 1, computes the structure record, renders through the session's renderer with the session's faults applied in the rendering on the composing side, and perceives the render. The perceptual condition returns the evidence document, the structure record, and the input summary of the work as originally written. The control condition returns the structure record and the artifact as written, records `perception_skipped`, and returns no evidence.
3. The model submits a decision with `sensoryne_submit_decision` in the platform decision shape: `{"faults": [{"description": "...", "where": "..."}], "revisions": [...], "complete": true or false, "assessment": "..."}`. The platform records the decision, applies revisions through the application's revision engine, records each applied and rejected operation with its error, stores the result as a new version with its structure record, re-renders, re-perceives, and returns new evidence, the new structure record, the applied and rejected lists, and the session so far. Revisions are applied before any termination is recorded, so an applied revision is always rendered.
4. Termination labels are exact: `model_complete` only when the decision's complete flag is true, `model_no_revisions` only when the decision carries no revisions and is not complete, `iteration_limit` at the iteration limit, and `decision_unparseable` when no well-formed decision can be parsed. `sensoryne_finish_session` ends a session as `model_complete`.
5. Revisions are addressed operations, never a free rewrite. Operation names and references are checked, numeric values are finite, ranges are enforced, the resulting artifact is validated as a whole, and a rejected operation leaves the current artifact unchanged. No-op and repeated revisions are detected by artifact hash and rejected. Versions are never overwritten, and each version carries its own copy of the source it was rendered from, checked against a hash of the rendered source. Versions are identified everywhere by the store's version identifier, `v1`, `v2`, and so on, including at both ends of a delta.

Additional platform records, computed by the platform and stored with each version:

- **Delta evidence.** After a revision, a perceptual version's evidence carries a `delta` key: `from_version` and `to_version` in the store's version identifiers, `changed_measurements` sorted by dotted path rooted at `measurements`, each entry holding `before`, `after`, and for numeric leaves `absolute_delta` and, when before is not zero, `relative_delta`, `added_measurements` and `removed_measurements` as sorted dotted paths rooted at `measurements`, `unchanged_keys` as sorted measurement paths rooted at `measurements`, and `structural_diff` with added, removed, and modified dotted paths only. A measurement that appears or disappears is listed under added or removed, never also under changed. Boolean values are leaves, never numbers. A first version has no delta. A control version carries no delta, because nothing was perceived. A rejected, no-op, or repeated revision produces no version and no delta.
- **Invariant checks.** Each application declares invariants. Every evidence document carries an `invariants` block with `checked` entries holding `id`, `description`, `held`, `value`, and `violations` where applicable, plus a `summary` with total, held, and broken counts. Invariants are evaluated on every version.
- **Reliability metadata.** Every evidence document carries a `reliability` block stating the instrument's name and its known limits as facts about the instrument.
- **Objective coverage.** Every evidence document carries an `objective_coverage` list: each detectable part of the session's stated objective, in the objective's own terms, with `item` and `present`. Where an objective states nothing detectable, the list is empty.
- **Robustness perception.** The music application declares variations, tempo scaled by 0.95 and by 1.05 and all velocities scaled by 0.9. The same score is rendered under each variation, perceived by the same hearing, and reported in a `robustness` block with per-variation headline pitch and loudness figures and per-part note counts, plus the spread of each across variations. Other applications declare no variations and produce no robustness block.

Two further tools:

- `sensoryne_probe` takes session_id and a decision. Only its revisions are used. The platform applies them to a throwaway copy of the current artifact, renders and perceives the result, and returns what a submission of that hypothetical artifact returns in the session's condition, with `probe: true` at the top level. In the perceptual condition that is the evidence document, carrying a delta against the current version whose hypothetical end is named as the next version identifier with a `-probe` suffix, such as `v2-probe`. In the control condition it is the control submission shape, with no evidence and no delta. A probe creates no version, changes no stored record, and cannot end a session. An invalid probe returns validation failures and changes nothing.
- `sensoryne_recall` takes application and limit. The platform keeps, per application, the structure record and evidence summary of every session that ended `model_complete`, labeled with that termination and the session objective, in an on-disk episodic memory file. Recall returns those pairs most recent first, in both conditions.

## Applications

Sensoryne ships with eight applications on one interface. Each provides `NAME`, `validate`, `composition_prompt`, `composition_retry_prompt`, `control_prompt`, `perceptual_prompt`, `input_summary`, `structure_record`, `generate_ground_truth`, `plant_faults`, `perceive`, `apply_revision`, `apply_revisions`, and its instrument. Each has a renderer at the repository root on the composing side. Each runs the three-session pattern: control with faults, perceptual with the same faults, perceptual clean. Every renderer takes one file path as its output: it writes its primary output file exactly at that path, and derives any additional files from it, animation frames as `<stem>_frame_NNN<ext>`.

- **Music.** The hearing is built into the application in `stage2/apps/music/hearing/` with numpy and scipy only: a cochlea and analyzer taking audio in blocks, frame percepts with band energies, loudness, brightness, spectral flux, interpolated fundamental pitch, a 32-place tonotopic place vector, block energy, and spectral flatness; silent, tonal, and pitched classification and whole-reading aggregation; a plain language reading with note name and cents when pitched; a second reading of the place-by-time field rendered as an image and read back; a 78-mode Chladni plate model from a 55 Hz reference spanning 55 to 14,575 Hz; 24 by 32 trajectory matching by dynamic time warping with chi-square cost and a pitch-invariant variant; a gallery of named exemplars; episodic memory with salience and consolidation; a word lexicon with ranked cohorts; phonetics with manner fractions, formants F1 to F3, and voice onset time; utterance segmentation; a harmonic stack scoring MIDI 21 to 108 with an octave guard, harmonic pitch with confidence, and the top five polyphony candidates; an oscilloscope channel; a 20-segment token pack with Chladni and spectrogram images and a phonetics block; calibration from labeled takes; and deterministic synthetic tone, chord, sweep, noise, and click sounds. The hearing measures pitch against the note table and reports cents from exact, and flags harmonic pitch disagreement with the cochlea pitch without overwriting the main reading. It works on handed-in renders only. It does not judge, and it does not measure the model's intent.
- **Music, vocal extension.** A vocal artifact holds `kind: vocal`, lyrics as a list of lines, each line a list of syllables or words in order, and a handed-in audio render. The performer is outside the platform. The hearing reports per lyric line: median pitch named against the note table with cents deviation, pitch range, vibrato rate and extent on sustained notes, formant figures, measured onset time, and syllable events detected against the stated count; phrase and breath boundaries from energy segmentation with gap durations. Vocal invariants check every lyric line detected as a phrase, syllable event count per line against the stated count, and pitch present in every line. Vocal revisions address the lyrics only: replace a line, add a line, remove a line. A revised vocal work is handed in again as a new render.
- **Music, instrument table.** `stage2/apps/music/instruments.py` holds every instrument's keys and notes, researched from orchestration reference works and maker specifications. The note table covers every note named in any instrument range, B0 to C8, with exact pitch in Hz at A4 = 440 Hz. Entries store written and sounding ranges, practical and full tops where both are given, open strings, tunings, and named variants as variants. The music artifact names its instrument. Validation checks written notes against that instrument's written range. The structure record and evidence speak in that instrument's terms. A user names the instrument their piece was played on; the instrument itself stays outside the platform.
- **Poetry.** Artifact `{"stanzas": [[line, ...], ...]}`. The instrument is a mechanical text analyzer over delivered text: per line word count, character count, and a syllable estimate by the stated rule, end word of every line, rhyme grouping by normalized ending, repeated words with counts and line numbers, and delivered counts per stanza and in total. Every syllable count is labeled an estimate. It does not judge meter, rhyme quality, or meaning.
- **Story.** Artifact `{"sections": [{"title": "...", "text": "..."}]}`. The instrument is a text analyzer over delivered text: per section word count, sentence count, dialogue share, and share of the whole; delivered section count and order; and which promised event strings appear in the delivered text and which do not. It does not judge plot, pacing, or quality.
- **Drawing.** Artifact `{"svg": "...", "elements": [...]}`. The instrument is a raster analyzer over the rendered PNG: dimensions, dominant colors and shares, ink coverage overall and per quadrant, and the bounding box of the non-background region. It does not say what the picture shows.
- **Code.** Artifact `{"files": [{"name": "main.py", "text": "..."}], "purpose": "...", "tests": [{"name": "...", "input": "...", "expected": "..."}]}`. The instrument is execution: the delivered program runs in a temporary directory with reduced environment and disabled sockets, once per stated test, with a 10 second limit per test, input fed on standard input, recording exit status, standard output, standard error, run time, whether the limit was hit, and whether output matches expected exactly. The execution environment is not a security sandbox and is not for untrusted external use. It does not judge code quality beyond the stated tests.
- **Typography.** Artifact: a page size and elements, each with id, kind (heading, body, caption, image block), text where it has text, font size, position and box, and alignment. The renderer draws the page to an image with Pillow. The instrument measures each element's realized box, margins against the page edges, alignment consistency as how many elements share each left edge, contrast of each text element against its background, the type hierarchy as heading, body, and caption size ratios, and whitespace share of the page. It does not judge layout quality.
- **Animation.** Artifact: duration, frame rate, canvas size, and elements, each with an id and a keyframe track of time, position, size, opacity, and an easing per segment (linear, ease-in, ease-out, ease-in-out). The renderer produces frames as images. The instrument measures per-element realized positions at sampled times against the keyframes, frame-to-frame difference totals, whether the final frame matches the first for a work declared a loop, total duration, and the share of frames in which each element is visible. It does not judge motion quality.
- **Sound design.** Artifact: a duration and a list of sound events, each with id, kind (noise burst, filtered noise, sine cluster, percussive hit, silence), start, duration, gain, and kind parameters filter cutoff and bandwidth. The renderer synthesizes the events to audio with numpy. The instrument measures with non-pitch analysis: per-event realized onset and duration, spectral centroid and bandwidth per event, the loudness contour of the whole, silence share, and event overlap. It does not judge sound quality.

## Viewer

The replay viewer serves recorded sessions from all applications. It shows the sessions list, the iteration timeline, the render, the evidence summary, the measurement fields, the structure record, and the model's findings for the selected iteration, with planted ground truth behind its toggle. For any session with more than one version, the delta view shows, between consecutive versions, the changed measurements with before, after, and deltas, the added and removed paths, and the structural diff, read from the stored delta blocks. Application-specific extras are shown only when present in that session's evidence. The viewer judges nothing.

## How Muse Ember was used

Every part of this project was written by Muse Ember. Muse Ember is the only model the project's AI functionality runs on. At runtime Muse Ember is a client of the platform through the Meta Model API at `POST https://api.meta.ai/v1/responses` with the key read from `MODEL_API_KEY`. Its default reasoning mode is used and is never overridden or disabled. The Responses API carries a tools array in each request; the model answers with `function_call` items; the host executes the call and returns a `function_call_output` item under the same chain via `previous_response_id`; `tool_choice` is `auto`. The host `run_demo.py` sends one opening task per session and then executes tool calls only. It composes no evaluation prompts, assembles no session-so-far text of its own, and decides nothing.

## Setup Instructions

Dependencies are this project's own only: Python 3.9 or later, numpy, scipy, Pillow, and cairosvg for the drawing application's renderer.

```bash
pip install numpy scipy Pillow cairosvg
```

Set MODEL_API_KEY in the environment.

```bash
python3 run_demo.py --app music
python3 run_demo.py --app poetry
python3 run_demo.py --app story
python3 run_demo.py --app drawing
python3 run_demo.py --app code
python3 run_demo.py --app typography
python3 run_demo.py --app animation
python3 run_demo.py --app sound_design
```

run_demo.py runs one application per run. Each run writes into sessions/<run_id>/, holding sessions/<app>-control-faulted, sessions/<app>-perceptual-faulted, and sessions/<app>-perceptual-clean, plus artifacts/ and spend.jsonl. Each session directory holds history.jsonl, renders, and report.html.

Replay recorded sessions with the viewer; no API key is needed.

```bash
python3 -m stage2.viewer
```

Then open http://127.0.0.1:8000/.
