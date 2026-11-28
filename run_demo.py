"""run_demo.py - thin host for Sensoryne demonstrations.

The host builds the run directory, the artifact store, and the
ledger. For each demonstration session it sends one opening
request to the Responses API: the tools array, and an opening
task naming the application, the objective, the condition, and
whether the session is a test session. That opening task is
the only prompt the host composes. After that the host
executes tool calls only: each function_call the model returns
is dispatched to the platform operation it names, and the
result is returned as a function_call_output under the same
response chain. The host composes no evaluation prompts,
assembles no session-so-far text of its own, and decides
nothing.
"""

import argparse
import copy
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Host-side wiring for the music renderer in the sibling
# embers-piano project (see mcp_server.py). The piano and its
# renderer are a separate project beside this folder; adding
# the sibling to sys.path lets "music_renderer" resolve there
# without copying piano code into this project.
_SIBLING = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "embers-piano",
)
if os.path.isdir(_SIBLING) and _SIBLING not in sys.path:
    sys.path.append(_SIBLING)

from stage2.ember import ResponsesClient, SpendLedger, function_call_output
from stage2.report import build_report
from stage2.store import ArtifactStore
from stage2.tools import TOOLS, ToolRegistry, dispatch

APPS = {
    "music": ("stage2.apps.music.app", "music_renderer"),
    "poetry": ("stage2.apps.poetry.app", "poetry_renderer"),
    "story": ("stage2.apps.story.app", "story_renderer"),
    "drawing": ("stage2.apps.drawing.app", "drawing_renderer"),
    "code": ("stage2.apps.code.app", "code_renderer"),
    "typography": ("stage2.apps.typography.app", "typography_renderer"),
    "animation": ("stage2.apps.animation.app", "animation_renderer"),
    "argument": ("stage2.apps.argument.app", "argument_renderer"),
}

OBJECTIVES = {
    "music": (
        "A balanced three-part piano study. The melody leads and stays "
        "clearly audible throughout; the harmony supports it at a lower "
        "level; the bass grounds the piece without covering either part. "
        "Dynamics stay even, with no swells or sudden changes the piece "
        "does not call for. Every written note sounds."
    ),
    "poetry": {
        "line_count": 12,
        "stanzas": 3,
        "syllables_per_line": 10,
        "rhyme_pairs": [[2, 4], [6, 8], [10, 12]],
    },
    "story": {
        "section_count": 5,
        "movements": ["opening", "rise", "climax", "fall", "ending"],
        "promised_events": [
            "a door opens",
            "a letter arrives",
            "a choice is made",
        ],
    },
    "drawing": {
        "intent": (
            "A simple scene: a circle sun over a rectangular hill, "
            "with a small text label."
        ),
        "elements": [
            {
                "id": "sun",
                "kind": "shape",
                "description": "circle in the upper left",
            },
            {
                "id": "hill",
                "kind": "shape",
                "description": "rectangle across the bottom",
            },
            {
                "id": "label",
                "kind": "text",
                "description": "text label near the hill",
            },
        ],
    },
    "code": (
        "A program that reads a line on standard input and prints it "
        "back, upper-cased, followed by its length."
    ),
    "typography": {
        "intent": (
            "A one-page layout with a heading, body text, and a "
            "caption, aligned to a shared left edge, with the "
            "heading larger than the body and the body larger than "
            "the caption."
        ),
        "page": {"width": 800, "height": 1000},
        "kinds": ["heading", "body", "caption"],
    },
    "animation": {
        "intent": (
            "A short scene in which one element moves across the "
            "canvas from the first keyframe to the last."
        ),
        "duration": 2.0,
        "frame_rate": 12,
        "canvas": {"width": 320, "height": 240},
    },
    "argument": {
        "claim_count": 4,
        "kinds": ["fact", "logic", "value"],
    },
}

_MAX_TURNS = 60


def _message(text):
    return {"role": "user", "content": text}


def _opening_task(application, objective, condition, test):
    return (
        "Open a session with sensoryne_open_session and then do "
        "the work through the tools. The tools are how the work "
        "is submitted, tested, and finished: compose inside "
        "your own session, submit the work with "
        "sensoryne_submit_work, judge what comes back, submit "
        "your decision with sensoryne_submit_decision, and "
        "declare completion with sensoryne_finish_session. "
        "sensoryne_probe applies a decision's revisions to a "
        "throwaway copy and returns hypothetical evidence without "
        "creating a version, and sensoryne_recall returns stored "
        "pairs from earlier sessions that ended model_complete "
        "for an application.\n"
        "application: %s\n"
        "objective: %s\n"
        "condition: %s\n"
        "test: %s"
        % (
            application,
            json.dumps(objective, ensure_ascii=False)
            if not isinstance(objective, str)
            else objective,
            condition,
            "true" if test else "false",
        )
    )


def run_session_host(
    client, registry, session_id, application, objective, condition, test, context
):
    """Drive one session: one opening request, then tool calls only."""
    registry.pending_context = context
    input_items = [
        _message(_opening_task(application, objective, condition, test))
    ]
    result = client.call(
        input_items,
        session_id=session_id,
        purpose="tool_loop",
        tools=TOOLS,
        previous_response_id=None,
    )
    print(
        "  [%s] opening call cost $%.4f; cumulative $%.4f"
        % (session_id, result["cost_usd"], client.ledger.total())
    )
    turns = 0
    while result.get("function_calls") and turns < _MAX_TURNS:
        turns += 1
        output_items = []
        for call in result["function_calls"]:
            try:
                tool_result = dispatch(registry, call["name"], call["arguments"])
            except Exception as exc:
                tool_result = {"error": str(exc)}
            output_items.append(
                function_call_output(call["call_id"], tool_result)
            )
        session_context = registry.get(session_id)
        result = client.call(
            output_items,
            session_id=session_id,
            purpose="tool_loop",
            tools=TOOLS,
            previous_response_id=result["response_id"],
        )
        print(
            "  [%s] turn %d cost $%.4f; cumulative $%.4f"
            % (session_id, turns, result["cost_usd"], client.ledger.total())
        )
        if session_context.get("ended_by") is not None and not result.get(
            "function_calls"
        ):
            break
    return registry.get(session_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sensoryne demo host")
    parser.add_argument("--app", choices=sorted(APPS), default="music")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    app_module, renderer_module = APPS[args.app]
    app = importlib.import_module(app_module)
    renderer = importlib.import_module(renderer_module).renderer
    base = os.path.dirname(os.path.abspath(__file__))
    run_id = args.run_id or ("%s-run" % args.app)
    run_dir = os.path.join(base, "sessions", run_id)
    sessions_root = os.path.join(run_dir, "sessions")
    store_root = os.path.join(run_dir, "artifacts")
    os.makedirs(sessions_root, exist_ok=True)
    os.makedirs(store_root, exist_ok=True)
    ledger = SpendLedger(os.path.join(run_dir, "spend.jsonl"))
    print("Spend so far: $%.4f" % ledger.total())
    store = ArtifactStore(store_root)
    objective = copy.deepcopy(OBJECTIVES[args.app])
    artifact_id = "%s-demo" % args.app
    registry = ToolRegistry()
    registry._session_counter = 0
    client = ResponsesClient(ledger, tools=TOOLS)
    sessions = [
        ("%s-control-faulted" % args.app, "control", True),
        ("%s-perceptual-faulted" % args.app, "perceptual", True),
        ("%s-perceptual-clean" % args.app, "perceptual", False),
    ]
    for session_id, condition, test in sessions:
        context = {
            "store": store,
            "app": app,
            "renderer": renderer,
            "session_root": sessions_root,
            "session_id": session_id,
            "artifact_id": artifact_id,
            "objective": copy.deepcopy(objective),
            "condition": condition,
            "test": test,
            "faults": [],
            "max_iterations": 4,
            "ledger_path": ledger.path,
        }
        session_context = run_session_host(
            client,
            registry,
            session_id,
            args.app,
            objective,
            condition,
            test,
            context,
        )
        report = build_report(session_context["session_dir"], ledger.path)
        print(
            "Session %s ended by %s; report %s"
            % (session_id, session_context.get("ended_by"), report)
        )
    print("Run outputs under %s" % run_dir)
    print("Final total spend: $%.4f" % ledger.total())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
