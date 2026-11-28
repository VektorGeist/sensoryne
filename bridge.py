"""bridge.py - the bridge between Muse Ember and the MCP servers.

A user directs Muse Ember to create something. Ember is the
maker. While it works, Ember calls tools: Sensoryne through
its MCP front door (mcp_server.py, which exposes the nine
sensoryne_ tools). This bridge carries the user's
direction to Ember, carries Ember's tool calls to the
servers, and carries the results back.

The bridge decides nothing: it invents no instructions,
opens no sessions, imposes no order, plants nothing, and
ends nothing. It sends Ember nothing beyond the user's
direction and the tool outputs. No system prompt, no extra
instruction, no summary of the tools written by the bridge.
Ember reads the tool definitions themselves. The bridge
never writes the API key anywhere; stage2/ember.py reads
MODEL_API_KEY from the environment.

Start it with:

    python3 bridge.py --direction "the user's direction, in the user's words"
    python3 bridge.py --direction-file direction.txt

Dependencies are this project's own only: the official mcp
Python package, version 2.3.0, numpy, scipy, Pillow, and
cairosvg when installed (used by the drawing renderer when
installed; otherwise drawing renders raise). Model calls go through stage2.ember.ResponsesClient
and SpendLedger, so the model (muse-ember-1.0), its default
reasoning mode, the cost accounting, and the existing spend
cap apply exactly as built.

Artifacts of record: the trace at bridge_trace.jsonl in the
working directory by default, and the spend ledger at
bridge_spend.jsonl by default. Together they are the run's
record.
"""

import argparse
import asyncio
import json
import os
import shlex
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from stage2.ember import ResponsesClient, SpendLedger, function_call_output

_HERE = os.path.dirname(os.path.abspath(__file__))


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


class Trace:
    """Append-only JSONL trace, written as each event happens."""

    def __init__(self, path):
        self.path = str(path)
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        if not os.path.exists(self.path):
            open(self.path, "a", encoding="utf-8").close()

    def write(self, event, **fields):
        entry = {"event": event, "ts": _utc_now()}
        entry.update(fields)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry


def _server_params(command_line, default_directory):
    parts = shlex.split(command_line)
    if not parts:
        raise ValueError("a server command line is required")
    command = parts[0]
    args = parts[1:]
    cwd = default_directory
    for arg in args:
        if arg.endswith(".py"):
            directory = os.path.dirname(os.path.abspath(arg))
            if os.path.isdir(directory):
                cwd = directory
            break
    return StdioServerParameters(command=command, args=args, cwd=cwd)


def _content_text(result):
    texts = []
    for item in getattr(result, "content", []) or []:
        if getattr(item, "type", None) == "text" and getattr(item, "text", None):
            texts.append(str(item.text))
    return "\n".join(texts)


def _tool_definitions(tool):
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
    }


def _total_tokens(ledger):
    total = 0
    for entry in ledger.entries():
        usage = entry.get("usage") or {}
        total += int(usage.get("input_tokens", 0) or 0)
        total += int(usage.get("output_tokens", 0) or 0)
    return total


async def _run(direction, sensoryne_command, piano_command, trace, ledger):
    """Run the conversation with both servers open throughout."""
    sessions = {}
    owners = {}
    definitions = []
    openers = []

    async def _open(server_id, command_line):
        params = _server_params(command_line, _HERE)
        stack = stdio_client(params)
        read, write = await stack.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()
        result = await session.list_tools()
        for tool in result.tools:
            definitions.append(_tool_definitions(tool))
            owners[tool.name] = server_id
        sessions[server_id] = session
        openers.append(stack)

    async def _close():
        for stack in reversed(openers):
            try:
                await stack.__aexit__(None, None, None)
            except Exception:
                pass

    await _open("sensoryne", sensoryne_command)
    if piano_command:
        await _open("piano", piano_command)

    try:
        client = ResponsesClient(ledger, tools=definitions)
        trace.write("start", direction=direction, tools=[d["name"] for d in definitions])
        input_items = [{"role": "user", "content": direction}]
        result = client.call(
            input_items,
            session_id="bridge",
            purpose="tool_loop",
            tools=definitions,
            previous_response_id=None,
        )
        trace.write(
            "model_reply",
            response_id=result.get("response_id"),
            usage=result.get("usage"),
            cost_usd=result.get("cost_usd"),
            output=result.get("output"),
        )
        while result.get("function_calls"):
            outputs = []
            for call in result["function_calls"]:
                name = call.get("name")
                arguments = call.get("arguments") or {}
                session = sessions.get(owners.get(name))
                if session is None:
                    text = "no MCP server owns the tool %r" % (name,)
                    is_error = True
                else:
                    tool_result = await session.call_tool(name, arguments)
                    text = _content_text(tool_result)
                    is_error = bool(getattr(tool_result, "is_error", False))
                trace.write(
                    "tool_call",
                    server=owners.get(name),
                    name=name,
                    arguments=arguments,
                    result=text,
                    is_error=is_error,
                )
                outputs.append(function_call_output(call.get("call_id"), text))
            result = client.call(
                outputs,
                session_id="bridge",
                purpose="tool_loop",
                tools=definitions,
                previous_response_id=result.get("response_id"),
            )
            trace.write(
                "model_reply",
                response_id=result.get("response_id"),
                usage=result.get("usage"),
                cost_usd=result.get("cost_usd"),
                output=result.get("output"),
            )
    finally:
        await _close()

    calls = ledger.session_call_count("bridge")
    tokens = _total_tokens(ledger)
    spend = ledger.total()
    trace.write(
        "end",
        calls=calls,
        total_tokens=tokens,
        total_spend_usd=spend,
    )
    final_text = result.get("text")
    if final_text:
        print(str(final_text))
    print("Model calls: %d" % calls)
    print("Total tokens: %d" % tokens)
    print("Total spend: $%.4f" % spend)
    return 0


def _resolve_direction(args):
    if bool(args.direction) == bool(args.direction_file):
        raise SystemExit(
            "exactly one of --direction and --direction-file is required"
        )
    if args.direction:
        return args.direction
    with open(args.direction_file, "r", encoding="utf-8") as handle:
        return handle.read()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bridge between Muse Ember and the MCP servers")
    parser.add_argument("--direction", default=None)
    parser.add_argument("--direction-file", default=None)
    parser.add_argument("--sensoryne-server", default="python3 mcp_server.py")
    parser.add_argument("--piano-server", default=None)
    parser.add_argument("--trace", default=os.path.join(os.getcwd(), "bridge_trace.jsonl"))
    parser.add_argument("--ledger", default=os.path.join(os.getcwd(), "bridge_spend.jsonl"))
    args = parser.parse_args(argv)
    direction = _resolve_direction(args)
    trace = Trace(args.trace)
    ledger = SpendLedger(args.ledger)
    asyncio.run(
        _run(direction, args.sensoryne_server, args.piano_server, trace, ledger)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
