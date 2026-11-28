"""mcp_server.py - Sensoryne MCP server.

A Model Context Protocol server over stdio that exposes
Sensoryne's ten tools, seven for the working model and three for the judge, to a host. The server is a front door
only: it lists the tools, passes arguments through unchanged,
dispatches each call to stage2.tools.dispatch, and returns
the result unchanged. It opens no sessions, imposes no order,
sets no defaults inside arguments, and decides nothing.

Start it with:

    python3 mcp_server.py

It adds one dependency: the official mcp Python package,
version 2.3.0.

Tools exposed (names, descriptions, and parameter schemas
are taken from stage2.tools.TOOLS at startup):

- sensoryne_open_session
- sensoryne_submit_work
- sensoryne_submit_decision
- sensoryne_finish_session
- sensoryne_probe
- sensoryne_recall
- sensoryne_resume_session
- sensoryne_judge_list_sessions
- sensoryne_judge_open
- sensoryne_judge_verdict

One stage2.tools.ToolRegistry is held for the server's whole
lifetime. A tool error from dispatch is returned as an MCP
tool error carrying the exception's message; the server does
not crash. This file reads no credentials and makes no
network calls.

Music rendering is wired on the host side: the piano and its
renderer are a separate project that sits beside this project
folder as ``embers-piano``. When that sibling folder exists,
it is added to ``sys.path`` so the controller's renderer
lookup (``music_renderer``) resolves to the sibling's
renderer, which in turn imports ``piano`` from the same
folder. No piano code is copied into this project. When the
sibling is absent, music sessions fail with the import error
from the renderer lookup, as before.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Host-side wiring for the music renderer in the sibling
# embers-piano project. The sibling sits beside this project
# folder; adding it to sys.path lets "import music_renderer"
# (and its "from piano import ..." ) resolve there.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SIBLING = os.path.join(os.path.dirname(_HERE), "embers-piano")
if os.path.isdir(_SIBLING) and _SIBLING not in sys.path:
    sys.path.append(_SIBLING)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

from stage2.tools import TOOLS as _CORE_TOOLS
from stage2.tools import ToolRegistry, dispatch

SERVER_NAME = "sensoryne"

_registry = ToolRegistry()

server = Server(SERVER_NAME)


def _tool_definitions():
    definitions = []
    for entry in _CORE_TOOLS:
        definitions.append(
            Tool(
                name=entry["name"],
                description=entry["description"],
                inputSchema=entry["parameters"],
            )
        )
    return definitions


async def _handle_list_tools(ctx, params):
    return ListToolsResult(tools=_tool_definitions())


async def _handle_call_tool(ctx, params):
    name = params.name
    arguments = params.arguments or {}
    try:
        result = dispatch(_registry, str(name), arguments)
    except Exception as exc:
        return CallToolResult(
            content=[TextContent(type="text", text=str(exc))],
            isError=True,
        )
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))],
        isError=False,
    )


server.add_request_handler("tools/list", ListToolsRequest, _handle_list_tools)
server.add_request_handler("tools/call", CallToolRequestParams, _handle_call_tool)


async def _main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
