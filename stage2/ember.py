"""stage2/ember.py - Ember client and spend ledger.

Muse Ember (model ID muse-ember-1.0) is a client of this platform
through the Meta Model API.  All model calls go to
POST https://api.meta.ai/v1/responses with the header
Authorization: Bearer <API key>, the key read from MODEL_API_KEY.
Ember is a reasoning model: its default reasoning mode is used
and is never overridden or disabled.

This module carries two clients.  ``ask_ember`` is the text-only
call.  ``ResponsesClient`` carries the tools array and the
response chain: it sends input items, returns the model's
function_call items and text, accepts function_call_output
items, and records every API call in the ledger with its usage
and cost, as ask_ember does.
"""

import json
import os
import urllib.request
from datetime import datetime, timezone

MODEL_ID = "muse-ember-1.0"
API_URL = "https://api.meta.ai/v1/responses"
BUDGET_CAP_USD = 9.50
LONG_CONTEXT_THRESHOLD = 200000


class BudgetExceeded(RuntimeError):
    """Raised instead of calling once the budget cap is reached."""


def compute_cost_usd(usage):
    """Cost for one call from its usage dict (USD)."""
    usage = usage or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    details = usage.get("input_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens", 0) or 0) if isinstance(details, dict) else 0
    cached_tokens = min(cached_tokens, input_tokens)
    fresh_input = input_tokens - cached_tokens
    if input_tokens > LONG_CONTEXT_THRESHOLD:
        input_rate, cached_rate, output_rate = 0.30, 0.012, 1.50
    else:
        input_rate, cached_rate, output_rate = 0.10, 0.004, 0.50
    return (
        fresh_input * input_rate
        + cached_tokens * cached_rate
        + output_tokens * output_rate
    ) / 1_000_000.0


class SpendLedger:
    """An append-only JSONL spend ledger."""

    def __init__(self, path):
        self.path = str(path)
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        if not os.path.exists(self.path):
            open(self.path, "a", encoding="utf-8").close()

    def entries(self):
        result = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    result.append(json.loads(line))
        return result

    def total(self):
        return float(sum(float(e.get("cost_usd", 0.0)) for e in self.entries()))

    def record(self, session_id, purpose, model, usage, cost_usd, prompt=None, reply=None):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "purpose": purpose,
            "model": model,
            "usage": usage or {},
            "cost_usd": float(cost_usd),
            "cumulative_usd": self.total() + float(cost_usd),
        }
        if prompt is not None:
            entry["prompt"] = prompt
        if reply is not None:
            entry["reply"] = reply
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def session_total(self, session_id):
        return float(
            sum(
                float(e.get("cost_usd", 0.0))
                for e in self.entries()
                if e.get("session_id") == session_id
            )
        )

    def session_call_count(self, session_id):
        return sum(1 for e in self.entries() if e.get("session_id") == session_id)


def _extract_reply_text(payload):
    texts = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content", []) or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("output_text", "text") and part.get("text"):
                texts.append(str(part["text"]))
    if texts:
        return "\n".join(texts)
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    raise ValueError("no output text found in model response: %r" % (payload,)[:500])


def _post(payload, ledger, session_id, purpose, prompt_record, reply_record):
    """Post one request, record it, and return the parsed payload."""
    if ledger.total() >= BUDGET_CAP_USD:
        raise BudgetExceeded(
            "spend cap of $%.2f reached (total $%.4f); not calling the model"
            % (BUDGET_CAP_USD, ledger.total())
        )
    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("MODEL_API_KEY environment variable is not set")
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": "Bearer %s" % api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))
    usage = result.get("usage") or {}
    cost_usd = compute_cost_usd(usage)
    ledger.record(
        session_id,
        purpose,
        payload.get("model", MODEL_ID),
        usage,
        cost_usd,
        prompt=prompt_record,
        reply=reply_record(result),
    )
    return result, usage, cost_usd


def ask_ember(prompt, ledger, session_id, purpose):
    """Call Muse Ember and return (reply_text, usage, cost_usd)."""
    if not isinstance(ledger, SpendLedger):
        ledger = SpendLedger(ledger)
    payload = {"model": MODEL_ID, "input": prompt}
    result, usage, cost_usd = _post(
        payload,
        ledger,
        session_id,
        purpose,
        prompt_record=prompt,
        reply_record=lambda r: _extract_reply_text(r),
    )
    return _extract_reply_text(result), usage, cost_usd


def extract_function_calls(payload):
    """Return the model's function_call items from a response."""
    calls = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        raw_arguments = item.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "function_call arguments are not valid JSON: %s" % exc
                ) from exc
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            raise ValueError("function_call arguments must be a JSON object")
        calls.append(
            {
                "call_id": item.get("call_id") or item.get("id"),
                "name": item.get("name"),
                "arguments": arguments,
                "item": item,
            }
        )
    return calls


def function_call_output(call_id, result):
    """Build a function_call_output input item carrying a tool result."""
    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": result,
    }


class ResponsesClient:
    """Responses client carrying tools and the response chain.

    Each call sends input items (messages, and
    function_call_output items on later turns), chains turns
    with previous_response_id, and returns a dict with the
    response id, the raw output items, any text, and the
    function_call items.  Every call is recorded in the ledger.
    """

    def __init__(self, ledger, model=MODEL_ID, tools=None):
        if not isinstance(ledger, SpendLedger):
            ledger = SpendLedger(ledger)
        self.ledger = ledger
        self.model = model
        self.tools = list(tools or [])

    def call(self, input_items, session_id, purpose, tools=None, previous_response_id=None):
        payload = {"model": self.model, "input": list(input_items)}
        effective_tools = list(tools) if tools is not None else list(self.tools)
        if effective_tools:
            payload["tools"] = effective_tools
            payload["tool_choice"] = "auto"
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        def _reply_record(result):
            return json.dumps(
                {
                    "response_id": result.get("id"),
                    "output": result.get("output", []),
                    "output_text": result.get("output_text"),
                },
                ensure_ascii=False,
            )

        result, usage, cost_usd = _post(
            payload,
            self.ledger,
            session_id,
            purpose,
            prompt_record=json.dumps(payload, ensure_ascii=False),
            reply_record=_reply_record,
        )
        texts = []
        for item in result.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text") and part.get("text"):
                    texts.append(str(part["text"]))
        return {
            "response_id": result.get("id"),
            "output": result.get("output", []),
            "text": "\n".join(texts) if texts else result.get("output_text"),
            "function_calls": extract_function_calls(result),
            "usage": usage,
            "cost_usd": cost_usd,
            "payload": result,
        }


def call_responses(ledger, session_id, purpose, input_items, tools=None, previous_response_id=None):
    """One chained Responses call; returns the client's result dict."""
    return ResponsesClient(ledger, tools=tools).call(
        input_items,
        session_id,
        purpose,
        previous_response_id=previous_response_id,
    )
