"""MCP connector that turns every resident model into a tool.

This is what lets one model delegate to another. A generalist that is mediocre at code can
call `list_models`, see that a coding specialist is resident, and hand the problem over
with `ask_model` -- without anything having to load, because both are already in VRAM.

Register it in `mcp_servers.json`:

    {"name": "models", "transport": "stdio",
     "command": "<repo>/.venv/Scripts/python.exe",
     "args": ["<repo>/ui/backend/connectors/model_router.py"],
     "enabled": true}

It talks to the control plane over loopback, so it inherits the routing that already
dispatches a request to whichever engine holds the named model.

**It does not load anything.** Deliberately: a tool call that silently triggers a
twelve-minute model load would look like a hang, and an agent retrying it would queue
several. Models are loaded from the console; this only uses what is already resident.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("model-router")

CONTROL_PLANE = os.getenv("FREESWARM_API_URL", "http://127.0.0.1:8000")
# A bearer token, when the control plane has accounts configured.
TOKEN = os.getenv("FREESWARM_API_TOKEN", "")

# A delegated answer can legitimately take minutes on a large offloaded model.
CALL_TIMEOUT_S = 600


def _request(path: str, payload: dict | None = None, timeout: int = 30):
    url = f"{CONTROL_PLANE}{path}"
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("detail", body)
        except ValueError:
            detail = body
        raise RuntimeError(f"control plane {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"cannot reach the FreeToken control plane at {CONTROL_PLANE}: {exc.reason}"
        ) from None


@mcp.tool()
def list_models() -> str:
    """List the models currently loaded and ready to answer, with what each is good at.

    Call this FIRST when a task might suit a different model than the one you are. Each
    entry gives the exact name to pass to `ask_model`. Only resident models are listed —
    anything not here cannot be used without a human loading it.
    """
    doc = _request("/api/engines")
    loaded = doc.get("loaded", [])
    if not loaded:
        return "No models are currently loaded."

    roles = _request("/api/model-roles").get("roles", {})
    lines = []
    for entry in loaded:
        name = entry.get("model") or entry.get("served_name") or "?"
        ready = "ready" if entry.get("ready") else f"loading ({entry.get('state')})"
        role = roles.get(name) or roles.get(entry.get("served_name") or "")
        line = f"- {name} [{ready}]"
        if role:
            line += f"\n    good at: {role}"
        lines.append(line)
    return (
        "Models resident right now:\n"
        + "\n".join(lines)
        + "\n\nPass the exact name above as `model` to ask_model."
    )


@mcp.tool()
def ask_model(model: str, prompt: str, system: str = "", max_tokens: int = 3000) -> str:
    """Ask another loaded model a question and return its answer.

    Use this to delegate work a different resident model handles better — for example a
    coding question to a code-specialised model. `model` must be a name from
    `list_models`; nothing is loaded on demand.

    Give `prompt` everything needed to answer standalone: the other model sees only what
    you send, not your conversation.

    `max_tokens` covers the delegate's internal reasoning as well as its answer, so a
    reasoning model needs real headroom -- a cap of 1024 is routinely spent thinking
    before the answer even begins.
    """
    if not model or not model.strip():
        return "error: `model` is required — call list_models to see what is loaded."
    if not prompt or not prompt.strip():
        return "error: `prompt` is empty."

    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        result = _request(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": messages,
                "max_tokens": max(1, min(int(max_tokens), 8192)),
                "stream": False,
            },
            timeout=CALL_TIMEOUT_S,
        )
    except RuntimeError as exc:
        # Surfaced to the calling model as text so it can recover — usually by calling
        # list_models and retrying with a name that is actually resident.
        return f"error: {exc}"

    choices = result.get("choices") or []
    if not choices:
        return "error: the model returned no choices."

    choice = choices[0]
    message = choice.get("message") or {}
    content = (message.get("content") or "").strip()
    # Reasoning models emit their chain of thought on a SEPARATE channel before any
    # content. Hit the cap mid-thought and `content` is empty while `reasoning_content`
    # holds everything -- reading only `content` reports "(empty response)" for a model
    # that in fact did the work, and the caller loops retrying it.
    reasoning = (message.get("reasoning_content") or "").strip()
    usage = result.get("usage") or {}
    used = usage.get("completion_tokens", 0)
    footer = f"\n\n[{model}: {used} tokens]" if usage else ""

    if content:
        return content + footer

    if reasoning:
        if choice.get("finish_reason") == "length":
            # Actionable, so the caller can retry with headroom instead of looping.
            return (
                f"error: {model} used all {used} tokens on internal reasoning and never "
                f"reached an answer. Retry with a larger max_tokens "
                f"(try {max(3000, used * 2)}), or ask a narrower question."
            )
        return f"(reasoning only, no final answer)\n\n{reasoning[:2000]}" + footer

    return (
        f"error: {model} returned nothing "
        f"(finish_reason={choice.get('finish_reason')})."
    )


@mcp.tool()
def model_status() -> str:
    """Report each engine: which model, which GPU, and whether it is ready.

    Useful when a delegation fails — it distinguishes "still loading" from "not loaded".
    """
    doc = _request("/api/engines")
    engines = doc.get("engines", [])
    if not engines:
        return "No engines are running."
    rows = []
    for e in engines:
        rows.append(
            f"- {e.get('model_id') or '(none)'} | instance {e.get('instance_id')} "
            f"| GPU {e.get('gpus')} | port {e.get('port')} | {e.get('state')}"
            + (f" | error: {e.get('error')}" if e.get("error") else "")
        )
    return "\n".join(rows)


if __name__ == "__main__":
    mcp.run()
