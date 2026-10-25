"""Character research through a user-supplied MCP server.

The app is the MCP client and the tool-loop host: it offers the server's tools to
the LLM as ordinary function definitions, executes the calls the model asks for,
and feeds the results back until the model stops. That keeps it independent of
the LLM host - llama-server, LM Studio and Ollama all work, because it is plain
OpenAI function calling - and of the search provider, because the provider is
whatever MCP server the user added.

Why the loop lives here rather than in the LLM server: everything the model reads
passes through this module, so the trace can be stored and shown. A description
that arrived without a trace is a description nobody can check.
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable, Dict, Iterable, List, Optional

from services import mcp_client

MAX_ROUNDS = 6
MAX_TOOL_CHARS = 7000

# Only the fields the renderer compiles. ``attire`` and ``carried`` are
# deliberately absent: they are scene props, and a reference source describes
# them for a moment in the story rather than for the character.
LOOKUP_FIELDS = ("sex", "species", "age", "height", "build", "hair_color",
                 "hair_style", "eye_color", "face", "marks")

_SYSTEM = (
    "You research the physical appearance of a fictional character for an "
    "illustration pipeline. Use the tools provided to find reference material.\n"
    "Rules:\n"
    "- Search at least once. If the search snippets do not themselves contain an "
    "appearance description, open a result page with the page-reading tool before "
    "answering. A snippet is a summary, not a source.\n"
    "- Report only what your sources state. Never guess, never fill a gap with "
    "what is typical for the setting, never use your own memory of the story.\n"
    "- Prefer sources that describe the book over screen adaptations.\n"
    "- If the sources describe nothing for a field, use null.\n"
    "- Reply with JSON only, no prose and no code fence.\n"
    'Schema: {"fields": {"sex": ..., "species": ..., "age": ..., "height": ..., '
    '"build": ..., "hair_color": ..., "hair_style": ..., "eye_color": ..., '
    '"face": ..., "marks": ...}, "sources": ["url", ...]}\n'
    "Field shapes:\n"
    "- sex: male or female.\n"
    "- species: lowercase noun; null for an ordinary human.\n"
    "- age: child / teen / young / adult / middle-aged / old, only if the source says so.\n"
    "- height: as stated, e.g. 5'10\" or close to ten feet.\n"
    "- build: adjectives only, e.g. lean, stocky, broad-shouldered.\n"
    "- hair_color and eye_color: colour words only.\n"
    "- hair_style: a short phrase, e.g. long braid, cropped, shoulder-length.\n"
    "- face: facial features or complexion.\n"
    "- marks: scars, tattoos, unusual features, missing parts.\n"
    "sources: the URLs you actually opened or read, not the ones you merely saw "
    "in results.\n"
    'If no tool succeeds, reply {"error": "no_web_access"}.'
)

_URL = re.compile(r"https?://[^\s\"'<>)\]]+")


def enabled_tools(tools: Iterable[Dict], prefs: Optional[Dict[str, bool]]) -> List[Dict]:
    """Tools the user left ticked. An unlisted tool is enabled (first connect)."""
    prefs = prefs or {}
    return [t for t in tools if prefs.get(t["name"], True)]


def _spec(tool: Dict) -> Dict:
    """MCP tool -> OpenAI function definition. The schemas are the same shape."""
    schema = tool.get("inputSchema") or {}
    if not isinstance(schema, dict) or schema.get("type") != "object":
        schema = {"type": "object", "properties": {}}
    return {"type": "function",
            "function": {"name": tool["name"],
                         "description": (tool.get("description") or "")[:1024],
                         "parameters": schema}}


def _parse_json(text: str) -> Optional[Dict]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _clean_fields(raw: Dict) -> Dict[str, str]:
    """Trim to our field set, drop empties, normalise the values we can."""
    out: Dict[str, str] = {}
    for field in LOOKUP_FIELDS:
        value = raw.get(field)
        if value is None:
            continue
        text = " ".join(str(value).split()).strip(" .,;")
        if not text or text.lower() in ("null", "none", "unknown", "n/a"):
            continue
        out[field] = text
    return out


_COLOUR_WORDS = {
    "black", "white", "gray", "grey", "greyish", "grayish", "silver", "silvery",
    "brown", "brownish", "blond", "blonde", "blondish", "red", "reddish",
    "auburn", "ginger", "golden", "gold", "copper", "chestnut", "honey",
    "sandy", "ash", "ashen", "raven", "ebony", "ivory", "amber", "hazel",
    "green", "greenish", "blue", "bluish", "yellow", "yellowish", "violet",
    "purple", "orange", "pink", "russet", "tawny", "wheaten", "straw",
    "strawberry", "platinum", "snowy", "salt", "pepper", "dark", "light",
    "pale", "olive", "bronze", "tan", "cream", "charcoal", "mousy", "flaxen",
}


def _is_colour(value: str) -> bool:
    """Whether a colour field holds a colour.

    Prose uses eyes for everything except colour - "his eyes burned", "her eyes
    widened" - so an unvalidated field collects verbs. Keeping only values built
    from colour words costs nothing and removes a whole class of junk.
    """
    tokens = re.findall(r"[a-z]+", (value or "").lower())
    return any(token in _COLOUR_WORDS for token in tokens)


def short_phrase(value: str, limit: int = 90) -> str:
    """One clause of a build, because a sentence of it renders as a paragraph.

    Asked for adjectives the model usually gives adjectives, but a companion
    volume's "About 5'7\" tall, a little shorter than Aviendha, she was quite
    tall for an Andoran woman. Her brother Gawyn was a head taller than she."
    arrives as build, and every word of it ends up in the image prompt. The first
    sentence carries the usable half; the rest is comparison.
    """
    text = " ".join((value or "").split()).strip(" ,;")
    text = re.sub(r"^(?:about|around|approximately|roughly)\s+", "", text, flags=re.I)
    text = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)[0]
    if len(text) > limit:
        cut = text[:limit]
        comma = cut.rfind(",")
        text = (cut[:comma] if comma > 30 else cut).rstrip(" ,;")
    return text.strip(" ,;")


def sanitise(fields: Dict[str, str]) -> Dict[str, str]:
    """Force the shapes the renderer expects.

    The model is told these shapes and mostly obeys; the few that matter are
    enforced here because a value like ``eye_color="large dark eyes"`` composes
    into "a woman with large dark eyes eyes" at render time.
    """
    out = dict(fields)
    for field in ("hair_color", "eye_color"):
        value = out.get(field)
        if value:
            value = re.sub(r"\b(colou?red|hair|eyes?)\b", " ", value, flags=re.I)
            value = " ".join(value.split()).strip(" ,;")
            if value and _is_colour(value):
                out[field] = value
            else:
                out.pop(field, None)
    for field in ("build", "height"):
        value = out.get(field)
        if not value:
            continue
        value = short_phrase(value)
        if value:
            out[field] = value
        else:
            out.pop(field, None)
    sex = out.get("sex", "").strip().lower()
    if sex in ("m", "male"):
        out["sex"] = "male"
    elif sex in ("f", "female"):
        out["sex"] = "female"
    elif sex:
        out.pop("sex", None)
    species = out.get("species", "").strip().lower()
    if species in ("human", "humans", "mankind"):
        out.pop("species", None)
    return out


def _tool_trace(observed: List[str], claimed: Iterable[str]) -> List[str]:
    """Sources the trace supports. A model may cite a page it never opened."""
    seen = {url.rstrip(".,;)") for url in observed}
    out = []
    for url in claimed:
        clean = str(url).strip().rstrip(".,;)")
        if clean in seen and clean not in out:
            out.append(clean)
    return out


async def lookup_character(
    name: str,
    series: str = "",
    *,
    mcp_url: str,
    mcp_auth: str = "",
    tool_prefs: Optional[Dict[str, bool]] = None,
    llm_url: str,
    model: str,
    query: str = "",
    should_cancel: Optional[Callable[[], bool]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Research one character. Never raises for a source failure, returns ``ok``.

    The result carries ``fields``, the ``sources`` the trace actually supports,
    and a per-round ``trace`` of tool calls. ``error`` is set when nothing could
    be looked up, so a misconfigured server is reported instead of being written
    down as "this character has no description".

    ``query`` is a user-edited first search. It is offered rather than commanded -
    the model rewords it when it returns nothing - but it is the only way to give
    a search a specific book title, a disambiguating surname or a quoted phrase
    without editing the prompt.
    """
    from services.prompt_engine import get_llm_response

    started = time.time()
    # One connection for the whole lookup: the handshake is three round trips,
    # and some servers bind tool state to the session id.
    client = mcp_client.McpHttpClient(mcp_url, mcp_auth)
    try:
        return await _lookup(client, name, series, tool_prefs=tool_prefs,
                             llm_url=llm_url, model=model, query=query,
                             should_cancel=should_cancel, log=log,
                             started=started)
    finally:
        client.close()


async def _lookup(client, name: str, series: str, *, tool_prefs,
                  llm_url: str, model: str, query: str = "", should_cancel, log,
                  started: float) -> Dict:
    from services.prompt_engine import get_llm_response

    def say(message: str) -> None:
        if log:
            log(message)

    try:
        client.connect()
        tools = enabled_tools(client.list_tools(), tool_prefs)
    except mcp_client.McpError as ex:
        return {"ok": False, "error": str(ex), "fields": {}, "sources": [],
                "trace": [], "seconds": round(time.time() - started, 1)}

    if not tools:
        return {"ok": False, "error": "no MCP tools are enabled",
                "fields": {}, "sources": [], "trace": [],
                "seconds": round(time.time() - started, 1)}

    offered = [_spec(tool) for tool in tools]
    messages: List[Dict] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"Character: {name}\n"
            + (f"Book series: {series}\n" if series else "")
            + (f"Start with this search query, verbatim: {query}\n"
               if query else "")
            + "Find a physical description of this character.")},
    ]
    trace: List[Dict] = []
    observed_urls: List[str] = []
    calls_made: Dict[str, int] = {}

    for round_no in range(1, MAX_ROUNDS + 1):
        if should_cancel and should_cancel():
            return {"ok": False, "error": "cancelled", "fields": {},
                    "sources": [], "trace": trace,
                    "seconds": round(time.time() - started, 1)}
        try:
            message, finish = await get_llm_response(
                "", llm_url, model, temperature=0.2, timeout=180,
                messages=messages, with_message=True,
                extra_payload={"tools": offered, "tool_choice": "auto",
                               "max_tokens": 700})
        except Exception as ex:
            return {"ok": False, "error": f"LLM call failed: {ex}", "fields": {},
                    "sources": [], "trace": trace,
                    "seconds": round(time.time() - started, 1)}

        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        if finish != "tool_calls" and not tool_calls:
            data = _parse_json(content) or {}
            if data.get("error"):
                return {"ok": False, "error": str(data["error"]), "fields": {},
                        "sources": [], "trace": trace,
                        "seconds": round(time.time() - started, 1)}
            fields = sanitise(_clean_fields(data.get("fields") or {}))
            claimed = data.get("sources") or []
            if isinstance(claimed, str):
                claimed = [claimed]
            return {"ok": True, "fields": fields,
                    "sources": _tool_trace(observed_urls, claimed),
                    "trace": trace, "rounds": round_no,
                    "seconds": round(time.time() - started, 1),
                    "raw": content}

        messages.append({"role": "assistant", "content": content,
                         "tool_calls": tool_calls})
        for call in tool_calls:
            function = call.get("function") or {}
            tool_name = function.get("name") or ""
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            key = json.dumps([tool_name, args], sort_keys=True)
            calls_made[key] = calls_made.get(key, 0) + 1
            if calls_made[key] > 1:
                # A 12B will happily repeat the same query; answering it again
                # burns tokens and never adds information.
                output = ("You already made exactly this call. Use a different "
                          "query or conclude with the JSON.")
                is_error = True
            else:
                try:
                    output, is_error = client.call_tool(tool_name, args)
                except mcp_client.McpError as ex:
                    output, is_error = f"ERROR: {ex}", True
            observed_urls.extend(_URL.findall(output))
            trace.append({"round": round_no, "tool": tool_name,
                          "args": args, "chars": len(output),
                          "error": is_error})
            say(f"[Lookup] {name}: {tool_name} -> {len(output)} chars"
                + (" (error)" if is_error else ""))
            messages.append({"role": "tool", "tool_call_id": call.get("id"),
                             "content": output[:MAX_TOOL_CHARS]})

    return {"ok": False, "error": "the model did not conclude", "fields": {},
            "sources": [], "trace": trace,
            "seconds": round(time.time() - started, 1)}
