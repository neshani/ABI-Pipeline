"""Minimal MCP client over the Streamable-HTTP transport.

Why this exists: the app researches characters from reference material, and the
best sources are behind a remote MCP server the user already runs (a SearXNG
wrapper, Brave, Tavily, Firecrawl...). Making the *app* the MCP client means any
such server works, with no configuration on the LLM host: the app offers the
server's tools to the model as ordinary OpenAI function definitions and executes
the calls itself.

Only the Streamable-HTTP transport is implemented. stdio servers (a child
process) are not, deliberately: spawning arbitrary commands from a settings
field is a security hole, and a URL covers every hosted service.

The protocol bits that matter here:
  * ``initialize`` must be answered with the negotiated ``protocolVersion``, and
    a ``notifications/initialized`` notification follows it. Neither is optional -
    a server may refuse ``tools/list`` before the handshake completes.
  * The server may pin the conversation with an ``Mcp-Session-Id`` response
    header, which must be echoed on every later request.
  * Responses may be ``application/json`` or ``text/event-stream``. The SSE form
    is a set of ``data:`` lines, one JSON-RPC message each; the reply to a request
    is the one whose ``id`` matches.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import httpx

PROTOCOL_VERSION = "2025-06-18"
_FALLBACK_PROTOCOL = "2024-11-05"

_CONNECT_TIMEOUT = 25.0
_CALL_TIMEOUT = 90.0


class McpError(RuntimeError):
    """A connection or protocol failure, with a message the UI can show."""


def parse_auth(value: str) -> Dict[str, str]:
    """Turn the auth setting into request headers.

    Accepts either a bare token (sent as ``Authorization: Bearer <token>``) or a
    full ``Name: value`` pair, so a service that wants ``X-API-Key`` works too.
    """
    value = (value or "").strip()
    if not value:
        return {}
    if ":" in value.split(" ")[0]:
        name, _, rest = value.partition(":")
        return {name.strip(): rest.strip()}
    return {"Authorization": f"Bearer {value}"}


def _select_reply(payload, request_id: Optional[int]):
    """The JSON-RPC reply for ``request_id`` from a body or an SSE frame list."""
    messages = payload if isinstance(payload, list) else [payload]
    for message in messages:
        if not isinstance(message, dict):
            continue
        if request_id is None or message.get("id") == request_id:
            return message
    return messages[-1] if messages else None


class McpHttpClient:
    """One connection to one MCP server. Use as a context manager."""

    def __init__(self, url: str, auth: str = ""):
        self.url = (url or "").strip()
        if not self.url:
            raise McpError("No MCP server URL is configured.")
        self._headers = {"Content-Type": "application/json",
                         "Accept": "application/json, text/event-stream",
                         **parse_auth(auth)}
        self._session_id = ""
        self._next_id = 1
        self._client = httpx.Client(timeout=_CALL_TIMEOUT, follow_redirects=True)
        self.server_info: Dict = {}
        self.protocol: str = ""

    # -- transport ---------------------------------------------------------
    def __enter__(self) -> "McpHttpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def _post(self, payload: Dict, timeout: float) -> Optional[Dict]:
        headers = dict(self._headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            response = self._client.post(self.url, json=payload, headers=headers,
                                         timeout=timeout)
        except httpx.HTTPError as ex:
            raise McpError(f"could not reach {self.url} ({type(ex).__name__})") from ex
        session = response.headers.get("mcp-session-id")
        if session:
            self._session_id = session
        if response.status_code >= 400:
            detail = " ".join((response.text or "").split())[:200]
            raise McpError(f"HTTP {response.status_code} from {self.url}: {detail}")
        if not response.content:
            return None
        ctype = (response.headers.get("content-type") or "").lower()
        if "text/event-stream" in ctype:
            frames = []
            for line in response.text.splitlines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk or chunk == "[DONE]":
                    continue
                try:
                    frames.append(json.loads(chunk))
                except json.JSONDecodeError:
                    continue
            return _select_reply(frames, payload.get("id"))
        try:
            return _select_reply(response.json(), payload.get("id"))
        except json.JSONDecodeError as ex:
            raise McpError(f"{self.url} returned non-JSON: "
                           f"{response.text[:120]!r}") from ex

    def _request(self, method: str, params: Optional[Dict] = None,
                 timeout: float = _CALL_TIMEOUT) -> Dict:
        request_id = self._next_id
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        reply = self._post(message, timeout)
        if not reply:
            raise McpError(f"no reply to {method}")
        if reply.get("error"):
            error = reply["error"]
            raise McpError(f"{method} failed: "
                           f"{error.get('message') or error}")
        return reply.get("result") or {}

    def _notify(self, method: str) -> None:
        try:
            self._post({"jsonrpc": "2.0", "method": method}, timeout=_CONNECT_TIMEOUT)
        except McpError:
            pass  # notifications are fire-and-forget

    # -- protocol ----------------------------------------------------------
    def connect(self) -> Dict:
        """Handshake and return the server info. Raises McpError on failure."""
        try:
            result = self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "ABI-Pipeline", "version": "1.0"},
            }, timeout=_CONNECT_TIMEOUT)
        except McpError as first_error:
            # Older servers reject an unknown protocol version outright; one
            # retry on the previous revision costs nothing and widens support.
            if "protocol" not in str(first_error).lower():
                raise
            result = self._request("initialize", {
                "protocolVersion": _FALLBACK_PROTOCOL,
                "capabilities": {},
                "clientInfo": {"name": "ABI-Pipeline", "version": "1.0"},
            }, timeout=_CONNECT_TIMEOUT)
        self._notify("notifications/initialized")
        self.protocol = str(result.get("protocolVersion") or "")
        self.server_info = dict(result.get("serverInfo") or {})
        return self.server_info

    def list_tools(self) -> List[Dict]:
        """``[{name, description, inputSchema}]`` in the server's order."""
        tools: List[Dict] = []
        cursor = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = self._request("tools/list", params, timeout=_CONNECT_TIMEOUT)
            for tool in result.get("tools") or []:
                if isinstance(tool, dict) and tool.get("name"):
                    tools.append({
                        "name": str(tool.get("name")),
                        "description": str(tool.get("description") or ""),
                        "inputSchema": tool.get("inputSchema") or {},
                    })
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: Dict) -> Tuple[str, bool]:
        """``(text, is_error)`` for one tool call."""
        result = self._request("tools/call",
                               {"name": name, "arguments": arguments or {}})
        parts = []
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        if not parts and result:
            parts.append(json.dumps(result, ensure_ascii=False))
        return "\n".join(parts), bool(result.get("isError"))


def list_tools(url: str, auth: str = "") -> Dict:
    """Connect, list tools, disconnect. For the settings UI."""
    with McpHttpClient(url, auth) as client:
        info = client.connect()
        return {"server": info, "protocol": client.protocol,
                "tools": client.list_tools()}
