"""Async SiftText MCP client over StreamableHTTP — with perf instrumentation."""

import asyncio
import json
import os
import re
import time

import httpx

from perf import perf

ENDPOINT = "https://app.sifttext.com/mcp"
PROTOCOL_VERSION = "2025-03-26"
PROPAGATION = "no_propagation: automated document decomposition"

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _parse_sse(text: str) -> dict:
    """Parse SSE response text into JSON data from the last 'message' event."""
    data_lines = []
    for line in text.splitlines():
        if line.startswith("data: "):
            data_lines.append(line[6:])
    if data_lines:
        return json.loads(data_lines[-1])
    raise RuntimeError(f"No SSE data lines found in response: {text[:200]}")


class SiftTextClient:
    """Thin async client for SiftText MCP server."""

    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("SIFTTEXT_API_KEY")
        if not api_key:
            raise SystemExit("Error: Set SIFTTEXT_API_KEY environment variable")
        self._api_key = api_key
        self._http = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            timeout=60.0,
        )
        self._session_id: str | None = None
        self._initialized = False
        self._request_id = 0
        self._init_lock = asyncio.Lock()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _post(self, payload: dict) -> dict | None:
        """POST to MCP endpoint. Returns parsed JSON-RPC result, or None for notifications."""
        headers = {}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        t0 = time.time()
        resp = await self._http.post(ENDPOINT, json=payload, headers=headers)
        http_ms = (time.time() - t0) * 1000

        resp.raise_for_status()

        # Log raw HTTP round-trip
        method = payload.get("method", "unknown")
        tool = ""
        if method == "tools/call":
            tool = payload.get("params", {}).get("name", "")
        perf.event("mcp_http", http_ms, method=method, tool=tool)

        # Capture session ID
        sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
        if sid:
            self._session_id = sid

        # Notifications get 202/204 with no body
        if resp.status_code in (202, 204) or not resp.text.strip():
            return None

        # Parse SSE or plain JSON
        ct = resp.headers.get("content-type", "")
        if "text/event-stream" in ct:
            return _parse_sse(resp.text)
        return resp.json()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            # Step 1: initialize handshake
            init_payload = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "hackathon-agent", "version": "0.1.0"},
                },
                "id": self._next_id(),
            }
            await self._post(init_payload)

            # Step 2: send initialized notification (no id, no response expected)
            notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
            await self._post(notif)
            self._initialized = True

    async def _call_tool(self, tool_name: str, **kwargs) -> str:
        """Call an MCP tool and return the text result."""
        await self._ensure_initialized()
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": kwargs},
            "id": self._next_id(),
        }
        t0 = time.time()
        data = await self._post(payload)
        dur = (time.time() - t0) * 1000
        perf.event("mcp_tool", dur, tool=tool_name)

        if not data:
            return ""
        # Extract text from first content block
        content = data.get("result", {}).get("content", [])
        if content and content[0].get("text"):
            return content[0]["text"]
        return json.dumps(data)

    # --- Convenience methods ---

    async def list_trees(self) -> str:
        return await self._call_tool("ideation_list_trees")

    async def create_tree(self, name: str, scope: str) -> dict:
        """Create a tree with confirm=True. Returns {tree_id, root_id}."""
        result = await self._call_tool("ideation_create_tree", name=name, scope=scope, confirm=True)

        # Extract tree_id from result text
        tree_match = _UUID_RE.search(result)
        if not tree_match:
            raise RuntimeError(f"Could not parse tree_id from create_tree response: {result}")
        tree_id = tree_match.group(0)

        # Get root_id from outline
        outline = await self.get_outline(tree_id, max_depth=1)
        root_match = _UUID_RE.search(outline)
        if not root_match:
            raise RuntimeError(f"Could not parse root_id from outline: {outline}")
        root_id = root_match.group(0)

        return {"tree_id": tree_id, "root_id": root_id}

    async def create_node(
        self,
        name: str,
        scope: str,
        parent_id: str,
        tree_id: str | None = None,
        propagation_check: str = PROPAGATION,
        crystallization: str | None = None,
        pipeline_mode: bool = False,
    ) -> str:
        """Create a child node. Returns the result text."""
        kwargs: dict = {
            "name": name,
            "scope": scope,
            "parent_id": parent_id,
            "propagation_check": propagation_check,
        }
        if tree_id:
            kwargs["tree_id"] = tree_id
        if crystallization:
            kwargs["crystallization"] = crystallization
        if pipeline_mode:
            kwargs["pipeline_mode"] = True
        return await self._call_tool("ideation_create_node", **kwargs)

    async def get_outline(self, tree_id: str, max_depth: int | None = None, node_id: str | None = None) -> str:
        kwargs: dict = {"tree_id": tree_id}
        if max_depth is not None:
            kwargs["max_depth"] = max_depth
        if node_id:
            kwargs["node_id"] = node_id
        return await self._call_tool("ideation_get_outline", **kwargs)

    async def get_node(self, node_id: str) -> str:
        return await self._call_tool("ideation_get_node", node_id=node_id)

    async def search(self, query: str, tree_id: str) -> str:
        return await self._call_tool("ideation_search", query=query, tree_id=tree_id)

    async def link_by_name(
        self,
        source_node_id: str,
        target_name: str,
        description: str,
        propagation_check: str = PROPAGATION,
        target_node_id: str | None = None,
        pipeline_mode: bool = False,
    ) -> str:
        kwargs: dict = {
            "source_node_id": source_node_id,
            "target_name_query": target_name,
            "description": description,
            "propagation_check": propagation_check,
        }
        if target_node_id:
            kwargs["target_node_id"] = target_node_id
        if pipeline_mode:
            kwargs["pipeline_mode"] = True
        return await self._call_tool("ideation_link_by_name", **kwargs)

    async def crystallize_append(
        self,
        node_id: str,
        text: str,
        propagation_check: str = PROPAGATION,
        pipeline_mode: bool = False,
    ) -> str:
        kwargs: dict = {
            "node_id": node_id,
            "crystallization": text,
            "propagation_check": propagation_check,
        }
        if pipeline_mode:
            kwargs["pipeline_mode"] = True
        return await self._call_tool("ideation_crystallize_append", **kwargs)

    async def close(self) -> None:
        await self._http.aclose()
