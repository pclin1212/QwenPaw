# -*- coding: utf-8 -*-
"""Transport abstraction for sandbox tool RPC.

The transport layer separates "how bytes are exchanged with the sandbox"
from "what gets exchanged".  This lets us start with HTTP/JSON and later
swap in stdio JSON-RPC, Unix sockets, or gRPC without touching any
business logic.

Protocol contract (transport-agnostic):
    request:  {"name": str, "arguments": dict}
    response: {"ok": bool, "content": list[ContentBlock], "error": str | None}

Where ContentBlock matches agentscope's ToolResultBlock content shape:
    [{"type": "text", "text": "..."}, {"type": "image", "url": "..."}, ...]
"""
from __future__ import annotations

import abc
import asyncio
import logging
from typing import Any

import httpx


logger = logging.getLogger(__name__)


class ToolTransport(abc.ABC):
    """Abstract transport interface."""

    @abc.abstractmethod
    async def call(self, tool_name: str, arguments: dict) -> dict:
        """Send one tool invocation, return decoded response dict."""

    @abc.abstractmethod
    async def list_tools(self) -> list:
        """Return list of tool names available in the sandbox."""

    @abc.abstractmethod
    async def health(self) -> bool:
        """Return True if the sandbox tool server is reachable."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release transport resources."""


class HttpTransport(ToolTransport):
    """HTTP/JSON transport. The simplest, most debuggable option.

    The sandbox runs a FastAPI server on a port that's published to the host;
    we POST tool invocations to /tools/call.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        connect_timeout: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
        )

    async def call(self, tool_name: str, arguments: dict) -> dict:
        import time
        url = f"{self.base_url}/tools/call"
        logger.info("[sandbox-rpc] POST %s tool=%s", url, tool_name)
        t0 = time.monotonic()
        try:
            resp = await self._http.post(
                url,
                json={"name": tool_name, "arguments": arguments},
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            resp.raise_for_status()
            logger.info(
                "[sandbox-rpc] tool=%s status=%d elapsed=%.1fms",
                tool_name, resp.status_code, elapsed_ms,
            )
            return resp.json()
        except httpx.HTTPStatusError as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "[sandbox-rpc] tool=%s status=%d elapsed=%.1fms HTTPStatusError",
                tool_name, e.response.status_code, elapsed_ms,
            )
            try:
                body = e.response.json()
                if isinstance(body, dict) and "error" in body:
                    return {"ok": False, "content": [], "error": body["error"]}
            except Exception:
                pass
            return {
                "ok": False,
                "content": [],
                "error": f"sandbox http {e.response.status_code}: {e.response.text[:500]}",
            }
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "[sandbox-rpc] tool=%s transport_error=%s elapsed=%.1fms",
                tool_name, type(e).__name__, elapsed_ms,
            )
            return {
                "ok": False,
                "content": [],
                "error": f"sandbox transport error: {type(e).__name__}: {e}",
            }

    async def list_tools(self) -> list:
        resp = await self._http.get(f"{self.base_url}/tools/list")
        resp.raise_for_status()
        data = resp.json()
        return list(data.get("tools", []))

    async def health(self) -> bool:
        try:
            resp = await self._http.get(f"{self.base_url}/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def wait_ready(
        self, max_wait_seconds: float = 30.0, interval: float = 0.5
    ) -> bool:
        """Block until /health returns 200 or timeout elapses."""
        elapsed = 0.0
        while elapsed < max_wait_seconds:
            if await self.health():
                return True
            await asyncio.sleep(interval)
            elapsed += interval
        return False

    async def close(self) -> None:
        await self._http.aclose()
