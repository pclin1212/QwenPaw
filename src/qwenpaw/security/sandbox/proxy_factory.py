# -*- coding: utf-8 -*-
"""Tool function proxy factory — the heart of the zero-wrapper sandbox.

Instead of writing a sandboxed_*() wrapper for each tool, we use Python
introspection to dynamically synthesize a proxy function that:
  1. Has the SAME signature, name, docstring, and type hints as the original.
  2. When called, forwards (name, kwargs) over RPC to the sandbox.
  3. Decodes the RPC response back into a ToolResponse instance.

Why preserve the signature? agentscope.Toolkit.register_tool_function()
inspects fn.__signature__ to build the JSON schema sent to the LLM.
A naive `def proxy(*args, **kwargs)` would lose all parameter info.

This means: adding a new tool to QwenPaw is zero work for the sandbox.
The new function lives in qwenpaw/agents/tools/, the sandbox image picks
it up automatically, and the proxy factory handles routing transparently.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import re
from typing import Any, Callable

from agentscope.message import (
    TextBlock,
    ToolResultBlock,
)

from .transport import ToolTransport


logger = logging.getLogger(__name__)


# Tools that fundamentally CANNOT run in the sandbox because they need
# host-side resources (browser, screen, native windowing, network outbound,
# inter-agent IPC, file delivery to user, etc.).  These keep their original
# implementations.  Adding a tool to this set is the ONLY per-tool change
# needed when introducing a new tool — and only if it's host-bound.
HOST_BOUND_TOOLS: frozenset[str] = frozenset({
    # GUI / display
    "browser_use",
    "browser_snapshot",
    "desktop_screenshot",
    "view_image",
    "view_video",
    # User-facing communication
    "send_file_to_user",
    # Multi-agent orchestration (lives in host's agent registry)
    "delegate_external_agent",
    "list_agents",
    "chat_with_agent",
    "submit_to_agent",
    "check_agent_task",
    "spawn_subagent",
    # Time / locale (host-perspective)
    "get_current_time",
    "set_user_timezone",
    # Token accounting (reads host model client state)
    "get_token_usage",
})


class PathTranslator:
    """Translate host paths to container paths in tool arguments.

    The sandbox container bind-mounts the workspace directory at /workspace.
    LLM-emitted absolute paths look like host paths (e.g. D:\\QwenPaw\\proj
    or /Users/x/proj on macOS); we rewrite them to /workspace/... before
    sending the call into the sandbox.
    """

    def __init__(self, host_workspace: str, container_workspace: str = "/workspace"):
        self.host_workspace = os.path.abspath(host_workspace)
        # Normalize to forward slashes for consistent matching on Windows
        self._host_norm = self.host_workspace.replace("\\", "/")
        self.container_workspace = container_workspace.rstrip("/")

    def translate(self, value: Any) -> Any:
        """Recursively rewrite host paths in any structure."""
        if isinstance(value, str):
            return self._translate_str(value)
        if isinstance(value, list):
            return [self.translate(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.translate(v) for v in value)
        if isinstance(value, dict):
            return {k: self.translate(v) for k, v in value.items()}
        return value

    def _translate_str(self, s: str) -> str:
        # Try both raw and normalized forms (Windows can have either)
        candidates = [self.host_workspace, self._host_norm]
        for prefix in candidates:
            if s == prefix:
                logger.info("[path-xlat] %r -> %r", s, self.container_workspace)
                return self.container_workspace
            if s.startswith(prefix + os.sep) or s.startswith(prefix + "/"):
                tail = s[len(prefix):].replace("\\", "/").lstrip("/")
                translated = f"{self.container_workspace}/{tail}"
                logger.info("[path-xlat] %r -> %r", s, translated)
                return translated
        return s



def _truncate_for_log(value: Any, max_len: int = 500) -> str:
    """Render value as a single-line string capped at max_len chars (for logs)."""
    try:
        text = repr(value)
    except Exception:
        text = "<unrepr>"
    if len(text) > max_len:
        text = text[:max_len] + f"...<+{len(text) - max_len}c>"
    return text


def _decode_response_to_tool_response(
    rpc_resp: dict,
    tool_name: str,
):
    """Convert RPC dict back into the ToolResponse type the host expects."""
    from agentscope.tool import ToolResponse

    if not rpc_resp.get("ok", False):
        err = rpc_resp.get("error") or "unknown sandbox error"
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=f"[sandbox error in {tool_name}] {err}",
                ),
            ],
        )

    content = rpc_resp.get("content") or []
    # The sandbox already serializes content blocks as plain dicts
    # (e.g. {"type": "text", "text": "..."}).  agentscope ToolResponse
    # accepts the dict form directly.
    return ToolResponse(content=content)


def make_sandbox_proxy(
    original_fn: Callable,
    tool_name: str,
    transport: ToolTransport,
    path_translator: PathTranslator | None,
) -> Callable:
    """Create an RPC proxy preserving the original function's surface.

    The returned function has identical __name__, __doc__, __signature__,
    and async-ness to the original — agentscope sees it as the same tool.
    """
    is_coro = asyncio.iscoroutinefunction(original_fn)
    sig = inspect.signature(original_fn)

    async def _async_proxy(**kwargs):
        # Bind kwargs to enforce signature (raise early on bad calls)
        bound = sig.bind_partial(**kwargs)
        bound.apply_defaults()
        args_dict = dict(bound.arguments)

        logger.info(
            "[sandbox-call] tool=%s raw_args=%s",
            tool_name, _truncate_for_log(args_dict),
        )

        if path_translator is not None:
            translated = path_translator.translate(args_dict)
            if translated != args_dict:
                logger.info(
                    "[sandbox-call] tool=%s args_after_xlat=%s",
                    tool_name, _truncate_for_log(translated),
                )
            args_dict = translated

        rpc_resp = await transport.call(tool_name, args_dict)
        ok = bool(rpc_resp.get("ok", False))
        if ok:
            logger.info(
                "[sandbox-call] tool=%s ok=True content_blocks=%d",
                tool_name, len(rpc_resp.get("content") or []),
            )
        else:
            logger.info(
                "[sandbox-call] tool=%s ok=False error=%s",
                tool_name, rpc_resp.get("error"),
            )
        return _decode_response_to_tool_response(rpc_resp, tool_name)

    # agentscope supports async tools natively; even sync originals can
    # become async proxies because the framework awaits when given a coro.
    proxy = _async_proxy

    # Critical: copy __name__, __doc__, signature so register_tool_function
    # sees the same surface it would have seen for the original.
    proxy.__name__ = original_fn.__name__
    proxy.__qualname__ = original_fn.__qualname__
    proxy.__doc__ = original_fn.__doc__
    proxy.__module__ = original_fn.__module__
    try:
        proxy.__signature__ = sig
        proxy.__annotations__ = dict(getattr(original_fn, "__annotations__", {}))
    except (AttributeError, TypeError):
        pass

    # Mark proxies so debugging/logging can identify them
    proxy.__qwenpaw_sandbox_proxy__ = True
    proxy.__qwenpaw_original__ = original_fn

    return proxy


def proxify_tool_dict(
    tool_functions: dict,
    transport: ToolTransport,
    path_translator: PathTranslator | None,
    host_bound: frozenset = HOST_BOUND_TOOLS,
) -> dict:
    """Replace every non-host-bound tool with its RPC proxy.

    This is the SINGLE integration point.  Adding a new tool requires zero
    changes here — it's automatically proxied by name.
    """
    proxied = 0
    skipped = 0
    out: dict = {}
    for name, fn in tool_functions.items():
        if name in host_bound:
            out[name] = fn
            skipped += 1
            continue
        try:
            out[name] = make_sandbox_proxy(fn, name, transport, path_translator)
            proxied += 1
        except (ValueError, TypeError) as e:
            # Some functions might be C-extensions or otherwise un-introspectable.
            # Fall back to the original; LLM still gets to use it on the host.
            logger.warning(
                "Cannot proxy tool %s (%s); using host implementation",
                name, e,
            )
            out[name] = fn
            skipped += 1

    logger.info(
        "Sandbox: proxied %d tools, kept %d host-bound tools",
        proxied, skipped,
    )
    return out
